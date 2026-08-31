# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import socket
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_remax_disagreement_mask,
    deduplicate_rollout_responses,
)

from verl.utils.metric import reduce_metrics
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.ray_utils import ray_host_url
from recipe.phimm.reward.error_book import get_eb


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        logger.log(data={"hostname": socket.gethostname(), "ray_url": ray_host_url()}, step=self.global_steps)
        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            print(f"Initial validation metrics: \n{val_metrics}", flush=True)
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        # breakpoint()
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                tensor_keys = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_keys = ["raw_prompt_ids", "multi_modal_data", "reward_model"]
                non_tensor_keys = [k for k in non_tensor_keys if k in new_batch.non_tensor_batch.keys()]
                gen_batch = new_batch.pop(batch_keys=tensor_keys, non_tensor_batch_keys=non_tensor_keys)

                gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    greedy_hyps = None
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["validate"] = True
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            # Decode greedy baseline responses and inject into extra_info
                            # so downstream reward functions (e.g. fmt_llm_judge_reward) can
                            # use them as comparison baselines.
                            greedy_responses = gen_baseline_output.batch["responses"]
                            greedy_hyps = self.tokenizer.batch_decode(greedy_responses, skip_special_tokens=True)

                            new_batch = new_batch.union(gen_baseline_output)

                            if "extra_info" not in new_batch.non_tensor_batch:
                                new_batch.non_tensor_batch["extra_info"] = np.array(
                                    [{} for _ in range(len(new_batch))], dtype=object
                                )
                            for i, hyp in enumerate(greedy_hyps):
                                ei = new_batch.non_tensor_batch["extra_info"][i]
                                if isinstance(ei, dict):
                                    ei["greedy_hyp"] = hyp
                                else:
                                    new_batch.non_tensor_batch["extra_info"][i] = {"greedy_hyp": hyp}

                            new_batch.meta_info["skip_examine"] = True
                            remax_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                            if remax_reward_keys:
                                # Multi-reward ReMax: capture per-dimension greedy baselines
                                # so each reward dimension can be decoupled in advantage calc.
                                baseline_result = self.reward_fn(new_batch, return_dict=True)
                                reward_baseline_tensor = baseline_result["reward_tensor"]
                                baseline_extra = baseline_result.get("reward_extra_info", {})
                                for key in remax_reward_keys:
                                    assert key in baseline_extra, (
                                        f"ReMax reward key '{key}' not found in greedy baseline "
                                        f"reward_extra_info. Available keys: {list(baseline_extra.keys())}."
                                    )
                                    new_batch.batch[f"reward_baselines_{key}"] = torch.tensor(
                                        np.asarray(baseline_extra[key], dtype=np.float32)
                                    )
                            else:
                                reward_baseline_tensor = self.reward_fn(new_batch)
                            new_batch.meta_info.pop("skip_examine", None)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            if self.config.algorithm.get("remax_mask", False):
                                rollout_n = self.config.actor_rollout_ref.rollout.n
                                sampled_resp = gen_batch_output.batch["responses"]
                                sampled_rlen = sampled_resp.size(1)
                                sampled_rmask = gen_batch_output.batch["attention_mask"][:, -sampled_rlen:]
                                baseline_resp = gen_baseline_output.batch["responses"]
                                baseline_rlen = baseline_resp.size(1)
                                baseline_rmask = gen_baseline_output.batch["attention_mask"][:, -baseline_rlen:]
                                baseline_index = np.arange(sampled_resp.size(0)) // rollout_n
                                gen_batch_output.batch["remax_mask"] = compute_remax_disagreement_mask(
                                    sampled_resp,
                                    sampled_rmask,
                                    baseline_resp,
                                    baseline_rmask,
                                    baseline_index=baseline_index,
                                )

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # Re-inject greedy_hyp into extra_info after repeat+union
                        # (repeat creates new arrays, union may overwrite non_tensor_batch).
                        if greedy_hyps is not None:
                            n_rollout = self.config.actor_rollout_ref.rollout.n
                            if "extra_info" not in new_batch.non_tensor_batch:
                                new_batch.non_tensor_batch["extra_info"] = np.array(
                                    [{} for _ in range(len(new_batch))], dtype=object
                                )
                            for i in range(len(new_batch)):
                                ei = new_batch.non_tensor_batch["extra_info"][i]
                                if not isinstance(ei, dict):
                                    ei = {}
                                    new_batch.non_tensor_batch["extra_info"][i] = ei
                                ei["greedy_hyp"] = greedy_hyps[i // n_rollout]

                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            assert all(
                                len(values) == len(new_batch) for values in reward_extra_infos_dict.values()
                            ), "Reward extras must be row-aligned before GDPO advantage computation."
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # Defer KL penalty to after old_log_probs and ref_log_prob are computed.
                        # For now, set token_level_rewards = token_level_scores (KL applied later).
                        new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]
                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:
                        # check zero std prompts
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )
                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)
                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)
                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                # progress_bar.update(1)
                                self.gen_steps += 1
                                is_last_step = self.global_steps >= self.total_training_steps
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Deduplicate identical rollout responses within each prompt group.
                    if self.config.trainer.get("dedup_responses", False):
                        dp_size = self.actor_rollout_wg.world_size
                        batch, n_total, n_kept, n_removed = deduplicate_rollout_responses(batch, dp_size=dp_size)
                        if n_removed > 0:
                            metrics["rollout/dedup_removed"] = n_removed
                            metrics["rollout/dedup_kept"] = n_kept
                            metrics["rollout/dedup_ratio"] = n_removed / n_total

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs (or reuse rollout_log_probs if configured)
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        use_rollout_as_old = getattr(self.config.actor_rollout_ref.rollout, "use_rollout_log_probs_as_old", False)
                        has_rollout = "rollout_log_probs" in batch.batch
                        if use_rollout_as_old and has_rollout:
                            batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"].clone()
                            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                        else:
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # apply KL penalty now that old_log_probs and ref_log_prob are available
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(
                            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)

                    # breakpoint()
                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # GDPO per-component reward metrics
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                # ReMax multi-reward per-component metrics (share the gdpo/ prefix)
                remax_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if remax_reward_keys and self.config.algorithm.adv_estimator in ("remax", AdvantageEstimator.REMAX):
                    for key in remax_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                        baseline_key = f"reward_baselines_{key}"
                        if baseline_key in batch.batch:
                            bvals = batch.batch[baseline_key].float()
                            metrics[f"gdpo/{key}/base_mean"] = float(bvals.mean())
                            metrics[f"gdpo/{key}/base_std"] = float(bvals.std())
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                gen_kept_frac = num_prompt_in_batch / (num_gen_batches * self.train_dataloader.batch_size)
                metrics.update(
                    {
                        "train/num_gen_batches": num_gen_batches,
                        "train/gen_kept_frac": gen_kept_frac,
                        "step": self.global_steps,
                        "progress": self.global_steps / self.total_training_steps,
                    }
                )
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)

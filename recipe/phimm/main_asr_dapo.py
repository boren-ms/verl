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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf, open_dict
from pathlib import Path

from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available
from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer
from recipe.phimm.utils.env import EnvMgr
from recipe.phimm.utils.shared import parse_asr_response
from verl.utils.ray_utils import ray_address, ray_host_url
from verl.utils.fs import copy_to_local


def get_env_vars():
    env_vars = EnvMgr().envs()
    required_envs = ["DATA_PATH"]
    assert all(k in env_vars for k in required_envs), (
        f"Missing env vars: {[k for k in required_envs if k not in env_vars]}"
    )
    return env_vars


@hydra.main(config_path="config", config_name="dapo_local_test", version_base=None)
def main(config):
    run_ppo(config)


def cwd():
    return Path(__file__).parents[2]


def _build_reward_config(config, reward_section):
    reward_config = OmegaConf.merge(config)
    overrides = config.get(reward_section, {})
    with open_dict(reward_config):
        for key in ("custom_reward_function", "reward_functions", "reward_function_by_data_source"):
            if key in overrides:
                reward_config[key] = overrides[key]
    return reward_config


def run_ppo(config) -> None:
    env_vars = get_env_vars()
    # Register the custom Qwen3.5-Audio HF model in every Ray process (TaskRunner +
    # FSDP/vLLM workers) via ``import verl`` so checkpoints load with
    # ``trust_remote_code=False`` (no dependency on per-checkpoint remote *.py files).
    env_vars.setdefault("VERL_USE_EXTERNAL_MODULES", "hf_qwen35_audio")
    print(f"Cluster Env: {env_vars}")
    if not ray.is_initialized():
        # this is for local ray cluster
        default_runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_SLEEP_LEVEL": "1",
                "VLLM_PLUGINS": "qwen35_audio",
                "HF_HUB_OFFLINE": "1",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                **env_vars,
            },
            "excludes": [str(cwd() / ".git")],
        }
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"Ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    try:
        env_vars["RAY_ADDRESS"] = ray_address()
        print(f"Worker Env: {env_vars}")
        runtime_env = {"env_vars": env_vars}
        if (
            is_cuda_available
            and config.global_profiler.tool == "nsys"
            and OmegaConf.select(config.global_profiler, "steps") is not None
            and len(OmegaConf.select(config.global_profiler, "steps")) > 0
        ):
            nsight_options = OmegaConf.to_container(
                config.global_profiler.global_tool_config.nsys.controller_nsight_options
            )
            runtime_env["nsight"] = nsight_options
        runner = TaskRunner.options(runtime_env=runtime_env).remote()
        ray.get(runner.run.remote(config))
    finally:
        if ray.is_initialized():
            ray.shutdown()


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf
        import recipe.phimm.reward.long_audio_grouped  # noqa: F401

        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {}, {}))

        os.chdir(str(cwd()))
        print(f"HostName: {socket.gethostname()}, RAY_HOST: {ray_host_url()}, PID: {os.getpid()}, CWD: {os.getcwd()}")

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        assert config.actor_rollout_ref.model.path is not None, "Please specify the actor model path"
        # download the checkpoint from remote azure blob
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.actor_rollout_ref.model.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        from verl.single_controller.ray import RayWorkerGroup

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}

            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id
        train_reward_config = _build_reward_config(config, "train_reward")
        reward_fn = load_reward_manager(
            train_reward_config,
            tokenizer,
            config.data.get("train_num_examine", 0),
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )

        val_reward_config = _build_reward_config(config, "val_reward")
        if val_rm := config.get("val_reward", {}).get("reward_manager"):
            val_reward_config.reward_model.reward_manager = val_rm
        val_reward_fn = load_reward_manager(
            val_reward_config,
            tokenizer,
            config.data.get("eval_num_examine", 1),
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            dump_fn=lambda t: parse_asr_response(t)["text"],
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()

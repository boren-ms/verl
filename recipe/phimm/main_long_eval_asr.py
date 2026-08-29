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
"""Long-recording ASR evaluation using the trainer-v1 rollout stack.

Trainer v1 generates every SVAD segment. This module then regroups segments by
``parent_audio_path``, orders them by ``seg_start``, concatenates their parsed
ASR text, and scores each original recording once. Results retain the legacy
per-source ``details.jsonl`` and ``measures.json`` layout.
"""

import asyncio
import inspect
import json
import os
import re
import socket
from collections import defaultdict
from pprint import pprint

import blobfile as bf
import hydra
import ray
from omegaconf import DictConfig, OmegaConf, open_dict

from recipe.phimm.reward.asr_inhouse_measure import eval_score
from recipe.phimm.utils.env import EnvMgr
from recipe.phimm.utils.shared import parse_asr_response
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.reward import get_val_reward_fn
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.trainer.ppo.v1 import AgentLoopManagerTQ, PPOTrainerSync
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.logging_utils import configure_verl_logging

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def dummy_score(*args, **kwargs) -> dict[str, float]:
    """Avoid redundant per-segment ASR scoring in the trainer-v1 reward loop."""
    return {"score": 0.0}


def _parent_key(extra_info: dict, audio_path, fallback: str) -> str:
    """Resolve the parent recording id for an exploded segment."""
    if extra_info:
        for key in ("parent_audio_path", "audio_path"):
            value = extra_info.get(key)
            if value:
                return str(value).split("#", 1)[0]
    if audio_path:
        return str(audio_path).split("#", 1)[0]
    return fallback


def _seg_start(extra_info: dict) -> float:
    if not extra_info:
        return 0.0
    value = extra_info.get("seg_start")
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _write_jsonl(records: list[dict], path: str) -> None:
    bf.makedirs(os.path.dirname(path.rstrip("/")))
    with bf.BlobFile(path, "w") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _write_json(obj, path: str) -> None:
    bf.makedirs(os.path.dirname(path.rstrip("/")))
    with bf.BlobFile(path, "w") as file:
        file.write(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def _micro(aggregate: dict) -> dict:
    if aggregate["metric"] == "wer":
        return {
            "wer": aggregate["n_err"] / max(aggregate["n_ref"], 1),
            "n_err": aggregate["n_err"],
            "n_ref": aggregate["n_ref"],
            "n_recordings": aggregate["n"],
        }
    return {
        "dter": aggregate["dter_n_err"] / max(aggregate["dter_n_ref"], 1),
        "dter_n_err": aggregate["dter_n_err"],
        "dter_n_ref": aggregate["dter_n_ref"],
        "eer": aggregate["eer_n_err"] / max(aggregate["eer_n_ref"], 1),
        "eer_n_err": aggregate["eer_n_err"],
        "eer_n_ref": aggregate["eer_n_ref"],
        "n_recordings": aggregate["n"],
    }


def _segment_details(members: list[dict]) -> list[dict]:
    return [
        {
            "seg_start": member["seg_start"],
            "audio_path": member["audio_path"],
            "prompt": member.get("prompt"),
            "ref": member["ref"],
            "response": member["response"],
            "raw_response": member["raw_response"],
        }
        for member in members
    ]


def score_segments(segments, measure_kwargs=None, score_fn=eval_score):
    """Group generated segments and score each complete recording once."""
    measure_kwargs = measure_kwargs or {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for segment in segments:
        groups[segment["parent"]].append(segment)

    details_by_source: dict[str, list[dict]] = defaultdict(list)
    aggregate = defaultdict(
        lambda: {
            "metric": None,
            "n_err": 0,
            "n_ref": 0,
            "dter_n_err": 0,
            "dter_n_ref": 0,
            "eer_n_err": 0,
            "eer_n_ref": 0,
            "n": 0,
        }
    )

    for parent, members in groups.items():
        members.sort(key=lambda member: member["seg_start"])
        hypothesis = " ".join(member["response"].strip() for member in members if member["response"].strip())
        head = members[0]
        data_source = str(head["data_source"] or "all")
        extra_info = dict(head.get("extra_info") or {})
        if head.get("language"):
            extra_info["language"] = head["language"]
        result = score_fn(
            hypothesis,
            head["ref"],
            data_source=data_source,
            extra_info=extra_info,
            **measure_kwargs,
        )
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        score = result if isinstance(result, dict) else {"score": result}

        record = {
            "parent_audio_path": parent,
            "id": head.get("id"),
            "data_source": data_source,
            "language": head.get("language"),
            "n_segments": len(members),
            "ref": head["ref"],
            "hyp": hypothesis,
            "response": [member["response"] for member in members],
            "segment_details": _segment_details(members),
        }
        record.update(score)
        details_by_source[data_source].append(record)

        totals = aggregate[data_source]
        metric = "dter" if "dter_n_ref" in score else "wer"
        if totals["metric"] not in (None, metric):
            raise ValueError(f"Inconsistent metric types returned for {data_source!r}")
        totals["metric"] = metric
        if metric == "wer":
            if "n_err" not in score or "n_ref" not in score:
                raise ValueError("A WER scorer must return n_err and n_ref")
            totals["n_err"] += int(score["n_err"])
            totals["n_ref"] += int(score["n_ref"])
        else:
            totals["dter_n_err"] += int(score.get("dter_n_err") or 0)
            totals["dter_n_ref"] += int(score.get("dter_n_ref") or 0)
            totals["eer_n_err"] += int(score.get("eer_n_err") or 0)
            totals["eer_n_ref"] += int(score.get("eer_n_ref") or 0)
        totals["n"] += 1

    return {
        source: {"details": details_by_source[source], "measure": _micro(totals)}
        for source, totals in aggregate.items()
    }


def _slug(source: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in str(source))


def write_results(results_by_source, output_dir, log_first_n_samples=0) -> None:
    """Write legacy per-source long-evaluation artifacts locally or to blob."""
    for source, result in results_by_source.items():
        details_path = f"{output_dir}/{_slug(source)}/details.jsonl"
        measures_path = f"{output_dir}/{_slug(source)}/measures.json"
        _write_jsonl(result["details"], details_path)
        _write_json(result["measure"], measures_path)

        measure = result["measure"]
        if "wer" in measure:
            print(
                f"[{source}] WER: {measure['wer']:.2%} "
                f"[{measure['n_err']}/{measure['n_ref']}] on {measure['n_recordings']} recordings"
            )
        else:
            print(
                f"[{source}] DTER: {measure['dter']:.2%} "
                f"[{measure['dter_n_err']}/{measure['dter_n_ref']}]  "
                f"EER: {measure['eer']:.2%} "
                f"[{measure['eer_n_err']}/{measure['eer_n_ref']}]  "
                f"on {measure['n_recordings']} recordings"
            )
        print(f"  Saved per-recording details to {details_path}")
        print(f"  Saved aggregate measures to {measures_path}")
        if log_first_n_samples:
            samples = result["details"][: int(log_first_n_samples)]
            print(json.dumps({"measure": measure, "samples": samples}, ensure_ascii=False, indent=2, default=str))


class LongASREvalTrainer(PPOTrainerSync):
    """Trainer-v1 sync mode specialized for validation-only long ASR."""

    def _recompute_val_reward(self, outputs, reward_models, data_sources, extra_infos):
        segments = []
        for index, (output, reward_model, data_source, extra_info) in enumerate(
            zip(outputs, reward_models, data_sources, extra_infos, strict=True)
        ):
            reward_model = getattr(reward_model, "data", reward_model)
            extra_info = getattr(extra_info, "data", extra_info)
            reward_model = reward_model if isinstance(reward_model, dict) else {}
            extra_info = extra_info if isinstance(extra_info, dict) else {}
            parsed_response = parse_asr_response(output).get("text") or ""
            parsed_response = re.sub(r"<nonspeech>", "", parsed_response, flags=re.IGNORECASE).strip()
            audio_path = extra_info.get("audio_path")
            segments.append(
                {
                    "parent": _parent_key(extra_info, audio_path, fallback=f"__row_{index}__"),
                    "seg_start": _seg_start(extra_info),
                    "raw_response": output,
                    "response": parsed_response,
                    "ref": reward_model.get("ground_truth", ""),
                    "prompt": extra_info.get("prompt"),
                    "audio_path": audio_path,
                    "data_source": data_source,
                    "id": extra_info.get("id"),
                    "language": extra_info.get("language"),
                    "extra_info": extra_info,
                }
            )

        val_reward_fn = get_val_reward_fn(self.config)

        def score_fn(solution_str, ground_truth, *, data_source, **kwargs):
            if val_reward_fn is None:
                return eval_score(solution_str, ground_truth, data_source=data_source, **kwargs)
            return val_reward_fn(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                **kwargs,
            )

        measure_kwargs = OmegaConf.select(self.config, "data.measure_kwargs", default={}) or {}
        if OmegaConf.is_config(measure_kwargs):
            measure_kwargs = OmegaConf.to_container(measure_kwargs, resolve=True)
        self.long_eval_results = score_segments(segments, measure_kwargs=measure_kwargs, score_fn=score_fn)

        records_by_parent = {
            record["parent_audio_path"]: record
            for result in self.long_eval_results.values()
            for record in result["details"]
        }
        scores = []
        reward_extra_infos = []
        for segment in segments:
            record = records_by_parent[segment["parent"]]
            scores.append(record.get("score", 0.0))
            reward_extra_infos.append(
                {key: value for key, value in record.items() if key not in {"segment_details", "response"}}
            )
        return scores, reward_extra_infos

    def _validate(self):
        if len(self.val_dataloader) != 1:
            raise ValueError(
                "Long ASR evaluation requires data.val_batch_size <= 0 so all segments can be regrouped before scoring."
            )
        metrics = super()._validate()
        output_dir = self.config.data.get("output_path")
        if not output_dir:
            raise ValueError("Please specify data.output_path")
        write_results(
            self.long_eval_results,
            str(output_dir).rstrip("/"),
            log_first_n_samples=self.config.data.get("log_first_n_samples", 0),
        )
        return metrics


@ray.remote(num_cpus=1)
class LongASREvalTaskRunner:
    """Ray controller that wires the long evaluator into trainer v1."""

    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def _init_agent_loop_manager(self):
        manager_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        manager_cls = load_class_from_fqn(manager_fqn, "AgentLoopManager") if manager_fqn else AgentLoopManagerTQ
        self.agent_loop_manager = manager_cls.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            teacher_client=self.trainer.get_teacher_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )

    def run(self, config: DictConfig):
        configure_verl_logging()
        import transfer_queue as tq

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        OmegaConf.register_new_resolver("eval", lambda expression: eval(expression, {}, {}), replace=True)
        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config

        tq.init(config.transfer_queue)
        succeeded = False
        try:
            self.trainer = LongASREvalTrainer(config=config)
            self.trainer.init()
            self._init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(self.trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


def _prepare_config(config: DictConfig) -> None:
    """Force validation-only settings and bridge the legacy eval keys."""
    env_vars = EnvMgr().envs()
    env_vars.setdefault("VERL_USE_EXTERNAL_MODULES", "hf_qwen35_audio")
    env_vars.setdefault("VLLM_PLUGINS", "qwen35_audio")
    env_vars.setdefault("HF_HUB_OFFLINE", "1")
    env_vars.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    with open_dict(config):
        config.trainer.use_v1 = True
        config.trainer.v1.trainer_mode = "sync"
        config.trainer.val_before_train = True
        config.trainer.val_only = True
        config.trainer.resume_mode = "disable"
        config.data.val_batch_size = -1
        config.data.validation_shuffle = False
        config.actor_rollout_ref.rollout.n = 1
        config.actor_rollout_ref.rollout.val_kwargs.n = 1
        config.val_reward.group_segment = True
        if config.get("reward_functions"):
            config.val_reward.reward_functions = OmegaConf.create(
                OmegaConf.to_container(config.reward_functions, resolve=False)
            )
        if config.get("reward_function_by_data_source"):
            config.val_reward.reward_function_by_data_source = OmegaConf.create(
                OmegaConf.to_container(config.reward_function_by_data_source, resolve=False)
            )
        config.reward_functions = {}
        config.reward_function_by_data_source = {}
        runtime_env = config.ray_kwargs.ray_init.get("runtime_env", {})
        runtime_env["env_vars"] = {**runtime_env.get("env_vars", {}), **env_vars}
        config.ray_kwargs.ray_init.runtime_env = runtime_env


def run_eval(config: DictConfig) -> None:
    """Prepare and launch a trainer-v1 long-recording evaluation."""
    _prepare_config(config)
    auto_set_device(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    run_ppo(config, task_runner_class=LongASREvalTaskRunner)


@hydra.main(
    config_path="config/eval",
    config_name="long_eval_mixlang_fy26q2",
    version_base=None,
)
def main(config):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expression: eval(expression, {}, {}))
    run_eval(config)


if __name__ == "__main__":
    main()

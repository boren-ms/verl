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
"""Long-audio ASR evaluation via ``generate_sequences``.

This is a lightweight, gen-style evaluator (modelled on
``recipe.phimm.main_asr_gen``) tailored for *long* recordings:

1. The dataset is exploded into <=``max_len_sec`` segments by
   :func:`recipe.phimm.data.dataset.svad_explode` (configured in the val_data
   ``pre_process`` block). Every child row carries ``parent_audio_path`` /
   ``seg_start`` so segments can be regrouped after generation.
2. ``generate_sequences`` transcribes every segment.
3. Per-segment hypotheses are grouped by ``parent_audio_path``, sorted by
   ``seg_start`` and concatenated, then scored *once per parent* against the
   full reference using :func:`recipe.phimm.reward.asr_inhouse_measure.eval_score`
   (DisfluencyTolerant TER + entity EER).
4. Per-recording results are written as JSONL and the aggregate TER/EER measures
   as JSON.

Unlike ``main_asr_eval`` (full PPO trainer + reward manager), this script only
spins up a rollout worker group, so it is cheap to run for eval-only sweeps.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import blobfile as bf
import hydra
import numpy as np
import ray
import uuid
from datasets import Dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from recipe.phimm.data.rl_dataset import RLHFDataset
from recipe.phimm.reward.asr_inhouse_measure import eval_score
from recipe.phimm.reward.asr_edge import eval_score as edge_eval_score  # noqa: F401 (keeps resolver parity)
from recipe.phimm.utils.env import EnvMgr
from recipe.phimm.utils.shared import parse_asr_response


def cwd():
    return Path(__file__).parents[2]


def get_env_vars():
    env_vars = EnvMgr().envs()
    required_envs = ["DATA_PATH"]
    assert all(k in env_vars for k in required_envs), (
        f"Missing env vars: {[k for k in required_envs if k not in env_vars]}"
    )
    return env_vars


@hydra.main(config_path="config/eval", config_name="long_eval_test", version_base=None)
def main(config):
    run_eval(config)


def run_eval(config) -> None:
    env_vars = get_env_vars()
    print(f"Cluster Env: {env_vars}")
    if not ray.is_initialized():
        default_runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
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
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    ray.get(main_task.remote(config))


def _parent_key(extra_info: dict, audio_path, fallback: str) -> str:
    """Resolve the parent recording id for grouping exploded segments."""
    if extra_info:
        for k in ("parent_audio_path", "audio_path"):
            v = extra_info.get(k)
            if v:
                return str(v).split("#", 1)[0]
    if audio_path:
        return str(audio_path).split("#", 1)[0]
    return fallback


def _seg_start(extra_info: dict) -> float:
    if not extra_info:
        return 0.0
    v = extra_info.get("seg_start")
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _write_jsonl(records: list[dict], path: str) -> None:
    bf.makedirs(os.path.dirname(path.rstrip("/")))
    with bf.BlobFile(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _write_json(obj, path: str) -> None:
    bf.makedirs(os.path.dirname(path.rstrip("/")))
    with bf.BlobFile(path, "w") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def build_model(config):
    """Resolve the checkpoint locally and load its tokenizer + processor."""
    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.model.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
    assert tokenizer is not None, "Please specify a valid tokenizer"
    assert processor is not None, "Please specify a valid processor"
    return local_path, tokenizer, processor


def build_dataloader(config, tokenizer, processor):
    """Load the (svad-exploded) eval dataset and wrap it in a DataLoader."""
    ds_conf = config.data.get("gen_data", config.data.get("val_data", config.data.get("train_data", None)))
    assert ds_conf is not None, "Please specify data.val_data (or data.gen_data) in the config"
    dataset = RLHFDataset(ds_conf, tokenizer, config.data, processor)
    print(f"Loaded RLHFDataset with {len(dataset)} segments (post svad_explode).")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.data.get("batch_size", 256),
        num_workers=config.data.get("num_workers", 0),
        shuffle=False,
        drop_last=False,
        collate_fn=default_collate_fn,
    )
    return dataloader


def build_worker_group(config):
    """Spawn the rollout-only worker group and initialize the model on it."""
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    worker_dict_cls = create_colocated_worker_cls(class_dict={"rollout": ray_cls_with_init})
    process_on_nodes = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
    resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, max_colocate_count=1)
    wg_dict = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=worker_dict_cls,
        device_name=config.trainer.device,
    )
    wg = wg_dict.spawn(prefix_set=["rollout"])["rollout"]
    wg.init_model()
    return wg


def generate_segments(wg, dataloader, tokenizer):
    """Transcribe every segment and return a flat list of segment records."""
    segments: list[dict] = []
    total_batches = len(dataloader)
    for batch_idx, batch_dict in enumerate(tqdm(dataloader, total=total_batches, desc="generate")):
        prompts = [msg[0]["content"] for msg in batch_dict["prompt"]]
        refs = [x["ground_truth"] for x in batch_dict["reward_model"]]
        n_egs = len(refs)
        audio_paths = batch_dict.get("audio_path", [None] * n_egs)
        extras = batch_dict.get("extra_info", [{}] * n_egs)

        data = DataProto.from_single_dict(batch_dict)
        if "uid" not in data.non_tensor_batch:
            data.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(data.batch))], dtype=object
            )
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)
        output_padded = wg.generate_sequences(data_padded)
        output = unpad_dataproto(output_padded, pad_size=pad_size)

        for i in range(n_egs):
            data_item = output[i]
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = data_item.batch["responses"][:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos = tokenizer.eos_token
            if eos and response_str.endswith(eos):
                response_str = response_str[: -len(eos)]

            extra = extras[i] if extras[i] else {}
            segments.append(
                {
                    "parent": _parent_key(extra, audio_paths[i], fallback=f"__row_{batch_idx}_{i}__"),
                    "seg_start": _seg_start(extra),
                    "response": response_str,
                    "ref": refs[i],
                    "prompt": prompts[i],
                    "audio_path": audio_paths[i],
                    "data_source": extra.get("data_source") or batch_dict.get("data_source", [None] * n_egs)[i],
                    "id": extra.get("id"),
                    "language": extra.get("language"),
                }
            )

    print(f"Generated {len(segments)} segment hypotheses; grouping by parent recording.")
    return segments


def _micro(a):
    return {
        "dter": a["dter_n_err"] / max(a["dter_n_ref"], 1),
        "dter_n_err": a["dter_n_err"],
        "dter_n_ref": a["dter_n_ref"],
        "eer": a["eer_n_err"] / max(a["eer_n_ref"], 1),
        "eer_n_err": a["eer_n_err"],
        "eer_n_ref": a["eer_n_ref"],
        "n_recordings": a["n"],
    }


def score_segments(segments, measure_kwargs, num_examine):
    """Group segments by parent, concat hyps, score once per recording.

    Returns ``(details, measures)`` where ``details`` is the per-recording list
    and ``measures`` carries the micro-averaged overall/by_source TER + EER.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for seg in segments:
        groups[seg["parent"]].append(seg)

    details: list[dict] = []
    agg = defaultdict(lambda: {"dter_n_err": 0, "dter_n_ref": 0, "eer_n_err": 0, "eer_n_ref": 0, "n": 0})

    n_printed = 0
    for parent, members in tqdm(groups.items(), total=len(groups), desc="score"):
        members.sort(key=lambda m: m["seg_start"])
        concat_hyp = " ".join(m["response"].strip() for m in members if m["response"].strip())
        head = members[0]
        ref = head["ref"]
        data_source = head["data_source"] or "all"

        score = eval_score(concat_hyp, ref, **measure_kwargs)

        rec = {
            "parent_audio_path": parent,
            "id": head["id"],
            "data_source": data_source,
            "language": head["language"],
            "n_segments": len(members),
            "ref": ref,
            "hyp": concat_hyp,
            "dter": score.get("dter"),
            "dter_n_err": score.get("dter_n_err"),
            "dter_n_ref": score.get("dter_n_ref"),
            "eer": score.get("eer"),
            "eer_n_err": score.get("eer_n_err"),
            "eer_n_ref": score.get("eer_n_ref"),
            "dter_detail": score.get("dter_detail"),
        }
        details.append(rec)

        for key in (data_source, "__overall__"):
            a = agg[key]
            a["dter_n_err"] += int(score.get("dter_n_err") or 0)
            a["dter_n_ref"] += int(score.get("dter_n_ref") or 0)
            a["eer_n_err"] += int(score.get("eer_n_err") or 0)
            a["eer_n_ref"] += int(score.get("eer_n_ref") or 0)
            a["n"] += 1

        if n_printed < num_examine:
            n_printed += 1
            print(f"--- {parent} ({len(members)} segments) ---")
            print(f"  DTER: {(score.get('dter') or 0.0):.2%}  EER: {(score.get('eer') or 0.0):.2%}")
            print(f"  Ref: {str(rec['ref'])[:300]}")
            print(f"  Hyp: {str(rec['hyp'])[:300]}")

    measures = {
        "overall": _micro(agg["__overall__"]),
        "by_source": {src: _micro(a) for src, a in agg.items() if src != "__overall__"},
    }
    return details, measures


@ray.remote(num_cpus=1)
def main_task(config):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {}, {}))
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    output_dir = config.data.get("output_path", None)
    assert output_dir is not None, "Please specify data.output_path"
    output_dir = output_dir.rstrip("/")
    details_path = f"{output_dir}/result_details.jsonl"
    measures_path = f"{output_dir}/measures.json"
    num_examine = config.data.get("eval_num_examine", 1)

    measure_kwargs = config.data.get("measure_kwargs", {})
    if OmegaConf.is_config(measure_kwargs):
        measure_kwargs = OmegaConf.to_container(measure_kwargs, resolve=True)

    _, tokenizer, processor = build_model(config)
    dataloader = build_dataloader(config, tokenizer, processor)
    wg = build_worker_group(config)

    segments = generate_segments(wg, dataloader, tokenizer)
    details, measures = score_segments(segments, measure_kwargs, num_examine)

    _write_jsonl(details, details_path)
    _write_json(measures, measures_path)

    overall = measures["overall"]
    print(
        f"Overall DTER: {overall['dter']:.2%} [{overall['dter_n_err']}/{overall['dter_n_ref']}]  "
        f"EER: {overall['eer']:.2%} [{overall['eer_n_err']}/{overall['eer_n_ref']}]  "
        f"on {overall['n_recordings']} recordings ({len(segments)} segments)"
    )
    print(f"Saved per-recording details to {details_path}")
    print(f"Saved aggregate measures to {measures_path}")
    print("All Done")


if __name__ == "__main__":
    main()

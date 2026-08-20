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
    full reference. ``custom_reward_function`` can select a scorer;
    the default is DisfluencyTolerant TER + entity EER.
4. Per-recording results and aggregate measures are written as JSONL and JSON.

Unlike ``main_asr_eval`` (full PPO trainer + reward manager), this script only
spins up a rollout worker group, so it is cheap to run for eval-only sweeps.
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import blobfile as bf
import hydra
import numpy as np
import ray
import uuid
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
from verl.trainer.ppo.reward import get_custom_reward_fn

from recipe.phimm.data.rl_dataset import RLHFDataset
from recipe.phimm.reward.asr_inhouse_measure import eval_score
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
    # Register the custom Qwen3.5-Audio HF model in every Ray process via
    # ``import verl`` so tokenizer/processor/config load with
    # ``trust_remote_code=False`` (no dependency on per-checkpoint remote *.py files).
    env_vars.setdefault("VERL_USE_EXTERNAL_MODULES", "hf_qwen35_audio")
    print(f"Cluster Env: {env_vars}")
    if not ray.is_initialized():
        default_runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
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
    dataset = RLHFDataset(ds_conf, tokenizer, config.data, processor, is_train=False)
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
            raw_response = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos = tokenizer.eos_token
            if eos and raw_response.endswith(eos):
                raw_response = raw_response[: -len(eos)]
            response_str = parse_asr_response(raw_response).get("text") or ""
            response_str = re.sub(r"<nonspeech>", "", response_str, flags=re.IGNORECASE).strip()

            extra = extras[i] if extras[i] else {}
            segments.append(
                {
                    "parent": _parent_key(extra, audio_paths[i], fallback=f"__row_{batch_idx}_{i}__"),
                    "seg_start": _seg_start(extra),
                    "raw_response": raw_response,
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
    if a["metric"] == "wer":
        return {
            "wer": a["n_err"] / max(a["n_ref"], 1),
            "n_err": a["n_err"],
            "n_ref": a["n_ref"],
            "n_recordings": a["n"],
        }
    return {
        "dter": a["dter_n_err"] / max(a["dter_n_ref"], 1),
        "dter_n_err": a["dter_n_err"],
        "dter_n_ref": a["dter_n_ref"],
        "eer": a["eer_n_err"] / max(a["eer_n_ref"], 1),
        "eer_n_err": a["eer_n_err"],
        "eer_n_ref": a["eer_n_ref"],
        "n_recordings": a["n"],
    }


def _segment_details(members: list[dict]) -> list[dict]:
    """Keep the generated inputs and outputs for one long recording."""
    return [
        {
            "seg_start": member["seg_start"],
            "audio_path": member["audio_path"],
            "prompt": member["prompt"],
            "ref": member["ref"],
            "response": member["response"],
            "raw_response": member["raw_response"],
        }
        for member in members
    ]


def score_segments(segments, measure_kwargs, score_fn=eval_score):
    """Group segments by parent, concat hyps, score once per recording.

    Returns a dict keyed by ``data_source`` mapping to
    ``{"details": [...], "measure": {...}}`` (per-recording detail list plus
    micro-averaged DTER/EER or WER for that source).
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for seg in segments:
        groups[seg["parent"]].append(seg)

    details_by_source: dict[str, list[dict]] = defaultdict(list)
    agg = defaultdict(
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

    for parent, members in tqdm(groups.items(), total=len(groups), desc="score"):
        members.sort(key=lambda m: m["seg_start"])
        concat_hyp = " ".join(m["response"].strip() for m in members if m["response"].strip())
        responses = [m["response"] for m in members]
        segment_details = _segment_details(members)
        head = members[0]
        ref = head["ref"]
        data_source = head["data_source"] or "all"

        extra_info = {"language": head["language"]} if head["language"] else {}
        score = score_fn(concat_hyp, ref, extra_info=extra_info, **measure_kwargs)

        rec = {
            "parent_audio_path": parent,
            "id": head["id"],
            "data_source": data_source,
            "language": head["language"],
            "n_segments": len(members),
            "ref": ref,
            "hyp": concat_hyp,
            "response": responses,
            "segment_details": segment_details,
            "dter": score.get("dter"),
            "dter_n_err": score.get("dter_n_err"),
            "dter_n_ref": score.get("dter_n_ref"),
            "eer": score.get("eer"),
            "eer_n_err": score.get("eer_n_err"),
            "eer_n_ref": score.get("eer_n_ref"),
            "dter_detail": score.get("dter_detail"),
        }
        rec.update(score)
        details_by_source[data_source].append(rec)

        a = agg[data_source]
        metric = "dter" if "dter_n_ref" in score else "wer"
        if a["metric"] not in (None, metric):
            raise ValueError(f"Score function returned inconsistent metric types for {data_source!r}")
        a["metric"] = metric
        if metric == "wer":
            if "n_err" not in score or "n_ref" not in score:
                raise ValueError("Custom score function must return n_err and n_ref for WER aggregation")
            a["n_err"] += int(score["n_err"])
            a["n_ref"] += int(score["n_ref"])
        else:
            a["dter_n_err"] += int(score.get("dter_n_err") or 0)
            a["dter_n_ref"] += int(score.get("dter_n_ref") or 0)
            a["eer_n_err"] += int(score.get("eer_n_err") or 0)
            a["eer_n_ref"] += int(score.get("eer_n_ref") or 0)
        a["n"] += 1

    return {
        src: {"details": details_by_source[src], "measure": _micro(a)}
        for src, a in agg.items()
    }


def _slug(src: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(src))


def _log_sample_details(src: str, details: list[dict], measure: dict, n_samples: int) -> None:
    """Log the first long-recording results and source-level measures."""
    if n_samples <= 0:
        return
    samples = details[:n_samples]
    print(
        f"[{src}] First {len(samples)}/{len(details)} long-recording samples "
        f"with aggregate measures:\n{json.dumps({'measure': measure, 'samples': samples}, ensure_ascii=False, indent=2, default=str)}"
    )


def write_results(results_by_source, output_dir, log_first_n_samples=0):
    """Write per-data-source details JSONL + measures JSON under ``output_dir``.

    Each source is written to its own subdirectory (``{output_dir}/{slug}/``).
    """
    log_first_n_samples = int(log_first_n_samples or 0)
    for src, res in results_by_source.items():
        slug = _slug(src)
        details_path = f"{output_dir}/{slug}/details.jsonl"
        measures_path = f"{output_dir}/{slug}/measures.json"
        _write_jsonl(res["details"], details_path)
        _write_json(res["measure"], measures_path)

        m = res["measure"]
        if "wer" in m:
            print(f"[{src}] WER: {m['wer']:.2%} [{m['n_err']}/{m['n_ref']}] on {m['n_recordings']} recordings")
        else:
            print(
                f"[{src}] DTER: {m['dter']:.2%} [{m['dter_n_err']}/{m['dter_n_ref']}]  "
                f"EER: {m['eer']:.2%} [{m['eer_n_err']}/{m['eer_n_ref']}]  "
                f"on {m['n_recordings']} recordings"
            )
        print(f"  Saved per-recording details to {details_path}")
        print(f"  Saved aggregate measures to {measures_path}")
        _log_sample_details(src, res["details"], m, log_first_n_samples)


@ray.remote(num_cpus=1)
def main_task(config):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {}, {}))
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    output_dir = config.data.get("output_path", None)
    assert output_dir is not None, "Please specify data.output_path"
    output_dir = output_dir.rstrip("/")

    measure_kwargs = config.data.get("measure_kwargs", {})
    if OmegaConf.is_config(measure_kwargs):
        measure_kwargs = OmegaConf.to_container(measure_kwargs, resolve=True)
    score_fn = get_custom_reward_fn(config) or eval_score

    _, tokenizer, processor = build_model(config)
    dataloader = build_dataloader(config, tokenizer, processor)
    wg = build_worker_group(config)

    segments = generate_segments(wg, dataloader, tokenizer)
    results_by_source = score_segments(segments, measure_kwargs, score_fn=score_fn)

    write_results(
        results_by_source,
        output_dir,
        log_first_n_samples=config.data.get("log_first_n_samples", 0),
    )

    print(f"Scored {len(segments)} segments across {len(results_by_source)} data sources")
    print("All Done")


if __name__ == "__main__":
    main()

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
"""Long-audio ASR evaluation via standalone vLLM server replicas.

This is a lightweight, gen-style evaluator (modelled on
``recipe.phimm.main_asr_gen``) tailored for *long* recordings. Unlike the
upstream worker-group implementation, this verl-mirror port generates through
the same ``LLMServerManager`` machinery as ``main_asr_gen`` so it fits the
fully-async (server-based) rollout architecture.

Pipeline:

1. The dataset is exploded into <=``max_len_sec`` segments by
   :func:`recipe.phimm.data.dataset.svad_explode` (configured in the val_data
   ``pre_process`` block), or pre-segmented offline. Every child row carries
   ``parent_audio_path`` / ``seg_start`` (via ``extra_keys``) so segments can be
   regrouped after generation.
2. ``LLMServerManager`` transcribes every segment.
3. Per-segment hypotheses are grouped by ``parent_audio_path``, sorted by
   ``seg_start`` and concatenated, then scored *once per parent* against the
   full reference using :func:`recipe.phimm.reward.asr_inhouse_measure.eval_score`
   (DisfluencyTolerant TER + entity EER).
4. Per-recording results are written as JSONL and the aggregate TER/EER measures
   as JSON, split per ``data_source``.
"""

import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import blobfile as bf
import hydra
import ray
from omegaconf import OmegaConf

from recipe.phimm.data.rl_dataset import RLHFDataset
from recipe.phimm.main_asr_gen import (  # mirror-native vLLM-server generation helpers, reused below
    _build_sampling_params,
    _generate_one,
    _load_processor,
    _normalize_config,
    _prepare_item,
    _resolve_model_path,
)
from recipe.phimm.reward.asr_edge import eval_score as edge_eval_score  # noqa: F401 (keeps resolver parity)
from recipe.phimm.reward.asr_inhouse_measure import eval_score
from recipe.phimm.utils.env import EnvMgr
from recipe.phimm.utils.shared import parse_asr_response
from verl.utils import hf_tokenizer
from verl.workers.rollout.llm_server import LLMServerManager

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = logging.getLogger(__name__)


def cwd():
    return Path(__file__).parents[2]


# ---------------------------------------------------------------------------
# Grouping / scoring helpers (ported from the upstream evaluator)
# ---------------------------------------------------------------------------

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


def _write_jsonl(records: list, path: str) -> None:
    bf.makedirs(os.path.dirname(path.rstrip("/")))
    with bf.BlobFile(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _write_json(obj, path: str) -> None:
    bf.makedirs(os.path.dirname(path.rstrip("/")))
    with bf.BlobFile(path, "w") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def _decode_response(result, tokenizer) -> tuple:
    """Decode a generation result to (raw_response, parsed_text)."""
    if isinstance(result, Exception):
        logger.error("Generation failed: %s", result)
        return f"ERROR: {result}", ""
    raw_response = tokenizer.decode(result.token_ids, skip_special_tokens=True)
    eos = tokenizer.eos_token
    if eos and raw_response.endswith(eos):
        raw_response = raw_response[: -len(eos)]
    response_str = parse_asr_response(raw_response).get("text") or ""
    response_str = re.sub(r"<nonspeech>", "", response_str, flags=re.IGNORECASE).strip()
    return raw_response, response_str


def _micro(a: dict) -> dict:
    return {
        "dter": a["dter_n_err"] / max(a["dter_n_ref"], 1),
        "dter_n_err": a["dter_n_err"],
        "dter_n_ref": a["dter_n_ref"],
        "eer": a["eer_n_err"] / max(a["eer_n_ref"], 1),
        "eer_n_err": a["eer_n_err"],
        "eer_n_ref": a["eer_n_ref"],
        "n_recordings": a["n"],
    }


def score_segments(segments: list, measure_kwargs: dict) -> dict:
    """Group segments by parent, concat hyps, score once per recording.

    Returns a dict keyed by ``data_source`` mapping to
    ``{"details": [...], "measure": {...}}`` (per-recording detail list plus the
    micro-averaged TER + EER for that source).
    """
    groups: dict = defaultdict(list)
    for seg in segments:
        groups[seg["parent"]].append(seg)

    details_by_source: dict = defaultdict(list)
    agg = defaultdict(lambda: {"dter_n_err": 0, "dter_n_ref": 0, "eer_n_err": 0, "eer_n_ref": 0, "n": 0})

    for parent, members in groups.items():
        members.sort(key=lambda m: m["seg_start"])
        concat_hyp = " ".join(m["response"].strip() for m in members if m["response"].strip())
        responses = [m["response"] for m in members]
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
            "response": responses,
            "dter": score.get("dter"),
            "dter_n_err": score.get("dter_n_err"),
            "dter_n_ref": score.get("dter_n_ref"),
            "eer": score.get("eer"),
            "eer_n_err": score.get("eer_n_err"),
            "eer_n_ref": score.get("eer_n_ref"),
            "dter_detail": score.get("dter_detail"),
        }
        details_by_source[data_source].append(rec)

        a = agg[data_source]
        a["dter_n_err"] += int(score.get("dter_n_err") or 0)
        a["dter_n_ref"] += int(score.get("dter_n_ref") or 0)
        a["eer_n_err"] += int(score.get("eer_n_err") or 0)
        a["eer_n_ref"] += int(score.get("eer_n_ref") or 0)
        a["n"] += 1

    return {src: {"details": details_by_source[src], "measure": _micro(a)} for src, a in agg.items()}


def _slug(src: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(src))


def write_results(results_by_source: dict, output_dir: str) -> None:
    """Write per-data-source details JSONL + measures JSON under ``output_dir``."""
    for src, res in results_by_source.items():
        slug = _slug(src)
        details_path = f"{output_dir}/{slug}/details.jsonl"
        measures_path = f"{output_dir}/{slug}/measures.json"
        _write_jsonl(res["details"], details_path)
        _write_json(res["measure"], measures_path)

        m = res["measure"]
        print(
            f"[{src}] DTER: {m['dter']:.2%} [{m['dter_n_err']}/{m['dter_n_ref']}]  "
            f"EER: {m['eer']:.2%} [{m['eer_n_err']}/{m['eer_n_ref']}]  "
            f"on {m['n_recordings']} recordings"
        )
        print(f"  Saved per-recording details to {details_path}")
        print(f"  Saved aggregate measures to {measures_path}")


# ---------------------------------------------------------------------------
# Generation (server-based, mirrors main_asr_gen continuous-feed loop)
# ---------------------------------------------------------------------------

async def _generate_segments_async(config, tracker=None) -> list:
    """Transcribe every segment and return a flat list of segment records.

    When ``tracker`` is provided (a :class:`verl.utils.tracking.Tracking`
    instance), generation progress (segments done / fraction / throughput) is
    logged to the configured backends (e.g. wandb) every ``log_interval``
    segments, so long eval runs show a live progress curve.
    """
    OmegaConf.resolve(config)
    _normalize_config(config)

    local_model_path = _resolve_model_path(config.actor_rollout_ref.model.path.rstrip("/"))
    OmegaConf.update(config, "actor_rollout_ref.model.path", local_model_path)

    trust_rc = config.actor_rollout_ref.model.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=trust_rc)
    processor = _load_processor(local_model_path, trust_rc)
    processor_sr = getattr(
        processor, "feature_extractor", getattr(processor, "audio_feature_extractor", None)
    ).sampling_rate

    ds_conf = (
        OmegaConf.select(config, "data.gen_data", default=None)
        or OmegaConf.select(config, "data.val_data", default=None)
        or OmegaConf.select(config, "data.train_data")
    )
    assert ds_conf is not None, "Please specify data.val_data (or data.gen_data) in the config"
    dataset = RLHFDataset(data_files=ds_conf, tokenizer=tokenizer, config=config.data, processor=processor)
    logger.info("Loaded RLHFDataset with %d segments (post svad_explode).", len(dataset))

    sampling_params = _build_sampling_params(config)
    logger.info("Launching vLLM server replicas …")
    server_manager = await LLMServerManager.create(config=config)
    client = server_manager.get_client()
    logger.info("Server replicas ready")

    audio_tok_ids = tokenizer.encode("<audio>", add_special_tokens=False)
    newline_tok = tokenizer.encode("\n", add_special_tokens=False)
    n_total = len(dataset)
    concurrency = config.data.get("concurrency", config.data.get("batch_size", 256))
    log_interval = config.data.get("log_interval", 100)

    segments: list = [None] * n_total
    done_count = 0
    t0 = time.time()
    sem = asyncio.Semaphore(concurrency)

    async def _worker(idx):
        async with sem:
            gen_input, meta = _prepare_item(dataset[idx], audio_tok_ids, newline_tok)
            try:
                result = await _generate_one(client, gen_input, sampling_params, processor_sr)
            except Exception as exc:  # noqa: BLE001
                result = exc
            raw_response, response_str = _decode_response(result, tokenizer)
            extra = meta.get("extra_info") if isinstance(meta.get("extra_info"), dict) else {}
            return idx, {
                "parent": _parent_key(extra, meta.get("audio_path"), fallback=f"__row_{idx}__"),
                "seg_start": _seg_start(extra),
                "raw_response": raw_response,
                "response": response_str,
                "ref": meta.get("ground_truth"),
                "prompt": meta.get("prompt"),
                "audio_path": meta.get("audio_path"),
                "data_source": meta.get("data_source") or (extra.get("data_source") if extra else None),
                "id": extra.get("id") if extra else None,
                "language": extra.get("language") if extra else None,
            }

    pending_tasks: set = set()
    submit_idx = 0

    def _submit_one():
        nonlocal submit_idx
        if submit_idx < n_total:
            pending_tasks.add(asyncio.create_task(_worker(submit_idx)))
            submit_idx += 1

    for _ in range(min(concurrency, n_total)):
        _submit_one()

    while pending_tasks:
        finished, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in finished:
            idx, rec = await t
            segments[idx] = rec
            done_count += 1
            _submit_one()
            if done_count % log_interval == 0 or done_count == n_total:
                elapsed = time.time() - t0
                logger.info("%d/%d segments done | elapsed %.1fs", done_count, n_total, elapsed)
                if tracker is not None:
                    tracker.log(
                        data={
                            "eval/segments_done": done_count,
                            "eval/segments_total": n_total,
                            "eval/progress": done_count / max(n_total, 1),
                            "eval/throughput_segs_per_s": done_count / max(elapsed, 1e-6),
                            "eval/elapsed_sec": elapsed,
                        },
                        step=done_count,
                    )

    segments = [s for s in segments if s is not None]
    logger.info("Generated %d segment hypotheses; grouping by parent recording.", len(segments))
    return segments


def _build_tracker(config):
    """Create a :class:`verl.utils.tracking.Tracking` logger from the eval config.

    Mirrors the PPO trainer setup so the configured backends (``trainer.logger``,
    e.g. ``[console, wandb]``) receive long-eval progress + final metrics.
    Returns ``None`` if no trainer/logger section is present.
    """
    trainer_cfg = OmegaConf.select(config, "trainer", default=None)
    if trainer_cfg is None:
        return None
    backends = OmegaConf.select(config, "trainer.logger", default=["console"])
    if OmegaConf.is_config(backends):
        backends = OmegaConf.to_container(backends, resolve=True)
    try:
        from verl.utils.tracking import Tracking

        return Tracking(
            project_name=OmegaConf.select(config, "trainer.project_name", default="phimm_long_eval"),
            experiment_name=OmegaConf.select(config, "trainer.experiment_name", default="phimm_long_eval"),
            default_backend=backends,
            config=OmegaConf.to_container(config, resolve=True),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to init Tracking logger (%s); progress logging disabled.", exc)
        return None


def _log_final_metrics(tracker, results_by_source: dict, step: int) -> None:
    """Log per-source and overall micro-averaged DTER/EER to the tracker."""
    data: dict = {}
    tot = {"dter_n_err": 0, "dter_n_ref": 0, "eer_n_err": 0, "eer_n_ref": 0, "n": 0}
    for src, res in results_by_source.items():
        m = res["measure"]
        slug = _slug(src)
        data[f"eval/{slug}/dter"] = m["dter"]
        data[f"eval/{slug}/eer"] = m["eer"]
        data[f"eval/{slug}/n_recordings"] = m["n_recordings"]
        tot["dter_n_err"] += m["dter_n_err"]
        tot["dter_n_ref"] += m["dter_n_ref"]
        tot["eer_n_err"] += m["eer_n_err"]
        tot["eer_n_ref"] += m["eer_n_ref"]
        tot["n"] += m["n_recordings"]
    data["eval/overall/dter"] = tot["dter_n_err"] / max(tot["dter_n_ref"], 1)
    data["eval/overall/eer"] = tot["eer_n_err"] / max(tot["eer_n_ref"], 1)
    data["eval/overall/n_recordings"] = tot["n"]
    tracker.log(data=data, step=max(step, 1))


def run_eval(config) -> None:
    env_vars = EnvMgr().envs()
    print(f"Cluster Env: {env_vars}")
    if not ray.is_initialized():
        ray_init_kwargs = OmegaConf.to_container(
            config.get("ray_kwargs", {}).get("ray_init", {}), resolve=True
        ) or {}
        runtime_env = {
            **{
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "HF_HUB_OFFLINE": "1",
                    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                    **env_vars,
                },
                "excludes": [str(cwd() / ".git")],
            },
            **ray_init_kwargs.pop("runtime_env", {}),
        }
        ray_init_kwargs["runtime_env"] = runtime_env
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**ray_init_kwargs)

    output_dir = OmegaConf.select(config, "data.output_path", default=None)
    assert output_dir is not None, "Please specify data.output_path"
    output_dir = output_dir.rstrip("/")

    measure_kwargs = config.data.get("measure_kwargs", {})
    if OmegaConf.is_config(measure_kwargs):
        measure_kwargs = OmegaConf.to_container(measure_kwargs, resolve=True)

    pprint(OmegaConf.to_container(config, resolve=True))

    # Set up progress / metric logging (wandb, console, ...) the same way the
    # PPO trainer does, so long eval runs are visible in wandb.
    tracker = _build_tracker(config)

    segments = asyncio.run(_generate_segments_async(config, tracker=tracker))
    results_by_source = score_segments(segments, measure_kwargs)
    write_results(results_by_source, output_dir)

    if tracker is not None:
        _log_final_metrics(tracker, results_by_source, step=len(segments))

    print(f"Scored {len(segments)} segments across {len(results_by_source)} data sources")
    print("All Done")


@hydra.main(config_path="config/eval", config_name="long_eval_test", version_base=None)
def main(config):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {}, {}))
    run_eval(config)


if __name__ == "__main__":
    main()

"""Long-audio ASR rollout via FullyAsyncRollouter + MessageQueue consumer.

This combines two existing recipes:

* the fully-async (server-based) rollout engine from
  :mod:`recipe.phimm.asr_rollout` (``FullyAsyncRollouter`` feeding a
  ``MessageQueue`` that a consumer drains), and
* the long-recording segmentation / per-parent regrouping + DTER/EER scoring
  from :mod:`recipe.phimm.main_long_eval_asr`.

Pipeline:

1. The rollout dataset is exploded into <=``max_len_sec`` segments by
   :func:`recipe.phimm.data.dataset.svad_explode` (configured in the
   ``train_files`` ``pre_process`` block), or pre-segmented offline. Every child
   row carries ``parent_audio_path`` / ``seg_start`` (via ``extra_keys``) so
   segments can be regrouped after generation.
2. ``FullyAsyncRollouter`` transcribes every segment; each rollout sample is
   pushed onto a ``MessageQueue``.
3. The consumer decodes every segment sample, then once the queue drains groups
   the segments by ``parent_audio_path``, sorts by ``seg_start``, concatenates
   the per-segment hypotheses and scores *once per recording* against the full
   reference using
   :func:`recipe.phimm.reward.asr_inhouse_measure.eval_score`
   (DisfluencyTolerant TER + entity EER).
4. Per-recording results are written as JSONL and the aggregate TER/EER measures
   as JSON, split per ``data_source`` (same layout as
   :func:`recipe.phimm.main_long_eval_asr.write_results`).

Usage:
    python3 -m recipe.phimm.long_asr_rollout \
        --config-path=config/rollout \
        --config-name=long_rollout_test
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from pprint import pprint

import blobfile as bf
import hydra
import ray
from omegaconf import OmegaConf

# Rollout engine + sample-decoding helpers (reused from the short-audio rollout).
from recipe.phimm.asr_rollout import _normalize_config, _ntb_get, _resolve_model_path

# Long-recording grouping / scoring helpers (reused from the gen-style long eval).
from recipe.phimm.main_long_eval_asr import (
    _parent_key,
    _seg_start,
    score_segments,
    write_results,
)
from recipe.phimm.utils.env import EnvMgr
from recipe.phimm.utils.shared import parse_asr_response
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.utils import hf_tokenizer

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = logging.getLogger(__name__)


def _decode_rollout_segment(sample_bytes, tokenizer) -> dict:
    """Decode a rollout sample into a segment record for :func:`score_segments`.

    Mirrors the response cleaning used by ``main_long_eval_asr`` (``<TXT>``
    parse via :func:`parse_asr_response`, ``<nonspeech>`` stripping) so the
    concatenated per-parent hypotheses score identically to the gen-style long
    eval.
    """
    rollout_sample = ray.cloudpickle.loads(sample_bytes)
    ret = rollout_sample.full_batch

    response_ids = ret.batch.get("responses")
    raw_response = ""
    if response_ids is not None:
        raw_response = tokenizer.decode(
            [t for t in response_ids[0].tolist() if t != 0], skip_special_tokens=True
        )
        eos = tokenizer.eos_token
        if eos and raw_response.endswith(eos):
            raw_response = raw_response[: -len(eos)]
    response_str = parse_asr_response(raw_response).get("text") or ""
    response_str = re.sub(r"<nonspeech>", "", response_str, flags=re.IGNORECASE).strip()

    ntb = ret.non_tensor_batch

    extra = _ntb_get(ntb, "extra_info")
    extra = extra if isinstance(extra, dict) else {}

    reward_model_data = _ntb_get(ntb, "reward_model")
    if isinstance(reward_model_data, dict):
        ref = reward_model_data.get("ground_truth", reward_model_data.get("gt_output", "")) or ""
    else:
        ref = _ntb_get(ntb, "ground_truth") or _ntb_get(ntb, "text") or extra.get("ground_truth") or ""

    audio_path = (
        _ntb_get(ntb, "audio_path")
        or _ntb_get(ntb, "audio")
        or extra.get("audio_path")
        or ""
    )
    data_source = _ntb_get(ntb, "data_source", "") or extra.get("data_source", "")

    fallback = f"__sample_{rollout_sample.sample_id}__"
    return {
        "parent": _parent_key(extra, audio_path, fallback=fallback),
        "seg_start": _seg_start(extra),
        "raw_response": raw_response,
        "response": response_str,
        "ref": str(ref) if ref else "",
        "audio_path": audio_path,
        "data_source": data_source,
        "id": extra.get("id"),
        "language": extra.get("language"),
    }


async def _consume_segments(
    mq_client: MessageQueueClient,
    tokenizer,
    rollouter,
    output_dir: str,
    measure_kwargs: dict,
    log_interval: int = 100,
) -> tuple:
    """Drain the MessageQueue, then group by parent recording and score.

    Unlike the short-audio rollout consumer (which writes parquet parts
    incrementally), long-audio scoring needs *all* segments of a recording
    before it can concatenate and score, so we collect every decoded segment
    first and run :func:`score_segments` / :func:`write_results` once the queue
    is fully drained.
    """
    segments: list = []
    t0 = time.time()

    while True:
        result = await mq_client.get_sample()
        if result is None:
            break
        sample_bytes, _ = result
        if sample_bytes is None:
            break

        segments.append(_decode_rollout_segment(sample_bytes, tokenizer))
        if len(segments) % log_interval == 0:
            rate = len(segments) / max(time.time() - t0, 1e-6)
            print(f"[Consumer] {len(segments)} segments | {rate:.1f} seg/s")

    logger.info("Drained %d segment hypotheses; grouping by parent recording.", len(segments))
    results_by_source = score_segments(segments, measure_kwargs)
    write_results(results_by_source, output_dir)

    tot = {"dter_n_err": 0, "dter_n_ref": 0, "eer_n_err": 0, "eer_n_ref": 0, "n": 0}
    for res in results_by_source.values():
        m = res["measure"]
        tot["dter_n_err"] += m["dter_n_err"]
        tot["dter_n_ref"] += m["dter_n_ref"]
        tot["eer_n_err"] += m["eer_n_err"]
        tot["eer_n_ref"] += m["eer_n_ref"]
        tot["n"] += m["n_recordings"]
    summary = {
        "n_segments": len(segments),
        "n_recordings": tot["n"],
        "dter": tot["dter_n_err"] / max(tot["dter_n_ref"], 1),
        "dter_n_err": tot["dter_n_err"],
        "dter_n_ref": tot["dter_n_ref"],
        "eer": tot["eer_n_err"] / max(tot["eer_n_ref"], 1),
        "eer_n_err": tot["eer_n_err"],
        "eer_n_ref": tot["eer_n_ref"],
    }
    summary_path = f"{output_dir}/summary.json"
    bf.makedirs(os.path.dirname(summary_path.rstrip("/")))
    with bf.BlobFile(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    rate = len(segments) / max(time.time() - t0, 1e-6)
    print(
        f"\nDone: {len(segments)} segments | {rate:.1f} seg/s | "
        f"overall DTER {summary['dter']:.2%} EER {summary['eer']:.2%} "
        f"on {summary['n_recordings']} recordings | summary: {summary_path}"
    )
    return segments, results_by_source


async def _run_long_asr_rollout(config) -> None:
    OmegaConf.resolve(config)
    _normalize_config(config)

    local_model_path = _resolve_model_path(config.actor_rollout_ref.model.path.rstrip("/"))
    OmegaConf.update(config, "actor_rollout_ref.model.path", local_model_path)
    tokenizer = hf_tokenizer(
        local_model_path,
        trust_remote_code=config.actor_rollout_ref.model.get("trust_remote_code", False),
    )

    output_dir = OmegaConf.select(config, "data.output_path", default=None)
    assert output_dir is not None, "Please specify data.output_path"
    output_dir = output_dir.rstrip("/")

    measure_kwargs = config.data.get("measure_kwargs", {})
    if OmegaConf.is_config(measure_kwargs):
        measure_kwargs = OmegaConf.to_container(measure_kwargs, resolve=True)
    log_interval = config.data.get("log_interval", 100)

    mq_actor = MessageQueue.remote(
        config=config, max_queue_size=config.async_training.get("max_queue_size", 1000)
    )
    mq_client = MessageQueueClient(mq_actor)

    rollouter = FullyAsyncRollouter.remote(
        config=config, tokenizer=tokenizer, processor=None, local_model_path=local_model_path
    )
    await rollouter.init_workers.remote()
    await rollouter.set_message_queue_client.remote(mq_client)
    await rollouter.set_max_required_samples.remote()
    ray.get(rollouter.load_checkpoint.remote())

    rollouter_future = rollouter.fit.remote()
    await _consume_segments(mq_client, tokenizer, rollouter, output_dir, measure_kwargs, log_interval)

    try:
        await asyncio.wrap_future(rollouter_future.future())
    except Exception as e:  # noqa: BLE001
        print(f"[LongASRRollout] Rollouter: {e}")
    await mq_client.shutdown()


@hydra.main(config_path="config/rollout", config_name="long_rollout_test", version_base=None)
def main(config):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {}, {}))

    env_vars = EnvMgr().envs()

    if not ray.is_initialized():
        ray_init_kwargs = (
            OmegaConf.to_container(config.get("ray_kwargs", {}).get("ray_init", {}), resolve=True) or {}
        )
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "HF_HUB_OFFLINE": "1",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                **env_vars,
            },
            "excludes": [str(Path(__file__).parents[2] / ".git")],
            **ray_init_kwargs.pop("runtime_env", {}),
        }
        ray_init_kwargs["runtime_env"] = runtime_env
        ray.init(**ray_init_kwargs)

    pprint(OmegaConf.to_container(config, resolve=True))
    asyncio.run(_run_long_asr_rollout(config))


if __name__ == "__main__":
    main()

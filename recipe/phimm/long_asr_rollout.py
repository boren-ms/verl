"""Long-audio ASR rollout via FullyAsyncRollouter + MessageQueue consumer.

This combines two existing recipes:

* the fully-async (server-based) rollout engine from
  :mod:`recipe.phimm.asr_rollout` (``FullyAsyncRollouter`` feeding a
  ``MessageQueue`` that a consumer drains), and
* the long-recording segmentation / per-parent regrouping + DTER/EER scoring
  (grouping + scoring helpers defined below in this module).

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
   :func:`write_results`).

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
from collections import defaultdict
from pprint import pprint

import blobfile as bf
import hydra
import ray
from omegaconf import OmegaConf

# Rollout engine + sample-decoding helpers (reused from the short-audio rollout).
from recipe.phimm.asr_rollout import (
    _ntb_get,
    init_ray,
    prepare_model,
    run_rollout_engine,
)
from recipe.phimm.reward.asr_inhouse_measure import eval_score
from recipe.phimm.utils.shared import parse_asr_response
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.trainer.ppo.reward import get_reward_fn_dispatcher

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Long-audio grouping + scoring helpers.
#
# These regroup exploded segments by parent recording, concatenate the
# per-segment hypotheses, score each recording once against the full reference
# (DisfluencyTolerant TER + entity EER via
# :func:`recipe.phimm.reward.asr_inhouse_measure.eval_score`), and write
# per-data-source ``details.jsonl`` + ``measures.json`` artifacts.
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


def _micro(a: dict) -> dict:
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


def score_segments(segments: list, score_fn=eval_score) -> dict:
    """Group segments by parent, concat hyps, score once per recording.

    Returns a dict keyed by ``data_source`` mapping to
    ``{"details": [...], "measure": {...}}`` (per-recording detail list plus the
    micro-averaged DTER/EER or WER for that source).
    """
    groups: dict = defaultdict(list)
    for seg in segments:
        groups[seg["parent"]].append(seg)

    details_by_source: dict = defaultdict(list)
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

    for parent, members in groups.items():
        members.sort(key=lambda m: m["seg_start"])
        concat_hyp = " ".join(m["response"].strip() for m in members if m["response"].strip())
        responses = [m["response"] for m in members]
        head = members[0]
        ref = head["ref"]
        data_source = head["data_source"] or "all"

        extra_info = {"language": head["language"]} if head["language"] else {}
        score = score_fn(concat_hyp, ref, data_source=data_source, extra_info=extra_info)

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
    score_fn=eval_score,
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
    results_by_source = score_segments(segments, score_fn=score_fn)
    write_results(results_by_source, output_dir)

    measures = [res["measure"] for res in results_by_source.values()]
    n_recordings = sum(m["n_recordings"] for m in measures)
    if measures and "wer" in measures[0]:
        n_err = sum(m["n_err"] for m in measures)
        n_ref = sum(m["n_ref"] for m in measures)
        summary = {
            "n_segments": len(segments),
            "n_recordings": n_recordings,
            "wer": n_err / max(n_ref, 1),
            "n_err": n_err,
            "n_ref": n_ref,
        }
        metric_text = f"overall WER {summary['wer']:.2%}"
    else:
        dter_n_err = sum(m["dter_n_err"] for m in measures)
        dter_n_ref = sum(m["dter_n_ref"] for m in measures)
        eer_n_err = sum(m["eer_n_err"] for m in measures)
        eer_n_ref = sum(m["eer_n_ref"] for m in measures)
        summary = {
            "n_segments": len(segments),
            "n_recordings": n_recordings,
            "dter": dter_n_err / max(dter_n_ref, 1),
            "dter_n_err": dter_n_err,
            "dter_n_ref": dter_n_ref,
            "eer": eer_n_err / max(eer_n_ref, 1),
            "eer_n_err": eer_n_err,
            "eer_n_ref": eer_n_ref,
        }
        metric_text = f"overall DTER {summary['dter']:.2%} EER {summary['eer']:.2%}"
    summary_path = f"{output_dir}/summary.json"
    _write_json(summary, summary_path)

    rate = len(segments) / max(time.time() - t0, 1e-6)
    print(
        f"\nDone: {len(segments)} segments | {rate:.1f} seg/s | "
        f"{metric_text} "
        f"on {summary['n_recordings']} recordings | summary: {summary_path}"
    )
    return segments, results_by_source


async def _run_long_asr_rollout(config) -> None:
    local_model_path, tokenizer = prepare_model(config)

    output_dir = OmegaConf.select(config, "data.output_path", default=None)
    assert output_dir is not None, "Please specify data.output_path"
    output_dir = output_dir.rstrip("/")

    log_interval = config.data.get("log_interval", 100)
    if config.get("reward_functions"):
        reward_dispatcher = get_reward_fn_dispatcher(config)
    else:
        reward_dispatcher = None

    if reward_dispatcher is not None:

        def score_fn(solution_str, ground_truth, *, data_source, **kwargs):
            return reward_dispatcher(data_source, solution_str, ground_truth, **kwargs)

    else:
        def score_fn(solution_str, ground_truth, *, data_source, **kwargs):
            return eval_score(solution_str, ground_truth, data_source=data_source, **kwargs)

    async def consume(mq_client, rollouter):
        await _consume_segments(
            mq_client,
            tokenizer,
            rollouter,
            output_dir,
            score_fn=score_fn,
            log_interval=log_interval,
        )

    await run_rollout_engine(config, tokenizer, local_model_path, consume, tag="LongASRRollout")


@hydra.main(config_path="config/rollout", config_name="long_rollout_test", version_base=None)
def main(config):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {}, {}))

    init_ray(config)
    pprint(OmegaConf.to_container(config, resolve=True))
    asyncio.run(_run_long_asr_rollout(config))


if __name__ == "__main__":
    main()

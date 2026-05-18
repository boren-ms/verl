"""
ASR evaluation client for the vLLM multi-node serving setup.

Sends lightweight audio path requests to the proxy, which routes to worker
sidecars that handle the actual audio loading on the GPU node. This means
the client only sends tiny JSON payloads — no audio I/O or base64 encoding.

Data flow:
  Client: {"audio_path": "az://...", "prompt": "..."} → Proxy → Worker → vLLM
  Client only does: dataset iteration + WER scoring (both fast, CPU-only)

GPU utilization optimizations (targeting >90%):
- Client sends audio paths, not audio data — no client-side I/O bottleneck.
- Workers load audio locally (co-located with GPU) — fast blob/disk access.
- High concurrency (512 default, ~64 per GPU): keeps vLLM batches full.
- Continuous streaming: all requests fire concurrently, semaphore governs parallelism.

Configuration is loaded from ``eval.yaml`` via Hydra. Override any
field on the command line with Hydra-style key=value tokens, e.g.::

    python -m recipe.phimm.vllm_server.eval_asr \\
        eval.proxy_url=http://proxy-host:8000 \\
        data.num_egs=100 \\
        data.max_concurrent=256
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import blobfile as bf
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parents[3]))

from datasets import Dataset

logger = logging.getLogger("eval_asr")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def extract_response_text(result: dict) -> str:
    """Extract the generated text from OpenAI chat completion response."""
    if "error" in result:
        return ""
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return ""


PART_PREFIX = "data_"
PART_SUFFIX = ".jsonl"
SUMMARY_SUFFIX = ".summary.json"
PART_DIGITS = 6


def _part_path(output_path: str, part_idx: int) -> str:
    return os.path.join(output_path, f"{PART_PREFIX}{part_idx:0{PART_DIGITS}d}{PART_SUFFIX}")


def _part_summary_path(output_path: str, part_idx: int) -> str:
    return os.path.join(output_path, f"{PART_PREFIX}{part_idx:0{PART_DIGITS}d}{SUMMARY_SUFFIX}")


def _part_glob(output_path: str) -> str:
    return os.path.join(output_path, f"{PART_PREFIX}*{PART_SUFFIX}")


def _summarize_rows(rows: list[dict]) -> dict:
    """Aggregate per-part counters + the audio_paths needed for resume."""
    n_err = n_ref = n_edge = 0
    audio_paths: list[str] = []
    for r in rows:
        n_err += int(r.get("n_err", 0) or 0)
        n_ref += int(r.get("n_ref", 0) or 0)
        n_edge += int(r.get("n_edge", 0) or 0)
        ap = r.get("audio_path")
        if ap is not None:
            audio_paths.append(ap)
    return {
        "count": len(rows),
        "n_err": n_err,
        "n_ref": n_ref,
        "n_edge": n_edge,
        "audio_paths": audio_paths,
    }


def write_part(output_path: str, rows: list[dict], part_idx: int) -> str:
    """Write ``rows`` to ``data_NNNNNN.jsonl`` and a matching summary JSON.

    The summary (``data_NNNNNN.summary.json``) holds the per-part WER counters
    and the list of ``audio_path`` values, so resume can rebuild state by
    reading just the summary files instead of re-parsing every JSONL row.
    """
    bf.makedirs(output_path)
    part_file = _part_path(output_path, part_idx)
    with bf.BlobFile(part_file, "w") as f:
        for r in rows:
            row = {
                k: (v if not isinstance(v, float) or not (math.isnan(v) or math.isinf(v)) else str(v))
                for k, v in r.items()
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_file = _part_summary_path(output_path, part_idx)
    with bf.BlobFile(summary_file, "w") as f:
        json.dump(_summarize_rows(rows), f, ensure_ascii=False)
    return part_file


def load_resume_state(output_path: str | None) -> tuple[set[str], int, int, int, int, int]:
    """Rebuild resume state from per-part summary files in ``output_path``.

    For each ``data_NNNNNN.jsonl`` we expect a sibling
    ``data_NNNNNN.summary.json`` written by :func:`write_part`. The summary
    holds the per-part WER counters and the list of completed ``audio_path``
    values, so resume only has to read these small JSON files.

    Falls back to scanning a part's JSONL rows if its summary is missing or
    unreadable (e.g. produced by an older run).

    Returns ``(done_paths, next_part_idx, n_saved, tn_err, tn_ref, tn_edge)``.
    On any failure, returns empty/zero state so the caller starts fresh.
    """
    done_paths: set[str] = set()
    next_part_idx = 0
    n_saved = 0
    tn_err = tn_ref = tn_edge = 0

    if not output_path:
        return done_paths, next_part_idx, n_saved, tn_err, tn_ref, tn_edge

    def _load_from_jsonl(part_file: str) -> dict:
        rows = []
        with bf.BlobFile(part_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return _summarize_rows(rows)

    try:
        if not bf.exists(output_path):
            return done_paths, next_part_idx, n_saved, tn_err, tn_ref, tn_edge
        existing = sorted(bf.glob(_part_glob(output_path)))
        for part in existing:
            stem = os.path.basename(part)
            num_str = stem.removeprefix(PART_PREFIX).removesuffix(PART_SUFFIX)
            try:
                part_idx = int(num_str)
            except ValueError:
                continue
            next_part_idx = max(next_part_idx, part_idx + 1)

            summary_file = _part_summary_path(output_path, part_idx)
            summary: dict | None = None
            if bf.exists(summary_file):
                try:
                    with bf.BlobFile(summary_file, "r") as f:
                        summary = json.load(f)
                except Exception as e:
                    logger.warning("Resume: bad summary %s (%s); falling back to JSONL", summary_file, e)
                    summary = None
            if summary is None:
                summary = _load_from_jsonl(part)

            for ap in summary.get("audio_paths", []) or []:
                if ap and ap not in done_paths:
                    done_paths.add(ap)
                    n_saved += 1
            tn_err += int(summary.get("n_err", 0) or 0)
            tn_ref += int(summary.get("n_ref", 0) or 0)
            tn_edge += int(summary.get("n_edge", 0) or 0)

        if existing:
            logger.info(
                "Resume: loaded %d rows from %d existing part summary(ies); next part index=%d",
                n_saved,
                len(existing),
                next_part_idx,
            )
    except Exception as e:
        logger.warning("Resume: failed to scan %s (%s); starting fresh", output_path, e)
        return set(), 0, 0, 0, 0, 0

    return done_paths, next_part_idx, n_saved, tn_err, tn_ref, tn_edge


def wait_proxy_ready(proxy_url: str, timeout: float = 600.0, poll_interval: float = 2.0) -> bool:
    """Poll the proxy's ``/ready`` endpoint until it returns 200 (≥1 healthy
    backend) or ``timeout`` seconds elapse.
    """
    import httpx

    logger.info("Waiting for proxy %s to become ready (timeout=%.0fs)...", proxy_url, timeout)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{proxy_url}/ready", timeout=10)
            if r.status_code == 200:
                logger.info("Proxy ready: %s", r.json())
                return True
        except httpx.HTTPError as e:
            logger.debug("Proxy not reachable yet: %s", e)
        time.sleep(poll_interval)
    logger.error("Timed out waiting for proxy %s to become ready", proxy_url)
    return False


def load_dataset(source_config, num_egs: int | None = None) -> Dataset:
    """Load an eval dataset from a source_config (YAML path, dict, or list).

    Strips verl_format post-processing (eval only needs raw fields), applies
    ``num_egs`` to each source's loader, concatenates a dict-of-datasets into
    one, and truncates to ``num_egs`` if the loader didn't already.
    """
    from recipe.phimm.data.dataset import create_datasets
    from recipe.phimm.cache_dataset import _load_source_config, _strip_verl_format
    from datasets import concatenate_datasets

    if source_config is None:
        raise ValueError("source_config is required (path to a dataset YAML or an inline dict).")
    if OmegaConf.is_config(source_config):
        source_config = OmegaConf.to_container(source_config, resolve=True)
    logger.info("Loading dataset from source_config=%s", source_config)
    dataset_config = _strip_verl_format(_load_source_config(source_config))
    if num_egs:
        if isinstance(dataset_config, dict):
            dataset_config["num_egs"] = int(num_egs)
        elif isinstance(dataset_config, list):
            for item in dataset_config:
                if isinstance(item, dict):
                    item["num_egs"] = int(num_egs)

    ds_obj = create_datasets(dataset_config)
    if isinstance(ds_obj, dict):
        ds_obj = concatenate_datasets(list(ds_obj.values()))
    if num_egs and len(ds_obj) > int(num_egs):
        ds_obj = ds_obj.select(range(int(num_egs)))
    return ds_obj


async def run_evaluation(cfg: DictConfig):
    """Pipelined evaluation: sends audio paths to workers, scores results locally."""
    import httpx
    from recipe.phimm.reward.asr_edge import eval_score
    from recipe.phimm.utils.shared import parse_asr_response

    proxy_url = cfg.eval.proxy_url
    max_concurrent = int(cfg.data.max_concurrent)
    max_tokens = int(cfg.data.max_tokens)
    max_audio_dur = float(cfg.data.max_audio_dur)
    num_egs = cfg.data.get("num_egs")
    output_path = cfg.data.get("output_path")
    log_interval = max(int(cfg.eval.get("log_interval", 100)), 1)
    save_interval = max(int(cfg.eval.get("save_interval", 10000)), 1)
    resume = bool(cfg.eval.get("resume", True))
    wer_kwargs = cfg.eval.get("wer_kwargs", {}) or {}
    if OmegaConf.is_config(wer_kwargs):
        wer_kwargs = OmegaConf.to_container(wer_kwargs, resolve=True)

    # Always wait for the proxy to report ≥1 healthy backend before sending traffic.
    if not wait_proxy_ready(proxy_url, timeout=float(cfg.eval.get("wait_timeout", 600.0))):
        raise RuntimeError(f"Proxy {proxy_url} did not become ready")

    dataset = load_dataset(cfg.data.get("source_config"), num_egs=num_egs)
    total = len(dataset)
    logger.info("Loaded %d samples (wer_kwargs=%s)", total, wer_kwargs)

    # High-concurrency client
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(600.0, connect=30.0),
        limits=httpx.Limits(max_connections=max_concurrent + 50, max_keepalive_connections=100),
    )
    semaphore = asyncio.Semaphore(max_concurrent)

    # Shared counters
    completed = 0
    t_start = time.time()

    # Resume support: scan existing part files in output_path and skip their
    # audio_paths. Each flush writes a *new* part file rather than rewriting a
    # single jsonl, so resume is just "load every part already on disk".
    if resume:
        done_paths, next_part_idx, n_saved, tn_err, tn_ref, tn_edge = load_resume_state(output_path)
    else:
        done_paths, next_part_idx, n_saved, tn_err, tn_ref, tn_edge = set(), 0, 0, 0, 0, 0

    pending_rows: list[dict] = []
    flush_lock = asyncio.Lock()

    async def _flush_pending(force: bool = False) -> None:
        """Drain ``save_interval`` rows (or all if ``force``) into a new part file."""
        nonlocal n_saved, next_part_idx
        if not output_path:
            return
        threshold = 1 if force else save_interval
        if len(pending_rows) < threshold:
            return
        async with flush_lock:
            batch_size = len(pending_rows) if force else save_interval
            if len(pending_rows) < batch_size:
                return
            batch = pending_rows[:batch_size]
            del pending_rows[:batch_size]
            part_idx = next_part_idx
            next_part_idx += 1
            try:
                part_file = await asyncio.to_thread(write_part, output_path, batch, part_idx)
                n_saved += len(batch)
                logger.info(
                    "Checkpoint: wrote %d rows to %s (total saved=%d)",
                    len(batch),
                    part_file,
                    n_saved,
                )
            except Exception as e:
                logger.error("Failed to write part %d: %s", part_idx, e)
                # Put rows back so they're retried at the next flush.
                pending_rows[:0] = batch
                next_part_idx = part_idx

    async def _process_one(idx: int):
        """Send one audio path to worker, score the result."""
        nonlocal tn_err, tn_ref, tn_edge, completed

        sample = dataset[idx]
        audio_path = sample.get("audio_path") or sample.get("audio_file") or sample.get("audio_chunk")
        text = sample.get("Transcription") or sample.get("text")
        prompt = sample.get("prompt")
        assert audio_path, f"sample {idx} missing audio_path/audio_file/audio_chunk"
        assert prompt, f"sample {idx} missing prompt (set add_task_info in the source_config)"
        # Strip the <audio>\n placeholder added by add_task_info — the worker
        # server builds the audio chat content separately.
        if prompt.startswith("<audio>\n"):
            prompt = prompt[len("<audio>\n"):]
        if audio_path in done_paths:
            return
        payload = {
            "audio_path": audio_path,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "max_audio_dur": max_audio_dur,
        }
        async with semaphore:
            try:
                resp = await client.post(
                    f"{proxy_url}/asr/transcribe",
                    json=payload,
                    timeout=600.0,
                )
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                logger.error("Sample %d failed: %s", idx, e)
                result = {"error": str(e)}

        # Score (CPU-only, fast)
        response_str = extract_response_text(result)
        score = eval_score(response_str, text, **wer_kwargs)
        row = {
            "audio_path": audio_path,
            "text": text,
            "prompt": prompt,
            "response": parse_asr_response(response_str).get("text"),
            "raw_response": response_str,
            **score,
        }

        tn_err += row["n_err"]
        tn_ref += row["n_ref"]
        tn_edge += row["n_edge"]
        completed += 1
        done_paths.add(audio_path)
        pending_rows.append(row)

        await _flush_pending()

        if completed % log_interval == 0:
            elapsed = time.time() - t_start
            rps = completed / max(elapsed, 0.1)
            wer = tn_err / max(tn_ref, 1)
            logger.info(
                "%d/%d (%.1f req/s) | WER: %.2f%% [%d/%d] | Edge: %.2f%%",
                completed,
                total,
                rps,
                wer * 100,
                tn_err,
                tn_ref,
                tn_edge / max(tn_ref, 1) * 100,
            )

    try:
        # Fire ALL requests concurrently — semaphore governs parallelism
        logger.info("Launching %d requests (max_concurrent=%d)", total, max_concurrent)
        tasks = [asyncio.create_task(_process_one(i)) for i in range(total)]
        await asyncio.gather(*tasks, return_exceptions=True)

        for i, t in enumerate(tasks):
            if t.exception():
                logger.error("Sample %d exception: %s", i, t.exception())

    finally:
        await client.aclose()

    # Final summary
    elapsed = time.time() - t_start
    overall_wer = tn_err / max(tn_ref, 1)
    overall_edge = tn_edge / max(tn_ref, 1)
    total_done = n_saved + len(pending_rows)
    logger.info(
        "=== FINAL RESULTS ===\n  WER: %.2f%% [%d/%d]\n  Edge WER: %.2f%%\n  Total samples: %d\n  Elapsed: %.1fs (%.1f req/s)",
        overall_wer * 100,
        tn_err,
        tn_ref,
        overall_edge * 100,
        total_done,
        elapsed,
        total_done / max(elapsed, 0.1),
    )

    # Save results: flush any leftover rows as a final part file.
    if output_path:
        if pending_rows:
            await _flush_pending(force=True)
        logger.info("Saved %d total rows across part files in %s", n_saved, output_path)

        summary_file = os.path.join(output_path, "summary.json")
        with bf.BlobFile(summary_file, "w") as f:
            json.dump(
                {
                    "wer": overall_wer,
                    "edge_wer": overall_edge,
                    "n_err": tn_err,
                    "n_ref": tn_ref,
                    "n_edge": tn_edge,
                    "total_samples": total_done,
                    "elapsed_s": elapsed,
                    "req_per_sec": total_done / max(elapsed, 0.1),
                },
                f,
                indent=2,
            )
        logger.info("Saved summary to %s", summary_file)


@hydra.main(config_path=".", config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    asyncio.run(run_evaluation(cfg))


if __name__ == "__main__":
    main()

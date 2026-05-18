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

Usage:
    python -m recipe.phimm.vllm_server.eval_asr \\
        --proxy-url http://proxy-host:8000 \\
        --model /path/to/model \\
        --data-tsv az://orngwus2cresco/data/boren/data/LibriSpeech/asr_train_transcribe.tsv \\
        --output-path /tmp/eval_results \\
        --max-concurrent 512
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

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


async def run_evaluation(args):
    """Pipelined evaluation: sends audio paths to workers, scores results locally."""
    import httpx
    from recipe.phimm.reward.asr_edge import eval_score
    from recipe.phimm.utils.shared import parse_asr_response
    from recipe.phimm.data.dataset import create_audio_dataset
    from recipe.phimm.data.prompts import get_task_prompt

    # Load dataset metadata (no audio loading here — workers do that)
    logger.info("Loading dataset from %s", args.data_tsv)
    ds_conf = {
        "dataset_name": "tsv",
        "tsv_paths": args.data_tsv,
        "add_task_info": {"task": "asr"},
        "post_process": {
            "add_field": {"fields": {"data_source": "asr"}},
            "verl_format": {"prompt_key": "prompt"},
        },
    }
    if args.num_egs:
        ds_conf["num_egs"] = args.num_egs

    dataset = create_audio_dataset(**ds_conf)
    # Ensure num_egs limit is applied (create_audio_dataset may not filter before map)
    if args.num_egs and len(dataset) > args.num_egs:
        dataset = dataset.select(range(args.num_egs))
    total = len(dataset)
    logger.info("Loaded %d samples", total)

    prompt_text = get_task_prompt("asr", rand=False)

    # High-concurrency client
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(600.0, connect=30.0),
        limits=httpx.Limits(max_connections=args.max_concurrent + 50, max_keepalive_connections=100),
    )
    semaphore = asyncio.Semaphore(args.max_concurrent)

    wer_kwargs = {}
    if args.text_norm:
        wer_kwargs["text_norm"] = args.text_norm

    # Shared counters
    all_results = []
    tn_err, tn_ref, tn_edge = 0, 0, 0
    completed = 0
    t_start = time.time()
    log_interval = max(args.log_interval, 1)

    async def _process_one(idx: int):
        """Send one audio path to worker, score the result."""
        nonlocal tn_err, tn_ref, tn_edge, completed

        sample = dataset[idx]
        audio_path = sample.get("audio_path", sample.get("audio_file", ""))
        # Ground truth can be in multiple places depending on dataset format
        reward_model = sample.get("reward_model", {})
        if isinstance(reward_model, str):
            import json as _json
            try:
                reward_model = _json.loads(reward_model)
            except Exception:
                reward_model = {}
        text = reward_model.get("ground_truth", "") if reward_model else ""
        if not text:
            text = sample.get("Transcription", sample.get("text", ""))

        # Lightweight request: just the audio path (worker does the heavy lifting)
        payload = {
            "audio_path": audio_path,
            "prompt": prompt_text,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "max_audio_dur": args.max_audio_dur,
        }

        async with semaphore:
            try:
                resp = await client.post(
                    f"{args.proxy_url}/asr/transcribe",
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
        score["response"] = parse_asr_response(response_str).get("text")
        score["raw_response"] = response_str
        row = {"text": text, "audio_path": audio_path, "prompt": prompt_text, **score}

        tn_err += row["n_err"]
        tn_ref += row["n_ref"]
        tn_edge += row["n_edge"]
        completed += 1
        all_results.append(row)

        if completed % log_interval == 0:
            elapsed = time.time() - t_start
            rps = completed / max(elapsed, 0.1)
            wer = tn_err / max(tn_ref, 1)
            logger.info(
                "%d/%d (%.1f req/s) | WER: %.2f%% [%d/%d] | Edge: %.2f%%",
                completed, total, rps, wer * 100, tn_err, tn_ref,
                tn_edge / max(tn_ref, 1) * 100,
            )

    try:
        # Fire ALL requests concurrently — semaphore governs parallelism
        logger.info("Launching %d requests (max_concurrent=%d)", total, args.max_concurrent)
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
    logger.info(
        "=== FINAL RESULTS ===\n"
        "  WER: %.2f%% [%d/%d]\n"
        "  Edge WER: %.2f%%\n"
        "  Total samples: %d\n"
        "  Elapsed: %.1fs (%.1f req/s)",
        overall_wer * 100, tn_err, tn_ref,
        overall_edge * 100,
        len(all_results),
        elapsed, len(all_results) / max(elapsed, 0.1),
    )

    # Save results
    if args.output_path:
        import blobfile as bf
        bf.makedirs(args.output_path)
        output_file = os.path.join(args.output_path, "result_details.jsonl")
        with bf.BlobFile(output_file, "w") as f:
            for r in all_results:
                row = {k: (v if not isinstance(v, float) or not (math.isnan(v) or math.isinf(v)) else str(v))
                       for k, v in r.items()}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Saved results to %s", output_file)

        summary_file = os.path.join(args.output_path, "summary.json")
        with bf.BlobFile(summary_file, "w") as f:
            json.dump({
                "wer": overall_wer,
                "edge_wer": overall_edge,
                "n_err": tn_err,
                "n_ref": tn_ref,
                "n_edge": tn_edge,
                "total_samples": len(all_results),
                "elapsed_s": elapsed,
                "req_per_sec": len(all_results) / max(elapsed, 0.1),
            }, f, indent=2)
        logger.info("Saved summary to %s", summary_file)


def main():
    parser = argparse.ArgumentParser(description="ASR evaluation via vLLM proxy")
    parser.add_argument("--proxy-url", required=True, help="FastAPI proxy URL")
    parser.add_argument("--model", required=True, help="Model path (for basename in API)")
    parser.add_argument("--model-name", default=None, help="Model name for API")
    parser.add_argument(
        "--data-tsv",
        default="az://orngwus2cresco/data/boren/data/LibriSpeech/asr_train_transcribe.tsv",
        help="Path to LibriSpeech TSV",
    )
    parser.add_argument("--output-path", default=None, help="Output directory for results")
    parser.add_argument("--max-concurrent", type=int, default=512,
                        help="Max concurrent requests (default 512 ≈ 64/GPU × 8 GPUs)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens for generation")
    parser.add_argument("--max-audio-dur", type=float, default=40.0, help="Max audio duration in seconds")
    parser.add_argument("--num-egs", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--text-norm", default=None, help="Text normalization (e.g. openasr_en)")
    parser.add_argument("--log-interval", type=int, default=100, help="Log every N completions")
    args = parser.parse_args()

    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()

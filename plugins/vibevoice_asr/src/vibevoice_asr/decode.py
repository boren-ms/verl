"""Decode native ASR JSONL manifests with either offline model backend."""

import argparse
import gzip
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

from vibevoice_asr import MODEL_REVISION, UPSTREAM_REVISION
from vibevoice_asr.transcription import (
    close_truncated_json,
    collapse_audio_tokens,
    parse_transcription,
    transcription_text,
)

AMI_DEV_MANIFEST = (
    "az://orngwus2cresco/data/boren/data/Training/diarization/moe_en/full_v1/manifests/moe_ami_dev/part-*.jsonl.gz"
)


def read_manifest(pattern):
    import blobfile as bf

    paths = sorted(bf.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
    if not paths:
        raise FileNotFoundError(pattern)
    rows = []
    for path in paths:
        with bf.BlobFile(path, "rb") as stream:
            payload = stream.read()
        if path.endswith(".gz"):
            payload = gzip.decompress(payload)
        rows.extend(json.loads(line) for line in payload.decode().splitlines() if line.strip())
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Manifest contains duplicate sample IDs")
    return rows, paths


def load_waveform(row):
    import numpy as np
    from recipe.phimm.utils.audio import load_raw_audio
    from scipy.signal import resample_poly

    waveform, sr = load_raw_audio(row)
    if sr != 24000:
        import math

        divisor = math.gcd(sr, 24000)
        waveform = resample_poly(waveform, 24000 // divisor, sr // divisor)
    return np.asarray(waveform, dtype=np.float32)


def read_generations(path, rows):
    """Require exact coverage and references before rescoring saved outputs."""
    details = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_id = {detail["id"]: detail for detail in details}
    if len(by_id) != len(details) or set(by_id) != {row["id"] for row in rows}:
        raise ValueError("Saved generations must cover every selected sample exactly once")
    for row in rows:
        detail = by_id[row["id"]]
        if detail["reference"] != row["text"] or detail["audio_path"] != row["audio_path"]:
            raise ValueError("Saved generation reference/audio does not match the manifest")
        if not isinstance(detail["raw_response"], str) or not isinstance(detail["truncated"], bool):
            raise ValueError("Invalid saved generation record")
    return by_id


class Decoder:
    def __init__(self, args):
        import torch

        from vibevoice_asr.compat import prepare_upstream

        prepare_upstream()
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

        self.args = args
        self.processor = VibeVoiceASRProcessor.from_pretrained(args.model)
        if args.backend == "hf":
            import hf_vibevoice_asr  # noqa: F401
            from transformers import AutoConfig, AutoModelForCausalLM

            config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
            # Upstream also reads this field inside encode_speech. Setting only
            # from_pretrained(dtype=...) can leave its encoders in float32.
            config.torch_dtype = torch.bfloat16
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    args.model,
                    config=config,
                    trust_remote_code=False,
                    dtype=torch.bfloat16,
                    attn_implementation=args.attention,
                )
                .to(device=args.device, dtype=torch.bfloat16)
                .eval()
            )
        else:
            os.environ["VLLM_PLUGINS"] = "vibevoice_asr"
            # Ray drivers may initialize CUDA before LLM construction. Fork
            # cannot inherit that CUDA context safely.
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            from vllm import LLM

            self.model = LLM(
                model=args.model,
                trust_remote_code=False,
                dtype="bfloat16",
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
                max_num_seqs=args.batch_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enable_prefix_caching=False,
                enforce_eager=args.enforce_eager,
                limit_mm_per_prompt={"audio": {"count": 1, "length": int(args.max_audio_seconds * 24000)}},
                seed=args.seed,
            )

    def decode(self, waves):
        import torch

        if any(len(wave) > int(self.args.max_audio_seconds * 24000) for wave in waves):
            raise ValueError("Audio exceeds --max-audio-seconds; increase the explicit profiling bound")
        encodings = [self.processor._process_single_audio(wave, sampling_rate=24000) for wave in waves]
        if self.args.backend == "vllm":
            from vllm import SamplingParams

            tokenizer = self.processor.tokenizer
            audio_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
            if audio_id is None or audio_id == tokenizer.unk_token_id:
                raise ValueError("Generate the official vLLM tokenizer assets before decoding")
            requests = []
            for encoding in encodings:
                ids = collapse_audio_tokens(
                    encoding["input_ids"],
                    start_id=self.processor.speech_start_id,
                    end_id=self.processor.speech_end_id,
                    audio_id=audio_id,
                )
                requests.append({"prompt_token_ids": ids, "multi_modal_data": {"audio": [(encoding["speech"], 24000)]}})
            outputs = self.model.generate(
                requests,
                SamplingParams(
                    temperature=0,
                    max_tokens=self.args.max_new_tokens,
                    seed=self.args.seed,
                ),
                use_tqdm=False,
            )
            results = []
            for output, encoding in zip(outputs, encodings, strict=True):
                if list(output.prompt_token_ids) != list(encoding["input_ids"]):
                    raise ValueError("HF/vLLM expanded prompt tokens differ")
                generated = output.outputs[0]
                results.append((generated.text, generated.finish_reason == "length", len(generated.token_ids)))
            return results
        # Official processor supplies correctly padded speech and text masks.
        batch = self.processor(audio=waves, sampling_rate=24000, return_tensors="pt", padding=True)
        batch = {
            key: value.to(self.args.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        prompt_length = batch["input_ids"].shape[1]
        with torch.inference_mode():
            output = self.model.generate(
                **batch,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.pad_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        results = []
        for tokens in output[:, prompt_length:]:
            raw = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
            eos = self.processor.tokenizer.eos_token_id
            results.append((raw, len(tokens) >= self.args.max_new_tokens and eos not in tokens.tolist(), len(tokens)))
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hf", "vllm"), required=True)
    parser.add_argument("--model", required=True, help="Local official checkpoint with generated tokenizer assets")
    parser.add_argument("--manifest", default=AMI_DEV_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rescore-from", type=Path, help="Score complete saved generations without model inference")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--audio-workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-audio-seconds", type=float, default=60)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Invalid batch size or shard selection")
    if args.max_audio_seconds <= 0 or args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Audio duration, generation length and limit must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, paths = read_manifest(args.manifest)
    total_rows = len(rows)
    rows = rows[args.shard_index :: args.num_shards]
    if args.limit is not None:
        rows = rows[: args.limit]
    import numpy as np
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = vars(args).copy()
    provenance_path = Path(args.model) / "vibevoice_provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    config.update(
        output_dir=str(args.output_dir),
        rescore_from=str(args.rescore_from) if args.rescore_from else None,
        manifest_files=paths,
        full_manifest_rows=total_rows,
        selected_rows=len(rows),
        upstream_revision=UPSTREAM_REVISION,
        model_provenance=provenance,
        expected_model_revision=MODEL_REVISION,
        model_config_sha256=hashlib.sha256((Path(args.model) / "config.json").read_bytes()).hexdigest(),
        manifest_rows_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        runtime={
            name: version(name)
            for name in ("torch", "transformers", "numpy", "scipy", "vibevoice", "verl-vibevoice-asr")
            + (("vllm",) if args.backend == "vllm" else ())
        },
        vllm_use_mean_override=os.environ.get("VIBEVOICE_USE_MEAN") if args.backend == "vllm" else None,
        scoring="recipe.phimm.reward.asr_eval.measure_openasr_en_wer; corpus micro WER",
        non_speech_policy="Exclude whole Content fields matching six official acoustic tags or observed [Sigh] event",
        truncation_policy="Keep all emitted speech; append only closing JSON delimiters when necessary and valid",
        event_timing_policy="Retain and flag inverted acoustic-event times; excluded events do not affect text WER",
    )
    saved = None
    if args.rescore_from:
        source = json.loads((args.rescore_from / "config.json").read_text())
        for key in ("manifest_rows_sha256", "model_config_sha256", "backend", "selected_rows"):
            if source[key] != config[key]:
                raise ValueError(f"Rescore source differs in {key}")
        details_path = args.rescore_from / "details.jsonl"
        saved = read_generations(details_path, rows)
        with details_path.open("rb") as stream:
            config["source_details_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        config["inference_config"] = source
    config["run_mode"] = "rescore" if saved is not None else "decode"
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    decoder = Decoder(args) if saved is None else None
    from recipe.phimm.reward.asr_eval import measure_openasr_en_wer

    totals = dict(n_err=0, n_ref=0, n_ins=0, n_del=0, n_sub=0)
    failures = 0
    truncated_rows = 0
    recovered_json_rows = 0
    invalid_event_timing_rows = 0
    start = time.monotonic()
    with (args.output_dir / "details.jsonl").open("w") as stream, ThreadPoolExecutor(args.audio_workers) as pool:
        for offset in range(0, len(rows), args.batch_size):
            samples = rows[offset : offset + args.batch_size]
            if saved is None:
                waves = list(pool.map(load_waveform, samples))
                predictions = decoder.decode(waves)
            else:
                predictions = [
                    (saved[s["id"]]["raw_response"], saved[s["id"]]["truncated"], saved[s["id"]]["generated_tokens"])
                    for s in samples
                ]
            for sample, (raw, truncated, token_count) in zip(samples, predictions, strict=True):
                truncated_rows += int(truncated)
                detail = dict(
                    id=sample["id"],
                    recording_id=sample.get("recording_id"),
                    audio_path=sample["audio_path"],
                    reference=sample["text"],
                    raw_response=raw,
                    truncated=truncated,
                    generated_tokens=token_count,
                )
                try:
                    try:
                        segments = parse_transcription(raw, allow_invalid_event_times=True)
                    except (ValueError, TypeError):
                        if not truncated:
                            raise
                        segments = parse_transcription(close_truncated_json(raw), allow_invalid_event_times=True)
                        detail["parse_recovery"] = "closed_truncated_json_delimiters; no words changed"
                        recovered_json_rows += 1
                    invalid_event_timing_rows += int(any(s.get("timing_valid") is False for s in segments))
                    hypothesis = transcription_text(segments)
                    score = measure_openasr_en_wer(hypothesis, sample["text"])
                    detail.update(segments=segments, hypothesis=hypothesis, **score)
                    for key in totals:
                        totals[key] += score[key]
                except (ValueError, TypeError) as exc:
                    failures += 1
                    detail["error"] = str(exc)
                stream.write(json.dumps(detail, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                json.dumps(
                    dict(
                        completed=min(offset + len(samples), len(rows)),
                        total=len(rows),
                        failures=failures,
                        elapsed_s=round(time.monotonic() - start, 1),
                    )
                ),
                flush=True,
            )
    summary = dict(
        backend=args.backend,
        run_mode=config["run_mode"],
        full_manifest_rows=total_rows,
        selected_rows=len(rows),
        scored_rows=len(rows) - failures,
        failed_rows=failures,
        truncated_rows=truncated_rows,
        recovered_truncated_json_rows=recovered_json_rows,
        all_generations_finished=truncated_rows == 0,
        invalid_event_timing_rows=invalid_event_timing_rows,
        **totals,
        wer=totals["n_err"] / totals["n_ref"] if totals["n_ref"] and not failures else None,
        complete=not failures,
        full_dataset=args.limit is None and args.num_shards == 1,
        elapsed_s=time.monotonic() - start,
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if failures:
        raise SystemExit("Invalid generations retained in details.jsonl; full WER unavailable")


if __name__ == "__main__":
    main()

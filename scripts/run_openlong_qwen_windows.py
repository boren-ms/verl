#!/usr/bin/env python3
"""Run Qwen forced alignment on prepared OpenLong windows.

Produces word timestamps only, not final segment labels. Audio is read from the
existing ZIP members into memory; no standalone parent audio files are created.
"""

import argparse
import asyncio
import hashlib
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_zip_member(source):
    match = re.fullmatch(r"(.+\.zip)!(\d+):([1-9]\d*)", source, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid ZIP audio source: {source}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def read_zip_member(source, chunk_bytes=8 * 1024 * 1024, concurrency=8):
    """Fetch only the stored member's byte ranges, without importing verl."""
    from boostedblob.globals import session_context
    from boostedblob.read import read_byte_range

    archive, offset, size = parse_zip_member(source)

    async def fetch():
        async with session_context():
            sem = asyncio.Semaphore(concurrency)

            async def chunk(start):
                async with sem:
                    return await read_byte_range(archive, (start, min(start + chunk_bytes, offset + size)))

            parts = await asyncio.gather(*(chunk(start) for start in range(offset, offset + size, chunk_bytes)))
            return b"".join(parts)

    content = asyncio.run(fetch())
    if len(content) != size:
        raise ValueError(f"Short ZIP member read: expected {size} bytes, got {len(content)}")
    return content


def shard_for(dataset, parent, shard_count):
    value = hashlib.sha256(f"{dataset}\0{parent}".encode()).digest()
    return int.from_bytes(value[:8], "big") % shard_count


class DecodedAudio:
    """Seek over already-decoded samples, avoiding MP3 bit-reservoir seek errors."""

    def __init__(self, samples, samplerate):
        self.samples = samples
        self.samplerate = samplerate
        self.frames = len(samples)
        self.position = 0

    def seek(self, frame):
        self.position = frame

    def read(self, frames, dtype="float32", always_2d=False):
        data = self.samples[self.position:self.position + frames]
        self.position += len(data)
        if always_2d and data.ndim == 1:
            data = data[:, None]
        return data.astype(dtype, copy=False)


def read_completed(output, planned):
    recovered = defaultdict(list)
    with output.open(encoding="utf-8") as handle:
        for line in handle:
            result = json.loads(line)
            w = result["window"]
            original_ids = w.get("retry_original_core_segment_ids", w["core_segment_ids"])
            key = (w["dataset"], w["parent_audio_path"], tuple(original_ids))
            if key not in planned or result["status"] not in ("ok", "empty_seed", "unalignable_seed"):
                raise ValueError(f"Existing result does not match this shard: {key}")
            recovered[key].extend(w["core_segment_ids"])
    for key, core_ids in recovered.items():
        if len(core_ids) != len(set(core_ids)) or set(core_ids) != set(key[2]):
            raise ValueError(f"Incomplete or duplicated previous window: {key}")
    return set(recovered)


def align_window(aligner, window, audio_file):
    """Align one window; timestamps are returned relative and absolute to its parent."""
    start = float(window["audio_start"])
    end = float(window["audio_end"])
    if not 0 <= start < end or end - start > 180:
        raise ValueError(f"Invalid Qwen window: {start}..{end}")
    if not window["seed_text"].strip():
        return {"status": "empty_seed", "words": []}
    # Qwen's own tokenizer may discard punctuation-only seed text.
    if not aligner.aligner_processor.tokenize_space_lang(window["seed_text"]):
        return {"status": "unalignable_seed", "words": []}
    rate = audio_file.samplerate
    begin_frame, end_frame = int(start * rate), int(end * rate)
    if end_frame > audio_file.frames:
        raise ValueError(f"Window exceeds audio duration: {end} > {audio_file.frames / rate}")
    audio_file.seek(begin_frame)
    audio = audio_file.read(end_frame - begin_frame, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    if len(audio) != end_frame - begin_frame:
        raise ValueError("Short audio read")
    result = aligner.align(audio=(audio, rate), text=window["seed_text"], language="English")
    if len(result) != 1:
        raise ValueError("Qwen returned an unexpected batch size")
    duration = end - start
    words = []
    previous_start = -1.0
    outside_audio = 0
    max_excess = 0.0
    for item in result[0]:
        a, b = float(item.start_time), float(item.end_time)
        if not math.isfinite(a) or not math.isfinite(b) or a < 0 or b < a or a < previous_start:
            raise ValueError(f"Invalid or nonmonotonic Qwen timestamps: {a}..{b} for {duration}s")
        clipped = a > duration or b > duration
        if clipped:
            outside_audio += 1
            max_excess = max(max_excess, a - duration, b - duration)
        safe_start, safe_end = min(a, duration), min(b, duration)
        words.append({"text": item.text, "start": safe_start, "end": safe_end,
                      "parent_start": start + safe_start, "parent_end": start + safe_end,
                      "predicted_start": a, "predicted_end": b, "clipped_to_window": clipped})
        previous_start = a
    if not words:
        raise ValueError("Qwen returned no words for nonempty tokenized text")
    zero_duration = sum(word["end"] <= word["start"] for word in words)
    collapsed = len(words) >= 5 and zero_duration / len(words) >= 0.8
    return {"status": "ok", "words": words,
            "quality": {"token_count": len(words), "zero_duration": zero_duration, "collapsed": collapsed,
                        "outside_audio": outside_audio, "max_excess_seconds": max_excess}}


def split_retry_windows(window, reason="collapsed_timestamps"):
    """Split a poor multi-clip window into independent single-clip retries."""
    segments = {segment["id"]: segment for segment in window["segments"]}
    result = []
    for sid, index in zip(window["core_segment_ids"], window["core_segment_indexes"], strict=True):
        segment = segments[sid]
        if segment["seg_index"] != index:
            raise ValueError("Retry segment id/index mismatch")
        retry = dict(window)
        retry.update({"audio_start": segment["seg_start"], "audio_end": segment["seg_end"],
                      "seed_text": segment["text"], "core_word_start": 0,
                      "core_word_end": len(segment["text"].split()), "core_segment_ids": [sid],
                      "core_segment_indexes": [index], "context_segment_indexes": [index],
                      "segments": [segment], "retry_reason": reason,
                      "retry_original_core_segment_ids": window["core_segment_ids"]})
        result.append(retry)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--flash-attn", action="store_true", help="Use FlashAttention 2 when installed.")
    parser.add_argument("--dataset", choices=["earnings21", "earnings22", "coraal", "earnings22_cleaned_aa"])
    parser.add_argument("--two-parent-only", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-windows", type=int, help="Limit inference for a pilot; never a complete dataset.")
    parser.add_argument("--resume", action="store_true", help="Continue an existing result file after validating completed windows.")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid shard index/count")
    if args.max_windows is not None and args.max_windows < 1:
        parser.error("max-windows must be positive")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    groups = defaultdict(list)
    with args.windows.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            w = json.loads(line)
            if args.dataset and w["dataset"] != args.dataset:
                continue
            if args.two_parent_only and not w["in_two_parent_subset"]:
                continue
            if shard_for(w["dataset"], w["parent_audio_path"], args.shard_count) != args.shard_index:
                continue
            groups[(w["dataset"], w["parent_audio_path"])].append(w)
    if not groups:
        raise ValueError("No windows matched the requested shard")
    planned = {(w["dataset"], w["parent_audio_path"], tuple(w["core_segment_ids"]))
               for group in groups.values() for w in group}
    completed = set()
    if args.output.exists():
        completed = read_completed(args.output, planned)
        print(f"Resume: {len(completed)} original windows already complete", flush=True)
    for key in list(groups):
        groups[key] = [w for w in groups[key] if (w["dataset"], w["parent_audio_path"],
                       tuple(w["core_segment_ids"])) not in completed]
        if not groups[key]:
            del groups[key]
    if not groups:
        print("All planned windows are already complete.")
        return
    import torch
    from qwen_asr import Qwen3ForcedAligner
    model_kwargs = {"dtype": torch.bfloat16, "device_map": args.device}
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    aligner = Qwen3ForcedAligner.from_pretrained(args.model_path, **model_kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = errors = 0
    with args.output.open("a" if args.output.exists() else "x", encoding="utf-8") as handle:
        for key in sorted(groups):
            windows = groups[key]
            members = {w["audio_source"] for w in windows}
            if len(members) != 1:
                raise ValueError(f"Inconsistent ZIP member: {key}")
            content = read_zip_member(members.pop())
            with sf.SoundFile(io.BytesIO(content)) as audio_file:
                # MP3 seeks can enter a frame without its bit reservoir. Decode
                # the member sequentially once, matching the training loader.
                if audio_file.format == "MP3":
                    samples = audio_file.read(dtype="float32", always_2d=False)
                    if len(samples) != audio_file.frames:
                        raise ValueError(f"Short MP3 decode for {key}")
                    audio = DecodedAudio(samples, audio_file.samplerate)
                else:
                    audio = audio_file
                for window in windows:
                    aligned = align_window(aligner, window, audio)
                    quality = aligned.get("quality", {})
                    if (quality.get("collapsed") or quality.get("max_excess_seconds", 0) > 0.5) and len(window["core_segment_ids"]) > 1:
                        reason = "collapsed_timestamps" if quality.get("collapsed") else "out_of_bounds_timestamps"
                        attempts = split_retry_windows(window, reason=reason)
                    else:
                        attempts = []
                    if not attempts:
                        records = [{"window": window, **aligned}]
                    else:
                        records = []
                        for retry in attempts:
                            retry_result = align_window(aligner, retry, audio)
                            records.append({"window": retry, **retry_result})
                    handle.write("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
                    handle.flush()
                    count += 1
                    if count % 25 == 0:
                        print(f"Processed {count} original windows, {errors} errors", flush=True)
                    if args.max_windows is not None and count >= args.max_windows:
                        break
            if args.max_windows is not None and count >= args.max_windows:
                break
    print(f"Wrote {count} window results, {errors} errors; pilot={args.max_windows is not None}.")


if __name__ == "__main__":
    main()

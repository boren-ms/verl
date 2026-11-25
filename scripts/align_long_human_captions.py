# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Qwen word alignment and adjacent reference-context pairs, without audio files.

Long inputs use existing adjusted caption timings only to locate bounded model
windows. Exact normalized human reference words are assigned through matching
anchors; uncertain mappings are retained as failures, never guessed by duration.
"""

import argparse
import difflib
import gzip
import hashlib
import importlib.metadata
import json
import math
import re
import struct
import unicodedata
from pathlib import Path

MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
MODEL_REVISION = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
QWEN_REVISION = "7c6daf77a2421100f5fb066495372c00129d39ff"


def write_row(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def fingerprint(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def clean_word(word):
    # Match Qwen's English clean_token while retaining the original label.
    return "".join(c for c in word if c == "'" or unicodedata.category(c)[0] in "LN").casefold()


def anchor_tokens(text):
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text.casefold().replace("’", "'"))


def alignment_windows(text, duration, captions, *, maximum=180.0, padding=1.0):
    """Partition exact reference words using conservative caption-boundary anchors."""
    words = text.split()
    if not words or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Empty reference or invalid waveform duration")
    if any(not clean_word(w) for w in words):
        raise ValueError("Reference contains a punctuation-only alignment unit")
    if duration <= maximum:
        return [{"start": 0.0, "end": duration, "word_start": 0, "word_end": len(words)}]
    if not captions:
        raise ValueError("Long recording has no adjusted caption timing spans")
    source, ends = [], []
    last_end = 0.0
    for caption in captions:
        start, end = float(caption["start_time"]), float(caption["end_time"])
        if not 0 <= start < end <= duration + 0.1 or start < last_end - 0.01:
            raise ValueError("Invalid or overlapping caption timing spans")
        last_end = end
        source.extend(anchor_tokens(caption["transcription"]))
        ends.append(len(source))
    target, target_boundaries = [], {0: 0}
    for index, word in enumerate(words):
        target.extend(anchor_tokens(word))
        target_boundaries[len(target)] = index + 1
    matcher = difflib.SequenceMatcher(None, source, target, autojunk=False)
    if matcher.ratio() < 0.90:
        raise ValueError("Caption/reference anchor agreement below 90 percent")
    boundaries = {}
    for block in matcher.get_matching_blocks():
        if block.size < 3:
            continue
        # Require context on both sides of an internal caption boundary.
        for boundary in ends[:-1]:
            if block.a + 1 <= boundary <= block.a + block.size - 1:
                position = block.b + boundary - block.a
                if position in target_boundaries:
                    boundaries[boundary] = target_boundaries[position]
    output, first_caption, first_word = [], 0, 0
    while first_caption < len(captions):
        start = max(0.0, float(captions[first_caption]["start_time"]) - padding)
        choices = []
        for last in range(first_caption, len(captions)):
            end = min(duration, float(captions[last]["end_time"]) + padding)
            if end - start > maximum:
                break
            stop = len(words) if last == len(captions) - 1 else boundaries.get(ends[last])
            if stop is not None and stop > first_word:
                choices.append((last, end, stop))
        if not choices:
            raise ValueError("No reliable caption/reference boundary within model window limit")
        last, end, stop = choices[-1]
        output.append({"start": start, "end": end, "word_start": first_word, "word_end": stop})
        first_caption, first_word = last + 1, stop
    if first_word != len(words):
        raise ValueError("Caption windows do not cover every reference word")
    return output


def segment_row(row, *, maximum=30.0):
    """Add path/text pairs with complete word coverage and no physical audio writes."""
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("Segment duration must be finite and positive")
    words, timestamps = row["text"].split(), row["word_timestamps"]
    duration = float(row["alignment_audio_duration"])
    if len(words) != len(timestamps) or not words:
        raise ValueError("Alignment does not cover the reference words exactly")
    previous, flags = 0.0, []
    for word, timestamp in zip(words, timestamps, strict=True):
        start, end = timestamp["start"], timestamp["end"]
        if clean_word(word) != clean_word(timestamp["text"]):
            raise ValueError("Aligned word differs from the human reference")
        if not all(math.isfinite(t) for t in (start, end)) or not 0 <= start <= end <= duration:
            raise ValueError("Word timestamp outside waveform bounds")
        if start < previous - 0.000001:
            raise ValueError("Overlapping word timestamps")
        if end - start > maximum:
            raise ValueError("One alignment word exceeds the segment duration limit")
        if start == end:
            flags.append("zero_duration_word")
        previous = end
    source = row.get("audio_chunk") or row["audio_path"]
    if "#" in source:
        raise ValueError("Expected a parent audio source without an existing time selector")
    segments, metadata, first = [], [], 0
    while first < len(words):
        start = timestamps[first]["start"]
        stop = first + 1
        while stop < len(words) and timestamps[stop]["end"] - start <= maximum:
            stop += 1
        end = timestamps[stop - 1]["end"]
        if end <= start:
            raise ValueError("Zero-duration segment")
        # Millisecond model outputs remain exact in the emitted selector.
        path = f"{source}#{start:.3f}:{end:.3f}"
        label = " ".join(words[first:stop])
        segments.append([path, label])
        metadata.append({"start": start, "end": end, "word_start": first, "word_end": stop})
        first = stop
    return {
        **row,
        "segments": segments,
        "segment_metadata": metadata,
        "segment_max_seconds": maximum,
        "alignment_flags": sorted(set(flags)),
    }


def filter_word_durations(row, *, minimum=0.04, maximum=3.0):
    """Retain every original and mark the entire parent unsafe if any word is atypical."""
    if not 0 <= minimum < maximum or not math.isfinite(maximum):
        raise ValueError("Invalid word-duration thresholds")
    outliers = []
    for index, word in enumerate(row.get("word_timestamps", [])):
        duration = word["end"] - word["start"]
        if not math.isfinite(duration) or duration < minimum - 1e-6 or duration > maximum + 1e-6:
            outliers.append(
                {
                    "word_index": index,
                    "text": word["text"],
                    "start": word["start"],
                    "end": word["end"],
                    "duration": duration,
                    "reason": "too_short" if duration < minimum else "too_long",
                }
            )
    complete = bool(row.get("word_timestamps")) and len(row["word_timestamps"]) == len(row["text"].split())
    return {
        **row,
        "word_duration_filter": {
            "minimum": minimum,
            "maximum": maximum,
            "scope": "entire_original_recording",
            "passed": complete and not outliers and not row.get("alignment_error"),
            "outliers": outliers,
        },
    }


def paired_rows(row):
    """Sliding adjacent pairs stay within one parent and have exactly two labels."""
    if row.get("alignment_error") or not row.get("word_duration_filter", {}).get("passed", False):
        return
    segments, metadata = row["segments"], row["segment_metadata"]
    for index in range(len(segments) - 1):
        selected = segments[index : index + 2]
        bounds = metadata[index : index + 2]
        yield {
            "id": f"{row['id']}:pair:{index}",
            "parent_id": row["id"],
            "parent_audio_path": row.get("audio_chunk") or row["audio_path"],
            "audio_path": [part[0] for part in selected],
            "text": " ".join(part[1] for part in selected),
            "segment_reference_texts": [part[1] for part in selected],
            "segments": selected,
            "segment_metadata": bounds,
            "segment_indices": [index, index + 1],
            "audio_length": sum(part["end"] - part["start"] for part in bounds),
            "language": "English",
            "keywords": [],
            "pairing_policy": "sliding_adjacent_stride_1_same_parent",
            "transcription_source": row["transcription_source"],
            "alignment_flags": row["alignment_flags"],
            "word_duration_filter": row["word_duration_filter"],
        }


def info_records(path):
    with Path(path).open("rb") as handle:
        if handle.read(8) != b"Info\x01\x00\x00\x00":
            raise ValueError("Unsupported Info chunk header")
        index = 0
        while header := handle.read(8):
            if len(header) != 8:
                raise ValueError("Truncated Info record header")
            record, size = struct.unpack("<II", header)
            payload = handle.read(size)
            if record != index or len(payload) != size:
                raise ValueError("Corrupt Info record")
            yield json.loads(payload)
            index += 1


def prepare(args):
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    handles = [
        gzip.open(root / f"input-{i:04d}.jsonl.gz", "wt", encoding="utf-8", compresslevel=1) for i in range(args.shards)
    ]
    counts, current, infos, chunk_index = [0] * args.shards, None, None, -1
    try:
        for row in rows(args.input):
            if row["chunk_name"] != current:
                if infos is not None and next(infos, None) is not None:
                    raise ValueError("Unconsumed Info records")
                current, chunk_index = row["chunk_name"], chunk_index + 1
                infos = iter(info_records(args.info_dir / (current + ".info")))
            info = next(infos)
            if row["text"] != info["tn_transcription"] or row["transcription_source"] != info["transcription_source"]:
                raise ValueError("Input human reference does not match Info record")
            shard = chunk_index % args.shards
            write_row(handles[shard], {**row, "alignment_caption_spans": info.get("segmentation_adjusted")})
            counts[shard] += 1
        if infos is not None and next(infos, None) is not None:
            raise ValueError("Unconsumed final Info records")
    finally:
        for handle in handles:
            handle.close()
    save(
        root / "COMPLETE.json",
        {
            "input_sha256": fingerprint(args.input),
            "rows": sum(counts),
            "policy": "whole_chunks_round_robin",
            "shards": [
                {
                    "path": str(root / f"input-{i:04d}.jsonl.gz"),
                    "rows": count,
                    "sha256": fingerprint(root / f"input-{i:04d}.jsonl.gz"),
                }
                for i, count in enumerate(counts)
            ],
        },
    )


def align(args):
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from qwen_asr import Qwen3ForcedAligner

    from recipe.phimm.utils.audio import load_raw_audio

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "raw_segments.jsonl"
    settings = {
        "input_sha256": fingerprint(args.input),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "qwen_repository_revision": QWEN_REVISION,
        "runner_sha256": fingerprint(Path(__file__)),
        "maximum_window_seconds": 180.0,
        "segment_max_seconds": args.segment_seconds,
        "word_duration_minimum": args.minimum_word_seconds,
        "word_duration_maximum": args.maximum_word_seconds,
        "seed": 42,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("qwen-asr", "torch", "transformers", "huggingface-hub", "soundfile", "blobfile")
        },
    }
    run = args.output_dir / "run.json"
    if run.exists() and json.loads(run.read_text()) != settings:
        raise ValueError("Resume input or environment differs from recorded run")
    if (args.output_dir / "COMPLETE.json").exists():
        raise ValueError("Alignment output already complete")
    if not run.exists():
        if manifest.exists():
            raise ValueError("Manifest exists without provenance")
        save(run, settings)
    done = {}
    if manifest.exists():
        for row in rows(manifest):
            if row["id"] in done:
                raise ValueError("Duplicate existing alignment ID")
            done[row["id"]] = bool(row.get("alignment_error"))
    torch.manual_seed(42)
    model_path = snapshot_download(MODEL, revision=MODEL_REVISION)
    aligner = Qwen3ForcedAligner.from_pretrained(str(model_path), dtype=torch.bfloat16, device_map="cuda:0")
    count = failures = 0
    with manifest.open("a", encoding="utf-8") as output:
        for row in rows(args.input):
            if args.limit and count >= args.limit:
                break
            count += 1
            if row["id"] in done:
                failures += done.pop(row["id"])
                continue
            captions = row.pop("alignment_caption_spans", None)
            timestamps = []
            try:
                waveform, sr = load_raw_audio({"audio_chunk": row["audio_chunk"]})
                waveform = np.asarray(waveform, dtype=np.float32)
                duration = len(waveform) / sr
                windows = alignment_windows(row["text"], duration, captions)
                row.update(alignment_audio_duration=duration, alignment_windows=windows)
                reference = row["text"].split()
                for window in windows:
                    first_frame, end_frame = int(window["start"] * sr), int(window["end"] * sr)
                    offset = first_frame / sr
                    own_words = reference[window["word_start"] : window["word_end"]]
                    result = aligner.align(
                        audio=(waveform[first_frame:end_frame], sr), text=" ".join(own_words), language="English"
                    )[0]
                    if len(result) != len(own_words):
                        raise ValueError("Qwen returned a different alignment unit count")
                    for word, item in zip(own_words, result, strict=True):
                        if clean_word(word) != clean_word(item.text):
                            raise ValueError("Qwen returned a different alignment word")
                        timestamps.append(
                            {
                                "text": word,
                                "start": round(offset + item.start_time, 3),
                                "end": round(offset + item.end_time, 3),
                            }
                        )
                row["word_timestamps"] = timestamps
                row = segment_row(row, maximum=args.segment_seconds)
            except (ValueError, KeyError, TypeError, IndexError) as error:
                failures += 1
                row = {
                    **row,
                    "word_timestamps": timestamps,
                    "segments": [],
                    "alignment_error": f"{type(error).__name__}: {error}",
                }
            except Exception as error:
                # Infrastructure/OOM errors must not silently exclude a whole shard.
                # The current parent remains uncommitted and is retried on resume.
                with (args.output_dir / "run_errors.jsonl").open("a", encoding="utf-8") as errors:
                    write_row(errors, {"id": row["id"], "error_type": type(error).__name__, "error": str(error)})
                raise
            row = filter_word_durations(row, minimum=args.minimum_word_seconds, maximum=args.maximum_word_seconds)
            write_row(output, row)
            output.flush()
            if count % 100 == 0:
                print(json.dumps({"parents": count, "failures": failures}), flush=True)
    if done:
        raise ValueError("Resume output contains IDs absent from this input scope")
    save(
        args.output_dir / ("PILOT.json" if args.limit else "COMPLETE.json"),
        {**settings, "rows": count, "failures": failures, "sha256": fingerprint(manifest)},
    )
    if failures:
        raise SystemExit(2)


def pairs(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    count = parents = rejected = accepted = alignment_failures = duration_exclusions = short_parents = 0
    with (
        gzip.open(args.output_dir / "pairs.jsonl.gz", "wt", encoding="utf-8", compresslevel=1) as output,
        gzip.open(
            args.output_dir / "filtered_originals.jsonl.gz", "wt", encoding="utf-8", compresslevel=1
        ) as originals,
        gzip.open(args.output_dir / "segments.jsonl.gz", "wt", encoding="utf-8", compresslevel=1) as segmented,
        (args.output_dir / "pair_exclusions.jsonl").open("w", encoding="utf-8") as exclusions,
    ):
        for row in rows(args.input):
            row = filter_word_durations(row, minimum=args.minimum_word_seconds, maximum=args.maximum_word_seconds)
            parents += 1
            if row.get("alignment_error"):
                alignment_failures += 1
            elif not row["word_duration_filter"]["passed"]:
                duration_exclusions += 1
            if row["word_duration_filter"]["passed"]:
                accepted += 1
                write_row(
                    originals,
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"segments", "segment_metadata", "segment_max_seconds"}
                    },
                )
                write_row(
                    segmented,
                    {
                        **row,
                        "audio_path": [part[0] for part in row["segments"]],
                        "segment_reference_texts": [part[1] for part in row["segments"]],
                    },
                )
            own_count = 0
            for pair in paired_rows(row):
                write_row(output, pair)
                count += 1
                own_count += 1
            if not own_count:
                rejected += 1
                short_parents += row["word_duration_filter"]["passed"]
                write_row(
                    exclusions,
                    {
                        "id": row["id"],
                        "reason": row.get("alignment_error")
                        or (
                            "unexpected_word_duration"
                            if not row["word_duration_filter"]["passed"]
                            else "fewer_than_two_segments"
                        ),
                        "word_duration_filter": row["word_duration_filter"],
                    },
                )
    save(
        args.output_dir / "COMPLETE.json",
        {
            "input_sha256": fingerprint(args.input),
            "parents": parents,
            "filtered_originals": accepted,
            "filtered_segments_parents": accepted,
            "alignment_failures": alignment_failures,
            "word_duration_exclusions": duration_exclusions,
            "fewer_than_two_segment_parents": short_parents,
            "pairs": count,
            "excluded_parents": rejected,
            "pairing_policy": "sliding_adjacent_stride_1_same_parent",
            "pairs_sha256": fingerprint(args.output_dir / "pairs.jsonl.gz"),
            "filtered_originals_sha256": fingerprint(args.output_dir / "filtered_originals.jsonl.gz"),
            "segments_sha256": fingerprint(args.output_dir / "segments.jsonl.gz"),
            "word_duration_limits": [args.minimum_word_seconds, args.maximum_word_seconds],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--info-dir", type=Path, required=True)
    prep.add_argument("--shards", type=int, default=8)
    runner = commands.add_parser("align")
    runner.add_argument("--segment-seconds", type=float, default=30.0)
    runner.add_argument("--limit", type=int)
    pairing = commands.add_parser("pairs")
    for command in (runner, pairing):
        command.add_argument("--minimum-word-seconds", type=float, default=0.04)
        command.add_argument("--maximum-word-seconds", type=float, default=3.0)
    for command in commands.choices.values():
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare" and args.shards <= 0:
        parser.error("--shards must be positive")
    globals()[args.command](args)


if __name__ == "__main__":
    main()

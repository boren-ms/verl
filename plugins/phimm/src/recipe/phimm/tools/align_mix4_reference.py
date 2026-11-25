"""Stream Qwen forced alignment with datasource provenance on one reserved GPU.

Use --ray-address on a shared Ray node to reserve one GPU and four CPUs.
Outputs retain the original transcript and cuts; no audio is recut here.
Words with zero duration are excluded from reference_text, with their original
timestamps and transcript character spans retained as removal evidence.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remove_zero_duration_words(text, timestamps, duration):
    """Keep positive-duration words and exact source characters for their refs.

    Validate the complete alignment before dropping anything. Preserve source
    text separately; never stretch a zero-duration word or clip an invalid one.
    Character slices include each word's trailing punctuation/whitespace.
    """

    def normalized(value):
        return "".join(c.casefold() for c in value if c.isalnum())

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid actual audio duration")
    for word in timestamps:
        start, end = word["start"], word["end"]
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end <= duration + 1e-9):
            raise ValueError(f"Timestamp outside actual audio length {duration}: {word}")
    midpoints = [(w["start"] + w["end"]) / 2 for w in timestamps]
    if midpoints != sorted(midpoints):
        raise ValueError("Nonmonotonic word timestamps")
    if normalized(text) != normalized("".join(w["text"] for w in timestamps)):
        raise ValueError("Alignment words differ from source transcript")
    # Case folding can expand a character (e.g. ß); map normalized positions
    # back to the original string instead of assuming one byte/character/unit.
    positions = [i for i, c in enumerate(text) if c.isalnum() for _ in c.casefold()]
    offsets, cursor, search_start = [], 0, 0
    for word in timestamps:
        lexical_length = len(normalized(word["text"]))
        if lexical_length:
            offsets.append(positions[cursor])
            cursor += lexical_length
            search_start = positions[cursor - 1] + 1
        else:
            # Qwen can emit standalone punctuation tokens such as apostrophes.
            # Locate their literal source span between adjacent lexical words.
            limit = positions[cursor] if cursor < len(positions) else len(text)
            start = text.find(word["text"], search_start, limit)
            if not word["text"] or start < 0:
                raise ValueError("Alignment punctuation differs from source transcript")
            offsets.append(start)
            search_start = start + len(word["text"])
    if offsets:
        offsets[0] = 0
    offsets.append(len(text))
    kept, removed, refs, spans = [], [], [], []
    for i, word in enumerate(timestamps):
        lo, hi = offsets[i : i + 2]
        if word["start"] == word["end"]:
            removed.append(
                {
                    **word,
                    "word_index": i,
                    "character_span": [lo, hi],
                    "reference_text": text[lo:hi],
                    "reason": "zero_duration",
                }
            )
        else:
            kept.append(dict(word))
            refs.append(text[lo:hi])
            spans.append([lo, hi])
    reference = "".join(refs) if timestamps else text
    return {
        "word_timestamps": kept,
        "unaltered_word_timestamps": [dict(w) for w in timestamps],
        "reference_text": reference,
        "reference_word_character_spans": spans,
        "removed_zero_duration_words": removed,
        "zero_duration_word_count": len(removed),
        "zero_duration_policy": "remove",
        "alignment_status": (
            "QWEN_ALIGNED" if kept else "NO_POSITIVE_DURATION_REFERENCE" if timestamps else "NO_LEXICAL_REFERENCE"
        ),
    }


def validate_parent_audio(row):
    """Reject segment audio/full-transcript alignment before loading audio."""
    path = row.get("audio_path")
    if not isinstance(path, str) or not path or "#" in path or ";" in path:
        raise ValueError("Alignment requires one full parent audio path, without segment selectors")
    if not isinstance(row.get("text"), str):
        raise ValueError("Alignment requires the full parent transcript")
    return path


def assign_parent_alignment(row, timestamps, duration):
    """Assign one full-parent alignment to unchanged original audio segments.

    Use absolute parent word midpoints, never align a segment with the full
    transcript. Crossing words go to one segment and are explicitly recorded.
    """
    parent = validate_parent_audio(row)
    paths = row.get("original_segment_audio_paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError("Original segment paths are required for assignment")
    ranges = []
    for path in paths:
        physical, selector = path.rsplit("#", 1)
        if physical != parent:
            raise ValueError("Segment does not belong to the aligned parent audio")
        endpoints = selector.split(":")
        if len(endpoints) != 2 or any(not value for value in endpoints):
            raise ValueError("Explicit start and end segment bounds are required")
        percent = endpoints[0].endswith("%")
        if percent != endpoints[1].endswith("%"):
            raise ValueError("Segment bound units differ")
        lo, hi = [float(value.rstrip("%")) for value in endpoints]
        if percent:
            lo, hi = lo * duration / 100, hi * duration / 100
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi <= duration + 1e-9):
            raise ValueError("Segment outside actual parent audio duration")
        ranges.append((lo, hi))
    if (
        ranges[0][0] != 0
        or abs(ranges[-1][1] - duration) > 1e-9
        or any(abs(left[1] - right[0]) > 1e-9 for left, right in zip(ranges, ranges[1:], strict=False))
    ):
        raise ValueError("Segments must cover the full parent audio once, in order")
    cleaned = remove_zero_duration_words(row["text"], timestamps, duration)
    refs = [""] * len(paths)
    assigned, crossings = [], []
    boundaries = [hi for _, hi in ranges[:-1]]
    for word, (lo, hi) in zip(cleaned["word_timestamps"], cleaned["reference_word_character_spans"], strict=True):
        midpoint = (word["start"] + word["end"]) / 2
        segment = sum(midpoint >= boundary for boundary in boundaries)
        refs[segment] += row["text"][lo:hi]
        assigned.append(segment)
        crossings.extend(
            {**word, "boundary_s": boundary, "assigned_segment": segment + 1}
            for boundary in boundaries
            if word["start"] < boundary < word["end"]
        )
    if not timestamps:
        # Punctuation-only rows have no lexical timing; retain punctuation with
        # an explicit status rather than fabricate a word alignment.
        refs[-1] = cleaned["reference_text"]
    assert "".join(refs) == cleaned["reference_text"]
    return {
        **cleaned,
        "segment_reference_texts": refs,
        "word_segment_indices": assigned,
        "reference_boundary_crossings": crossings,
        "alignment_scope": "full_parent_audio_then_original_segments",
        "reference_boundary_policy": "word_midpoint_original_cuts_zero_duration_removed",
    }


def duration_bounded_align(aligner, audio, text, language):
    """Decode Qwen timestamp logits over bins inside each actual audio.

    Keep raw predictions as evidence. This restricts model decoding before
    argmax; it never clips predicted timestamps or pads the source waveform.
    Qwen's own monotonic timestamp parser remains unchanged.
    """
    import torch
    from qwen_asr.inference.qwen3_forced_aligner import ensure_list, normalize_audios

    audios = normalize_audios(audio)
    texts, languages = ensure_list(text), ensure_list(language)
    if len(languages) == 1:
        languages = languages * len(audios)
    assert len(audios) == len(texts) == len(languages)
    encoded = [aligner.aligner_processor.encode_timestamp(t, lang) for t, lang in zip(texts, languages, strict=True)]
    inputs = aligner.processor(text=[e[1] for e in encoded], audio=audios, return_tensors="pt", padding=True)
    inputs = inputs.to(aligner.model.device).to(aligner.model.dtype)
    with torch.inference_mode():
        logits = aligner.model.thinker(**inputs).logits
    results, diagnostics = [], []
    for input_id, scores, (word_list, _), waveform in zip(inputs["input_ids"], logits, encoded, audios, strict=True):
        duration = len(waveform) / 16000
        max_bin = math.floor((duration * 1000 + 1e-9) / aligner.timestamp_segment_time)
        scores = scores[input_id == aligner.timestamp_token_id]
        raw = scores.argmax(dim=-1)
        bounded = scores[:, : min(max_bin + 1, scores.shape[-1])].argmax(dim=-1)
        timestamp_ms = (bounded * aligner.timestamp_segment_time).to("cpu").numpy()
        parsed = aligner.aligner_processor.parse_timestamp(word_list, timestamp_ms)
        for item in parsed:
            item["start_time"] = round(item["start_time"] / 1000, 3)
            item["end_time"] = round(item["end_time"] / 1000, 3)
        diagnostics.append(
            {
                "policy": "duration_bounded_logit_argmax",
                "duration_s": duration,
                "timestamp_bin_ms": aligner.timestamp_segment_time,
                "max_valid_bin": max_bin,
                "raw_timestamp_bins": raw.tolist(),
                "bounded_timestamp_bins": bounded.tolist(),
                "changed_timestamp_count": int((raw != bounded).sum()),
                "timestamp_count": len(raw),
            }
        )
        results.append(aligner._to_structured_items(parsed))
    aligner.last_bounds_diagnostics = diagnostics
    return results


def align(args):
    import ray

    if ray.is_initialized():
        allocated = ray.get_gpu_ids()
        if len(allocated) != 1:
            raise ValueError(f"Expected one Ray-reserved GPU, got {allocated}")
        # TorchFlow nodes can disable Ray's automatic CUDA visibility masking.
        # Apply our reservation before importing Torch or initializing CUDA.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(allocated[0]))
        print(json.dumps({"worker_pid": os.getpid(), "reserved_gpu": allocated[0]}), flush=True)
    import torch
    from qwen_asr import Qwen3ForcedAligner
    from recipe.phimm.utils.audio import load_raw_audio

    inputs = sorted(Path(args.input_dir).glob(args.source or "*.jsonl"))
    if not inputs:
        raise ValueError("No datasource JSONL inputs")
    model = Path(args.model)
    if not (model / "config.json").exists() or not list(model.glob("*.safetensors")):
        raise ValueError("Expected complete local Qwen3-ForcedAligner model")
    proof = {
        "inputs": {p.name: sha256(p) for p in inputs},
        "model_files": {str(p.relative_to(model)): sha256(p) for p in sorted(model.rglob("*")) if p.is_file()},
        "packages": {p: importlib.metadata.version(p) for p in ("qwen-asr", "transformers", "torch")},
        "language": "English",
        "batch_size": args.batch_size,
        "limit_per_source": args.limit,
        "duration_bounded_decoding": args.duration_bounded,
        "zero_duration_policy": "remove",
        "alignment_scope": "full_parent_audio_then_original_segments",
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    proof_path = output / "run.json"
    if proof_path.exists():
        if json.loads(proof_path.read_text()) != proof:
            raise ValueError("Resume inputs/model/settings differ")
    else:
        proof_path.write_text(json.dumps(proof, indent=2) + "\n")
    aligner = Qwen3ForcedAligner.from_pretrained(str(model), dtype=torch.bfloat16, device_map="cuda:0")
    if args.duration_bounded:
        from functools import partial

        aligner.align = partial(duration_bounded_align, aligner)
    totals = {}
    for source in inputs:
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if args.limit:
            rows = rows[: args.limit]
        expected = {r["id"]: r for r in rows}
        for row in rows:
            validate_parent_audio(row)
        if len(expected) != len(rows):
            raise ValueError(f"Duplicate IDs in {source}")
        target = output / source.name
        completed = {}
        if target.exists():
            for line in target.read_text().splitlines():
                row = json.loads(line)
                if row["id"] in completed or row["id"] not in expected:
                    raise ValueError("Duplicate/unexpected resumed ID")
                for key, value in expected[row["id"]].items():
                    if row.get(key) != value:
                        raise ValueError("Resumed row differs from input")
                completed[row["id"]] = row
            if args.retry_errors:
                # Terminal error evidence is preserved before rewriting the
                # journal. Only exact failed rows are retried; successful rows
                # remain verified against the original input.
                failed = [r for r in completed.values() if r.get("alignment_error")]
                if failed:
                    with (output / "error_lineage.jsonl").open("a") as lineage:
                        for row in failed:
                            lineage.write(json.dumps(row, ensure_ascii=False) + "\n")
                            if not any(c.isalnum() for c in row["text"]):
                                row.update(alignment_error="", alignment_status="NO_LEXICAL_REFERENCE")
                            else:
                                del completed[row["id"]]
                    temporary = output / "journal.tmp"
                    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in completed.values()))
                    temporary.replace(target)
        pending = [r for r in rows if r["id"] not in completed]
        with target.open("a") as stream:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                loaded = []
                for row in batch:
                    result = dict(row)
                    try:
                        audio, sr = load_raw_audio(row)
                        result["alignment_audio_duration_s"] = len(audio) / sr
                        loaded.append((result, (audio, sr)))
                    except Exception as error:
                        result.update(word_timestamps=[], alignment_error=f"{type(error).__name__}: {error}")
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                        completed[row["id"]] = result
                try:
                    aligned = (
                        aligner.align(
                            audio=[a for _, a in loaded],
                            text=[r["text"] for r, _ in loaded],
                            language=["English"] * len(loaded),
                        )
                        if loaded
                        else []
                    )
                    if len(aligned) != len(loaded):
                        raise ValueError("Alignment batch result count differs")
                    outcomes = [(words, "") for words in aligned]
                except Exception:
                    outcomes = []
                    for result, audio in loaded:
                        try:
                            single = aligner.align(audio=audio, text=result["text"], language="English")
                            if len(single) != 1:
                                raise ValueError("Expected one alignment result")
                            outcomes.append((single[0], ""))
                        except Exception as error:
                            outcomes.append(([], f"{type(error).__name__}: {error}"))
                for (result, _), (words, error) in zip(loaded, outcomes, strict=True):
                    try:
                        if error:
                            raise ValueError(error)
                        timestamps = [
                            {"text": w.text, "start": float(w.start_time), "end": float(w.end_time)} for w in words
                        ]
                        if not timestamps and any(c.isalnum() for c in result["text"]):
                            raise ValueError("Empty alignment for nonempty transcript")
                        duration = result["alignment_audio_duration_s"]
                        for word in timestamps:
                            if not (
                                math.isfinite(word["start"])
                                and math.isfinite(word["end"])
                                and 0 <= word["start"] <= word["end"] <= duration + 1e-9
                            ):
                                raise ValueError(f"Timestamp outside actual audio length {duration}: {word}")
                        midpoints = [(w["start"] + w["end"]) / 2 for w in timestamps]
                        if midpoints != sorted(midpoints):
                            raise ValueError("Nonmonotonic word timestamps")
                        result.update(assign_parent_alignment(result, timestamps, duration), alignment_error="")
                        if args.duration_bounded:
                            result["alignment_decode_diagnostics"] = aligner.last_bounds_diagnostics[0]
                    except Exception as error:
                        result.update(word_timestamps=[], alignment_error=f"{type(error).__name__}: {error}")
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    completed[result["id"]] = result
                stream.flush()
                os.fsync(stream.fileno())
                progress = {
                    "source": source.stem,
                    "completed": len(completed),
                    "total": len(rows),
                    "errors": sum(bool(r["alignment_error"]) for r in completed.values()),
                }
                temporary = output / "progress.tmp"
                temporary.write_text(json.dumps(progress) + "\n")
                temporary.replace(output / "progress.json")
                print(json.dumps(progress), flush=True)
        totals[source.stem] = {
            "rows": len(completed),
            "errors": sum(bool(r["alignment_error"]) for r in completed.values()),
            "sha256": sha256(target),
        }
    (output / "completion.json").write_text(json.dumps(totals, indent=2) + "\n")
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Smoke rows per datasource; use a separate output")
    parser.add_argument("--ray-address", help="Reserve 1 GPU / 4 CPUs through shared Ray")
    parser.add_argument("--source", help="Optional exact datasource filename")
    parser.add_argument(
        "--retry-errors", action="store_true", help="Retain error lineage and retry exact failed lexical rows"
    )
    parser.add_argument(
        "--duration-bounded",
        action="store_true",
        help="Restrict timestamp-logit decoding to actual audio duration; batch size must be 1",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        parser.error("Invalid batch size or limit")
    if args.duration_bounded and args.batch_size != 1:
        parser.error("Duration-bounded recovery requires batch size 1")
    if args.ray_address:
        import ray

        ray.init(address=args.ray_address, runtime_env={"py_executable": sys.executable})
        tasks = []
        for source in sorted(Path(args.input_dir).glob(args.source or "*.jsonl")):
            worker = argparse.Namespace(**vars(args))
            worker.source = source.name
            worker.output_dir = str(Path(args.output_dir) / source.stem)
            tasks.append(ray.remote(num_gpus=1, num_cpus=4)(align).remote(worker))
        # Collect all datasource results before rejecting failures. One source's
        # flagged row must not terminate the other source workers prematurely.
        results = ray.get(tasks)
        print(json.dumps(results), flush=True)
        if any(v["errors"] for result in results for v in result.values()):
            raise RuntimeError("Alignment failures retained in output; source is not ready")
    else:
        results = align(args)
        if any(v["errors"] for v in results.values()):
            raise RuntimeError("Alignment failures retained in output; source is not ready")


if __name__ == "__main__":
    main()

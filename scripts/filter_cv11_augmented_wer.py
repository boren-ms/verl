#!/usr/bin/env python3
"""Join complete CV11 ASR generation to its source and publish WER >= 1% rows.

Every source row must have exactly one matching generation row in manifest order.
Publication waits for complete coverage and the pinned source manifest checksum.
"""

import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
import sys

import blobfile as bf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.filter_asr_generation_wer import Stats, measurements


SOURCE_SHA256 = "9243d564a8b1f97c52a22858fa95d6709a32b5cbb18c0c7d679ab2bb0a16767f"
MODEL_PATH = "az://orngwus2cresco/data/speech/projects/phi-fastllm-2609/amlt-results/fast-llm-2609-qwen9b-flashenc-mtp-s2-data-v3.5.4-r2/54000/qwen_hf"
DATA_SOURCE = "noisy_cv11_en_train_2609r2_wer_ge1pct"


def join_row(source, generated):
    """Validate identity and measurements, retaining original training labels."""
    _, _, pointer, values = measurements(generated)
    if (source["id"] != generated.get("id") or source["text"] != generated["text"]
            or source["audio_path"] != pointer):
        raise ValueError(f"Generation/source identity mismatch: {source['id']}")
    if abs(source["duration"] - generated["audio_duration_seconds"]) > 1 / source["sample_rate"]:
        raise ValueError(f"Audio was trimmed or changed: {source['id']}")
    fields = {key: generated[key] for key in (
        "wer", "n_err", "n_ref", "response", "raw_response", "response_tokens", "response_at_limit")
        if key in generated}
    fields.update(filter_model_path=MODEL_PATH, filter_wer_threshold_fraction=0.01)
    return {**source, **fields, "data_source": DATA_SOURCE}, values


def keep_row(row):
    return 100 * row["n_err"] >= row["n_ref"]


def blob_lines(path, read_size=4 * 1024 * 1024):
    """Read large Blob ranges while preserving exact JSONL bytes for hashing.

    Default line buffering otherwise issues many small range requests when
    blobfile's use_streaming_read setting is disabled.
    """
    pending = b""
    with bf.BlobFile(path, "rb", streaming=True) as stream:
        for block in iter(lambda: stream.read(read_size), b""):
            lines = (pending + block).split(b"\n")
            pending = lines.pop()
            for line in lines:
                yield line + b"\n"
    if pending:
        yield pending


def generation_rows(paths):
    for path in paths:
        for line in blob_lines(path):
            yield json.loads(line)


def run(args):
    destination = args.output_dir.rstrip("/")
    if bf.exists(destination + "/COMPLETE.json"):
        raise FileExistsError("Filtered dataset is already complete")
    paths = sorted(bf.glob(args.generation_dir.rstrip("/") + "/part-*.jsonl"),
                   key=lambda p: int(re.search(r"part-(\d+)\.jsonl$", p).group(1)))
    indices = [int(re.search(r"part-(\d+)\.jsonl$", p).group(1)) for p in paths]
    if not paths or indices != list(range(len(paths))):
        raise ValueError("Generation parts must form a complete contiguous prefix")
    root = Path(args.work_dir)
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / "data.jsonl"
    original, filtered = Stats(), Stats()
    seen = set()
    source_digest = hashlib.sha256()
    source_words = filtered_words = 0
    with output_path.open("wb") as output:
        for source_line, generated in zip_longest(blob_lines(args.source_jsonl), generation_rows(paths)):
            if source_line is None or generated is None:
                raise ValueError("Generation does not cover every source row exactly once")
            source_digest.update(source_line)
            source = json.loads(source_line)
            if source["id"] in seen:
                raise ValueError("Duplicate source ID")
            seen.add(source["id"])
            joined, values = join_row(source, generated)
            original.add(generated, values)
            source_words += len(source["text"].split())
            if keep_row(generated):
                filtered.add(generated, values)
                filtered_words += len(source["text"].split())
                output.write((json.dumps(joined, ensure_ascii=False, allow_nan=False) + "\n").encode())
            if len(seen) % 100000 == 0:
                print(f"Verified {len(seen)}/{args.expected_rows} source rows", flush=True)
    if len(seen) != args.expected_rows or source_digest.hexdigest() != SOURCE_SHA256:
        raise ValueError("Source coverage or pinned manifest checksum failed; no dataset published")
    original_stats = original.summary(args.expected_rows)
    filtered_stats = filtered.summary(args.expected_rows)
    if not filtered_stats["count"]:
        raise ValueError("No samples meet the requested threshold")
    with output_path.open("rb") as stream:
        output_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    summary = {
        "source_manifest": args.source_jsonl, "source_manifest_sha256": SOURCE_SHA256,
        "generation_parts": paths, "model_path": MODEL_PATH,
        "scoring": "openasr_en_eval, version 2609; compound-aware HF English normalized WER, uncapped",
        "filter": "WER >= 1%; 100*n_err >= n_ref (inclusive)",
        "word_count_definition": "Whitespace-separated tokens in original preserved transcripts",
        "original": {**original_stats, "total_words": source_words},
        "filtered": {**filtered_stats, "total_words": filtered_words},
        "filtered_manifest_sha256": output_digest,
        "source_rows_and_transcripts_verified": True,
        "augmentation_metadata_preserved": True,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    comments = (
        f"# Filtered clips: {filtered_stats['count']:,}; retention: {filtered_stats['retention_rate']:.4%}.\n"
        f"# Audio hours: {filtered_stats['audio_hours']:.2f}; mean duration: {filtered_stats['audio_seconds']['mean']:.2f} s.\n"
        f"# Original transcript words: {filtered_words:,}; mean words/clip: {filtered_words / filtered_stats['count']:.2f}.\n"
        "# WER >= 1%, inclusive; original 2609r2 step 54000; English normalized compound-aware WER.\n"
    )
    training_text = Path(args.training_yaml).read_text()
    training_config = yaml.safe_load(training_text)
    if training_config["jsonl_paths"] != destination + "/data.jsonl":
        raise ValueError("Training YAML destination does not match filtered output")
    (root / "training.yaml").write_text(comments + training_text)
    (root / "remax_training.yaml").write_text(Path(args.remax_yaml).read_text())
    for name in ["summary.json", "training.yaml", "remax_training.yaml", "data.jsonl"]:
        bf.copy(str(root / name), destination + "/" + name, overwrite=True)
    digest = hashlib.sha256()
    with bf.BlobFile(destination + "/data.jsonl", "rb", streaming=True) as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != output_digest:
        raise ValueError("Uploaded filtered manifest checksum mismatch")
    complete = {"source_rows": args.expected_rows, "filtered_rows": filtered_stats["count"],
                "filtered_manifest_sha256": output_digest, "remote_manifest_sha256_verified": True}
    with bf.BlobFile(destination + "/COMPLETE.json", "w") as stream:
        json.dump(complete, stream, indent=2)
    (root / "COMPLETE.json").write_text(json.dumps(complete, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--expected-rows", type=int, default=948736)
    parser.add_argument("--training-yaml", required=True)
    parser.add_argument("--remax-yaml", required=True)
    run(parser.parse_args())

#!/usr/bin/env python3
"""Publish an exact top-K CV11 subset ordered by baseline word-error count.

The full candidate manifest must match its pinned checksum and completion evidence.
Rank by n_err descending, then original source row index ascending; never by WER.
"""
import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import sys

import blobfile as bf
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.filter_asr_generation_wer import Stats, measurements
from scripts.filter_cv11_augmented_wer import MODEL_PATH, blob_lines, keep_row


def select_topk(lines, count, expected_rows, source_sha256, source_mode):
    if count <= 0 or count > expected_rows:
        raise ValueError("Top-K count must be positive and no larger than the candidate count")
    heap, seen = [], set()
    digest = hashlib.sha256()
    stats = Stats()
    errors = Counter()
    words = 0
    for index, line in enumerate(lines):
        digest.update(line)
        row = json.loads(line)
        measured = {**row, "audio_duration_seconds": row["duration"]}
        _, _, _, values = measurements(measured)
        if not keep_row(row, source_mode) or row["filter_model_path"] != MODEL_PATH:
            raise ValueError("Candidate does not match the parent filter/model")
        if row["id"] in seen:
            raise ValueError("Duplicate source ID")
        seen.add(row["id"])
        stats.add(measured, values)
        words += len(row["text"].split())
        n_err = int(row["n_err"])
        errors[n_err] += 1
        item = (n_err, -index, line)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)
        if (index + 1) % 100000 == 0:
            print(f"Validated {index + 1}/{expected_rows} candidate rows", flush=True)
    if len(seen) != expected_rows or digest.hexdigest() != source_sha256:
        raise ValueError("Candidate count or pinned checksum mismatch; no dataset published")
    selected = sorted(heap, key=lambda item: (-item[0], -item[1]))
    summary = stats.summary(expected_rows)
    summary["total_words"] = words
    return selected, summary, errors


def run(args):
    destination = args.output_dir.rstrip("/")
    if bf.exists(destination + "/COMPLETE.json"):
        raise FileExistsError("Top-K output is already complete")
    parent = args.source_jsonl.rsplit("/", 1)[0]
    with bf.BlobFile(parent + "/COMPLETE.json", "r") as stream:
        complete = json.load(stream)
    with bf.BlobFile(parent + "/summary.json", "r") as stream:
        parent_summary = json.load(stream)
    if (complete["filtered_manifest_sha256"] != args.source_sha256
            or not complete["remote_manifest_sha256_verified"]
            or complete["filtered_rows"] != args.expected_rows
            or parent_summary["filtered_manifest_sha256"] != args.source_sha256
            or parent_summary["filter_mode"] != args.source_mode):
        raise ValueError("Parent completion/summary does not match the pinned candidates")
    root = Path(args.work_dir)
    root.mkdir(parents=True, exist_ok=True)
    selected, original_stats, all_error_counts = select_topk(
        blob_lines(args.source_jsonl), args.count, args.expected_rows,
        args.source_sha256, args.source_mode)
    training_text = Path(args.training_yaml).read_text()
    training_config = yaml.safe_load(training_text)
    target_source = training_config["post_process"]["add_field"]["fields"]["data_source"]
    if training_config["jsonl_paths"] != destination + "/data.jsonl":
        raise ValueError("Training YAML references the wrong output")
    filtered, filtered_words, selected_errors = Stats(), 0, []
    output_path = root / "data.jsonl"
    output_digest = hashlib.sha256()
    with output_path.open("wb") as stream:
        for rank, (errors, negative_index, line) in enumerate(selected, 1):
            row = json.loads(line)
            measured = {**row, "audio_duration_seconds": row["duration"]}
            _, _, _, values = measurements(measured)
            filtered.add(measured, values)
            filtered_words += len(row["text"].split())
            selected_errors.append(errors)
            row.update(data_source=target_source, selection_nerr_rank=rank,
                       selection_source_row_index=-negative_index)
            encoded = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode()
            stream.write(encoded)
            output_digest.update(encoded)
    stats = filtered.summary(args.expected_rows)
    stats["total_words"] = filtered_words
    error_array = np.asarray(selected_errors)
    stats["error_count"] = {"mean": float(error_array.mean()), "min": int(error_array.min()),
        "max": int(error_array.max()), "median": float(np.median(error_array))}
    cutoff = selected_errors[-1]
    selection = {"mode": "top_n_err", "count": args.count,
        "sort": "n_err descending, original source row index ascending",
        "cutoff_n_err": cutoff, "candidate_ties_at_cutoff": all_error_counts[cutoff],
        "selected_ties_at_cutoff": selected_errors.count(cutoff),
        "global_topk_verified": True}
    summary = {"source_manifest": args.source_jsonl,
        "source_manifest_sha256": args.source_sha256, "model_path": MODEL_PATH,
        "filter_mode": args.source_mode, "filter": parent_summary["filter"],
        "scoring": parent_summary["scoring"], "data_source": target_source,
        "selection": selection, "original": original_stats, "filtered": stats,
        "filtered_manifest_sha256": output_digest.hexdigest(),
        "word_count_definition": "Whitespace-separated tokens in preserved original transcripts",
        "original_labels_and_augmentation_metadata_preserved": True}
    comments = (
        f"# Top baseline n_err: {args.count:,} clips from {args.expected_rows:,} candidates.\n"
        f"# Sorted by n_err descending, ties by original source row order; cutoff n_err: {cutoff}.\n"
        f"# Audio hours: {stats['audio_hours']:.6f}; mean duration: {stats['audio_seconds']['mean']:.6f} s.\n"
        f"# Original transcript words: {filtered_words:,}; mean words/clip: {filtered_words / args.count:.6f}.\n"
        f"# Mean n_err: {stats['error_count']['mean']:.6f}; corpus baseline WER: {stats['corpus_wer_percent']:.6f}%.\n")
    (root / "training.yaml").write_text(comments + training_text)
    (root / "remax_training.yaml").write_text(Path(args.remax_yaml).read_text())
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name in ["data.jsonl", "training.yaml", "remax_training.yaml", "summary.json"]:
        bf.copy(str(root / name), destination + "/" + name, overwrite=True)
    remote_digest = hashlib.sha256()
    for line in blob_lines(destination + "/data.jsonl"):
        remote_digest.update(line)
    if remote_digest.hexdigest() != output_digest.hexdigest():
        raise ValueError("Full uploaded top-K manifest checksum mismatch")
    result = {"source_rows": args.expected_rows, "filtered_rows": args.count,
        "filtered_manifest_sha256": output_digest.hexdigest(),
        "remote_manifest_sha256_verified": True, "selection": selection}
    (root / "COMPLETE.json").write_text(json.dumps(result, indent=2) + "\n")
    bf.copy(str(root / "COMPLETE.json"), destination + "/COMPLETE.json", overwrite=True)
    print(json.dumps({"filtered": stats, "selection": selection,
        "filtered_manifest_sha256": output_digest.hexdigest()}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--expected-rows", required=True, type=int)
    parser.add_argument("--source-mode", required=True, choices=["wer_ge1pct", "wer_gt5pct_lt100pct"])
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--training-yaml", required=True)
    parser.add_argument("--remax-yaml", required=True)
    run(parser.parse_args())

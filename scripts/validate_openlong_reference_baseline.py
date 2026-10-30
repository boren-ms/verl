#!/usr/bin/env python3
"""Validate hybrid OpenLong candidates against original test and Qwen sources."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.build_openlong_high_error_segments import DATASETS, EXPECTED
from scripts.reconcile_openlong_reference_baseline import load_original_references


def validate(qwen_path, original_path, candidate_path, dataset):
    qwen = {}
    with qwen_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["id"] in qwen:
                    raise ValueError(f"Duplicate Qwen source ID: {row['id']}")
                qwen[row["id"]] = row
    references = load_original_references(original_path, dataset)
    if len(qwen) != EXPECTED[dataset][0]:
        raise ValueError(f"Unexpected Qwen source coverage: {dataset}")
    groups = defaultdict(list)
    seen = set()
    counts = Counter()
    with candidate_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ident = row["id"]
            if ident in seen or ident not in qwen:
                raise ValueError(f"Duplicate or unexpected candidate ID: {ident}")
            source = qwen[ident]
            for name in ("audio_path", "baseline_hypothesis", "parent_audio_path", "seg_index", "seg_start", "seg_end", "n_segments", "prior_inferred_text"):
                if row[name] != source[name]:
                    raise ValueError(f"Changed {name}: {ident}")
            if row["previous_qwen_text"] != source["text"] or row["qwen_alignment_flags"] != source["alignment_flags"]:
                raise ValueError(f"Qwen source provenance mismatch: {ident}")
            if row["alignment_status"] != ("needs_review" if row["alignment_flags"] else "hybrid_candidate"):
                raise ValueError(f"Inconsistent review flags: {ident}")
            if row["baseline_n_err"] != row["baseline_n_ins"] + row["baseline_n_del"] + row["baseline_n_sub"]:
                raise ValueError(f"Inconsistent S+D+I: {ident}")
            seen.add(ident)
            groups[row["parent_audio_path"]].append(row)
            counts["segments"] += 1
            counts["needs_review"] += int(row["alignment_status"] == "needs_review")
            counts["empty_text"] += int(not row["text"].strip())
    if seen != set(qwen) or set(groups) != set(references):
        raise ValueError(f"Missing candidates or parents: {dataset}")
    for parent, rows in groups.items():
        rows.sort(key=lambda row: row["seg_index"])
        if [row["seg_index"] for row in rows] != list(range(rows[0]["n_segments"])):
            raise ValueError(f"Incomplete parent indices: {parent}")
        original_words = references[parent].split()
        if [word for row in rows for word in row["text"].split()] != original_words:
            raise ValueError(f"Original test text lost, duplicated, or reordered: {parent}")
        offset = 0
        for row in rows:
            words = row["text"].split()
            if words and (row["reference_word_start"], row["reference_word_end"]) != (offset, offset + len(words)):
                raise ValueError(f"Invalid reference word range: {row['id']}")
            if not words and (row["reference_word_start"] is not None or row["reference_word_end"] is not None):
                raise ValueError(f"Unexpected empty reference range: {row['id']}")
            offset += len(words)
    return {**counts, "parents": len(groups)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-dir", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()
    for dataset in DATASETS:
        stats = validate(args.qwen_dir / f"{dataset}.jsonl", args.original_dir / f"{dataset}.jsonl",
                         args.candidate_dir / f"{dataset}.jsonl", dataset)
        print(dataset, json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()

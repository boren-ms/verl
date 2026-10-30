#!/usr/bin/env python3
"""Select the 100 highest word-error-count Qwen-aligned clips per corpus.

Errors include substitutions, deletions, and insertions against the inferred
segment reference. Empty references remain eligible: their insertions count,
although WER is undefined. Qwen candidates and their flags are not verified
ground truth.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from recipe.phimm.reward.asr_eval import measure_openasr_en_wer
from scripts.build_openlong_high_error_segments import DATASETS, EXPECTED, sha256


def build_dataset(source: Path, destination: Path, dataset: str, expected_source_sha: str, limit: int = 100) -> dict:
    source_sha = sha256(source)
    if source_sha != expected_source_sha:
        raise ValueError(f"Source checksum mismatch: {source}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    rows = []
    seen = set()
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in seen:
                raise ValueError(f"Duplicate ID: {row['id']}")
            seen.add(row["id"])
            metric = measure_openasr_en_wer(row["baseline_hypothesis"], row["text"])
            if (metric["n_err"] != row["baseline_n_err"] or metric["n_ref"] != row["baseline_n_ref"] or
                not math.isclose(metric["wer"], row["baseline_wer"], rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError(f"Saved error metric mismatch: {row['id']}")
            if metric["n_err"] != metric["n_ins"] + metric["n_del"] + metric["n_sub"]:
                raise ValueError(f"Inconsistent error components: {row['id']}")
            row.update({"baseline_n_ins": metric["n_ins"], "baseline_n_del": metric["n_del"],
                        "baseline_n_sub": metric["n_sub"],
                        "reference_word_count": len(metric["normalized_ref"].split())})
            rows.append(row)
    if len(rows) != EXPECTED[dataset][0] or len(rows) < limit:
        raise ValueError(f"Unexpected source count: {dataset} has {len(rows)} rows")
    # Equal total error counts are ordered by insertion count, then stable ID.
    rows.sort(key=lambda row: (-row["baseline_n_err"], -row["baseline_n_ins"], row["id"]))
    selected = rows[:limit]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for index, row in enumerate(selected, 1):
            row["error_count_rank"] = index
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    flags = Counter(flag for row in selected for flag in row["alignment_flags"])
    return {"source_sha256": source_sha, "output_sha256": sha256(destination),
            "source_segments": len(rows), "selected_segments": len(selected),
            "min_error_count": selected[-1]["baseline_n_err"], "max_error_count": selected[0]["baseline_n_err"],
            "selected_insertions": sum(row["baseline_n_ins"] for row in selected),
            "selected_deletions": sum(row["baseline_n_del"] for row in selected),
            "selected_substitutions": sum(row["baseline_n_sub"] for row in selected),
            "empty_normalized_references": sum(row["reference_word_count"] == 0 for row in selected),
            "needs_review": sum(row["alignment_status"] == "needs_review" for row in selected),
            "alignment_flags": dict(flags)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    source_summary = json.loads(args.source_summary.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"_provenance": {
        "source": "az://orngwus2cresco/data/boren/data/Evaluation/OpenLong_qwen_aligned_2609r2b_step400_v1/full/",
        "ranking": "Descending total word errors (substitutions + deletions + insertions), then insertion count, then ID; WER is not used for ranking.",
        "scoring": "OpenASR English normalizer and compound-aware kaldialign on baseline_hypothesis against Qwen-aligned text.",
        "empty_reference_note": "Empty normalized references remain eligible and their insertions count; reference_word_count=0 denotes undefined WER.",
        "quality_note": "Qwen-aligned reference text is a candidate, not human-verified; selection preserves all alignment flags.",
    }}
    for dataset in DATASETS:
        print(f"Scoring {dataset}...", flush=True)
        summary[dataset] = build_dataset(args.input_dir / f"{dataset}.jsonl", args.output_dir / f"{dataset}.jsonl",
                                         dataset, source_summary[dataset]["output_sha256"])
        print(dataset, json.dumps(summary[dataset]), flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

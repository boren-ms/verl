#!/usr/bin/env python3
"""Build top-error subsets from test-reference/baseline/Qwen consensus data."""

import argparse
import json
from pathlib import Path

from scripts.build_openlong_high_error_segments import DATASETS
from scripts.build_openlong_qwen_error_top100 import build_dataset


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
        "source": "az://orngwus2cresco/data/boren/data/Evaluation/OpenLong_testref_baseline_qwen_consensus_2609r2b_v1/full/",
        "ranking": "Descending total word errors (substitutions + deletions + insertions), then insertion count, then ID; WER is not used.",
        "scoring": "OpenASR English normalizer and compound-aware kaldialign on baseline_hypothesis against consensus text.",
        "empty_reference_note": "Empty normalized references remain eligible and their insertions count; reference_word_count=0 denotes undefined WER.",
        "quality_note": "Reference text is automatically reconciled, not human-verified; selection preserves all alignment flags.",
    }}
    for dataset in DATASETS:
        print(f"Scoring {dataset}...", flush=True)
        summary[dataset] = build_dataset(args.input_dir / f"{dataset}.jsonl", args.output_dir / f"{dataset}.jsonl",
                                         dataset, source_summary[dataset]["output_sha256"])
        print(dataset, json.dumps(summary[dataset]), flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

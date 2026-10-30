#!/usr/bin/env python3
"""Summarize reference-length diagnostics before and after reconciliation."""

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.build_openlong_high_error_segments import DATASETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"_note": "Word-length thresholds are diagnostic only and do not establish acoustic alignment correctness."}
    for dataset in DATASETS:
        counts = Counter()
        flags = Counter()
        with (args.qwen_dir / f"{dataset}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    n = len(row["text"].split())
                    counts["qwen_reference_words_over_150"] += int(n > 150)
                    counts["qwen_reference_words_over_200"] += int(n > 200)
        with (args.candidate_dir / f"{dataset}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    n = len(row["text"].split())
                    counts["segments"] += 1
                    counts["consensus_reference_words_over_150"] += int(n > 150)
                    counts["consensus_reference_words_over_200"] += int(n > 200)
                    counts["needs_review"] += int(row["alignment_status"] == "needs_review")
                    counts["empty_text"] += int(n == 0)
                    counts["changed_from_qwen"] += int(row["text"] != row["previous_qwen_text"])
                    flags.update(row["alignment_flags"])
        result[dataset] = {**counts, "flags": dict(flags)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize review flags and text changes in Qwen-aligned OpenLong JSONLs."""

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.build_openlong_high_error_segments import DATASETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"_warning": "Qwen timestamps are machine estimates; flags are conservative and do not establish correctness."}
    for dataset in DATASETS:
        counts = Counter()
        flags = Counter()
        with (args.input_dir / f"{dataset}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                counts["segments"] += 1
                counts["needs_review"] += int(row["alignment_status"] == "needs_review")
                counts["text_changed"] += int(row["text"] != row["prior_inferred_text"])
                counts["empty_text"] += int(not row["text"].strip())
                counts["empty_baseline_hypothesis"] += int(not row["baseline_hypothesis"].strip())
                flags.update(row["alignment_flags"])
        result[dataset] = {**counts, "flags": dict(flags)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

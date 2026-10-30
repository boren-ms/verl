#!/usr/bin/env python3
"""Derive the existing two-parent subsets from complete Qwen candidates."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts.build_openlong_high_error_segments import DATASETS, sha256


def derive(full: Path, selection: Path, destination: Path) -> dict:
    expected = {}
    parents = set()
    with selection.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in expected:
                raise ValueError(f"Duplicate selected ID: {row['id']}")
            expected[row["id"]] = (row["audio_path"], row["baseline_hypothesis"])
            parents.add(row["parent_audio_path"])
    if len(parents) != 2:
        raise ValueError(f"Expected exactly two selected parents in {selection}")
    selected = []
    seen = set()
    with full.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] not in expected:
                continue
            if row["id"] in seen or (row["audio_path"], row["baseline_hypothesis"]) != expected[row["id"]]:
                raise ValueError(f"Duplicate or changed selected segment: {row['id']}")
            seen.add(row["id"])
            selected.append(row)
    if seen != set(expected):
        raise ValueError(f"Missing {len(set(expected) - seen)} selected Qwen segments")
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["parent_audio_path"]].append(row)
    if set(grouped) != parents:
        raise ValueError("Selected parents changed")
    for parent, rows in grouped.items():
        if sorted(r["seg_index"] for r in rows) != list(range(rows[0]["n_segments"])):
            raise ValueError(f"Incomplete selected parent: {parent}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"segments": len(selected), "parents": len(parents),
            "needs_review": sum(r["alignment_status"] == "needs_review" for r in selected),
            "full_sha256": sha256(full), "selection_sha256": sha256(selection),
            "output_sha256": sha256(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"_warning": "Qwen-aligned candidates are not human-verified; flagged rows require review."}
    for dataset in DATASETS:
        summary[dataset] = derive(args.full_dir / f"{dataset}.jsonl", args.selection_dir / f"{dataset}.jsonl",
                                  args.output_dir / f"{dataset}.jsonl")
        print(dataset, json.dumps(summary[dataset]), flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

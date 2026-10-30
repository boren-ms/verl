#!/usr/bin/env python3
"""Check Qwen candidate coverage and source preservation."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.build_openlong_high_error_segments import DATASETS, EXPECTED


def validate(source, output, dataset):
    originals = {}
    parent_words = defaultdict(Counter)
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["id"] in originals:
                    raise ValueError(f"Duplicate source ID: {row['id']}")
                originals[row["id"]] = row
                parent_words[row["parent_audio_path"]].update(row["text"].split())
    if (len(originals), len(parent_words)) != EXPECTED[dataset]:
        raise ValueError(f"Unexpected source coverage: {dataset}")
    seen = set()
    output_words = defaultdict(Counter)
    parent_indexes = defaultdict(list)
    counts = Counter()
    with output.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ident = row["id"]
            if ident in seen or ident not in originals:
                raise ValueError(f"Duplicate or unexpected candidate ID: {ident}")
            source_row = originals[ident]
            for name in ("audio_path", "baseline_hypothesis", "parent_audio_path", "seg_index", "seg_start", "seg_end", "n_segments"):
                if row[name] != source_row[name]:
                    raise ValueError(f"Changed {name}: {ident}")
            if row["prior_inferred_text"] != source_row["text"]:
                raise ValueError(f"Changed prior text: {ident}")
            if row["alignment_method"] != "qwen3_forced_aligner_0.6b":
                raise ValueError(f"Unexpected alignment method: {ident}")
            flags = row["alignment_flags"]
            if row["alignment_status"] != ("needs_review" if flags else "qwen_candidate"):
                raise ValueError(f"Inconsistent review flags: {ident}")
            seen.add(ident)
            parent = row["parent_audio_path"]
            output_words[parent].update(row["text"].split())
            parent_indexes[parent].append(row["seg_index"])
            counts["segments"] += 1
            counts["needs_review"] += int(row["alignment_status"] == "needs_review")
    if seen != set(originals) or set(parent_indexes) != set(parent_words):
        raise ValueError(f"Missing candidate rows or parents: {dataset}")
    for parent, words in parent_words.items():
        if words - output_words[parent]:
            raise ValueError(f"Source words lost in candidate: {parent}")
        if sorted(parent_indexes[parent]) != list(range(len(parent_indexes[parent]))):
            raise ValueError(f"Incomplete parent indices: {parent}")
    return {**counts, "parents": len(parent_indexes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()
    for dataset in DATASETS:
        print(dataset, json.dumps(validate(args.source_dir / f"{dataset}.jsonl",
                                           args.candidate_dir / f"{dataset}.jsonl", dataset)), flush=True)


if __name__ == "__main__":
    main()

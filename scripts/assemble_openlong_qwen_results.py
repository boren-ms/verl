#!/usr/bin/env python3
"""Combine shard results, replace one corpus, and validate full window coverage."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.build_openlong_high_error_segments import sha256


def key_for(window):
    original = window.get("retry_original_core_segment_ids", window["core_segment_ids"])
    return window["dataset"], window["parent_audio_path"], tuple(original)


def assemble(plan_path, shards, replacement, dataset, output_dir):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    planned = {}
    with plan_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                window = json.loads(line)
                key = key_for(window)
                if key in planned:
                    raise ValueError(f"Duplicate planned window: {key}")
                planned[key] = window
    output_dir.mkdir(parents=True, exist_ok=True)
    partial = output_dir / "results.partial.jsonl"
    covered = defaultdict(list)
    counts = Counter()
    with partial.open("x", encoding="utf-8") as out:
        for source in [*shards, replacement]:
            is_replacement = source == replacement
            with source.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    window = row["window"]
                    if is_replacement != (window["dataset"] == dataset):
                        if is_replacement:
                            raise ValueError("Replacement contains an unexpected dataset")
                        continue
                    key = key_for(window)
                    if key not in planned:
                        raise ValueError(f"Unexpected result window: {key}")
                    reference = planned[key]
                    if window.get("retry_reason"):
                        if len(window["core_segment_ids"]) != 1:
                            raise ValueError(f"Invalid retry core: {key}")
                        segment = next((s for s in reference["segments"] if s["id"] == window["core_segment_ids"][0]), None)
                        if (segment is None or window["audio_source"] != reference["audio_source"] or
                            (window["audio_start"], window["audio_end"], window["seed_text"]) !=
                            (segment["seg_start"], segment["seg_end"], segment["text"])):
                            raise ValueError(f"Retry window differs from plan: {key}")
                    elif any(window[field] != reference[field] for field in
                             ("audio_source", "audio_start", "audio_end", "seed_text", "core_word_start", "core_word_end")):
                        raise ValueError(f"Result window differs from plan: {key}")
                    if row["status"] not in ("ok", "empty_seed", "unalignable_seed"):
                        raise ValueError(f"Unresolved alignment result: {key}")
                    covered[key].extend(window["core_segment_ids"])
                    counts["result_rows"] += 1
                    counts[f"status_{row['status']}"] += 1
                    counts["single_segment_retries"] += int(bool(window.get("retry_reason")))
                    counts["collapsed_results"] += int(bool(row.get("quality", {}).get("collapsed")))
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    if set(covered) != set(planned):
        raise ValueError(f"Missing {len(set(planned) - set(covered))} planned windows")
    for key, ids in covered.items():
        if len(ids) != len(set(ids)) or set(ids) != set(key[2]):
            raise ValueError(f"Incomplete or duplicate core coverage: {key}")
    final = output_dir / "results.jsonl"
    partial.rename(final)
    summary = {"planned_windows": len(planned), **counts, "replaced_dataset": dataset,
               "plan_sha256": sha256(plan_path), "results_sha256": sha256(final)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument("--replace-dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.plan, args.shards, args.replacement, args.replace_dataset, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

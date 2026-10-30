#!/usr/bin/env python3
"""Create bounded, overlapping Qwen alignment windows for OpenLong.

This is a preparation manifest only: seed text is the existing hypothesis-based
partition, not a forced-aligned or verified transcript.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts.build_openlong_high_error_segments import DATASETS, EXPECTED


def build_windows(dataset, rows, core_segments=3, context_segments=1, max_seconds=150.0):
    groups = defaultdict(list)
    for row in rows:
        groups[row["parent_audio_path"]].append(row)
    if (len(rows), len(groups)) != EXPECTED[dataset]:
        raise ValueError(f"Incomplete {dataset}: {len(rows)} segments, {len(groups)} parents")
    windows = []
    for parent in sorted(groups):
        group = sorted(groups[parent], key=lambda row: row["seg_index"])
        if [r["seg_index"] for r in group] != list(range(len(group))) or any(r["n_segments"] != len(group) for r in group):
            raise ValueError(f"Incomplete segment group: {parent}")
        audio_members = {r["audio_path"].rsplit("#", 1)[0] for r in group}
        if len(audio_members) != 1:
            raise ValueError(f"Multiple audio members for one parent: {parent}")
        member = audio_members.pop()
        core_start = 0
        while core_start < len(group):
            core_end = min(core_start + core_segments, len(group))
            left = max(0, core_start - context_segments)
            right = min(len(group), core_end + context_segments)
            # Reduce core first, then context, while retaining every core segment.
            while True:
                start = min(r["seg_start"] for r in group[left:right])
                end = max(r["seg_end"] for r in group[left:right])
                if end - start <= max_seconds:
                    break
                if core_end - core_start > 1:
                    core_end -= 1
                    right = min(len(group), core_end + context_segments)
                elif right > core_end:
                    right -= 1
                elif left < core_start:
                    left += 1
                else:
                    raise ValueError(f"Segment exceeds window duration: {parent} index={core_start}")
            pieces = [r["text"] for r in group[left:right]]
            word_counts = [len(piece.split()) for piece in pieces]
            before = sum(word_counts[:core_start - left])
            through = before + sum(word_counts[core_start - left:core_end - left])
            windows.append({
                "dataset": dataset,
                "parent_audio_path": parent,
                "audio_source": member,
                "audio_start": start,
                "audio_end": end,
                "seed_text": " ".join(piece for piece in pieces if piece),
                "core_word_start": before,
                "core_word_end": through,
                "core_segment_ids": [r["id"] for r in group[core_start:core_end]],
                "core_segment_indexes": list(range(core_start, core_end)),
                "context_segment_indexes": list(range(left, right)),
                "segments": [{"id": r["id"], "seg_index": r["seg_index"],
                              "seg_start": r["seg_start"], "seg_end": r["seg_end"], "text": r["text"]}
                             for r in group[left:right]],
                "seed_is_provisional": True,
            })
            core_start = core_end
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--two-parent-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--core-segments", type=int, default=3)
    parser.add_argument("--context-segments", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=150.0)
    args = parser.parse_args()
    if args.core_segments < 1 or args.context_segments < 0 or not 0 < args.max_seconds <= 180:
        parser.error("core-segments >= 1, context-segments >= 0, and 0 < max-seconds <= 180 are required")
    selected_summary = json.loads(args.two_parent_summary.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"_note": "Planning only; seed text is provisional and no Qwen inference has run.",
               "core_segments": args.core_segments, "context_segments": args.context_segments,
               "max_window_seconds": args.max_seconds}
    with (args.output_dir / "windows.jsonl").open("w", encoding="utf-8") as out:
        for dataset in DATASETS:
            source = args.input_dir / f"{dataset}.jsonl"
            with source.open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            selected = {item["parent"] for item in selected_summary[dataset]["parents"]}
            if len(selected) != 2 or not selected.issubset({r["parent_audio_path"] for r in rows}):
                raise ValueError(f"Two-parent subset not present in {dataset}")
            windows = build_windows(dataset, rows, args.core_segments, args.context_segments, args.max_seconds)
            core_ids = [sid for window in windows for sid in window["core_segment_ids"]]
            if len(core_ids) != len(rows) or len(set(core_ids)) != len(rows):
                raise ValueError(f"Window core coverage mismatch: {dataset}")
            for window in windows:
                window["in_two_parent_subset"] = window["parent_audio_path"] in selected
                out.write(json.dumps(window, ensure_ascii=False) + "\n")
            subset = [window for window in windows if window["in_two_parent_subset"]]
            summary[dataset] = {"parents": len({r["parent_audio_path"] for r in rows}), "segments": len(rows),
                                "windows": len(windows), "two_parent_windows": len(subset),
                                "empty_seed_windows": sum(not w["seed_text"].strip() for w in windows),
                                "max_window_seconds": max(w["audio_end"] - w["audio_start"] for w in windows)}
            print(dataset, json.dumps(summary[dataset]), flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

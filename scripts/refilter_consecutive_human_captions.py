# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Derive audited duration-filter versions from immutable Qwen raw alignment."""

import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
import math
from contextlib import ExitStack
from decimal import Decimal
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "original_alignment", Path(__file__).with_name("align_long_human_captions.py")
)
alignment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alignment)


def filter_consecutive_words(row, *, maximum_consecutive, long_seconds):
    """Count maximal same-category word runs independently within one parent."""
    if maximum_consecutive < 1 or long_seconds != 5:
        raise ValueError("Invalid consecutive-word policy")
    timestamps = row.get("word_timestamps", [])
    complete = bool(timestamps) and len(timestamps) == len(row["text"].split())
    runs, outliers, invalid = [], [], []
    current = None
    for index, word in enumerate(timestamps):
        start, end = word["start"], word["end"]
        duration = end - start
        finite = all(math.isfinite(v) for v in (start, end, duration))
        if finite:
            duration = float(Decimal(str(end)) - Decimal(str(start)))
        if not finite or duration < 0:
            invalid.append(index)
        category = "zero" if finite and duration == 0 else "above_5" if finite and duration > long_seconds else None
        if category:
            outliers.append(
                {
                    "word_index": index,
                    "text": word["text"],
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "reason": category,
                }
            )
        if current and current["category"] != category:
            runs.append(current)
            current = None
        if category:
            if current is None:
                current = {"category": category, "word_start": index, "word_end": index + 1, "length": 1}
            else:
                current["word_end"] = index + 1
                current["length"] += 1
    if current:
        runs.append(current)
    maxima = {
        category: max((r["length"] for r in runs if r["category"] == category), default=0)
        for category in ("zero", "above_5")
    }
    passed = complete and not invalid and not row.get("alignment_error") and max(maxima.values()) <= maximum_consecutive
    previous_v2 = (
        complete
        and not invalid
        and not row.get("alignment_error")
        and sum(not 0 < w["end"] - w["start"] < 5 for w in timestamps) <= 1
    )
    return {
        **row,
        "previous_word_duration_filter": row.get("word_duration_filter"),
        "previous_v2_filter_passed": previous_v2,
        "word_duration_filter": {
            "scope": "entire_original_recording",
            "passed": passed,
            "complete_word_coverage": complete,
            "maximum_consecutive_words": maximum_consecutive,
            "minimum": 0,
            "maximum": long_seconds,
            "bounds": "normal_0_exclusive_5_inclusive",
            "zero_condition": "duration == 0",
            "long_condition": "duration > 5",
            "invalid_word_indices": invalid,
            "outlier_count": len(outliers),
            "outliers": outliers,
            "runs": runs,
            "maximum_runs": maxima,
        },
    }


def update_run_statistics(stats, quality):
    """Record run counts, contained word counts, and parent maximum-run counts."""
    for category in ("zero", "above_5"):
        own = stats[category]
        own["parent_maximum_run_histogram"][str(quality["maximum_runs"][category])] += 1
        for run in quality["runs"]:
            if run["category"] == category:
                length = run["length"]
                own["run_length_histogram"][str(length)] += 1
                own["word_count_by_run_length"][str(length)] += length


def derive(args):
    """Stream one source shard once, retain decisions, and hash every new manifest."""
    source_receipt = json.loads(args.source_receipt.read_text())
    if source_receipt["runner_sha256"] != alignment.fingerprint(Path(alignment.__file__)):
        raise ValueError("Original alignment runner changed from the trusted source receipt")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    counts = collections.Counter()
    stats = {
        category: {
            name: collections.Counter()
            for name in ("run_length_histogram", "word_count_by_run_length", "parent_maximum_run_histogram")
        }
        for category in ("zero", "above_5")
    }
    histogram = collections.Counter()
    versions = {
        name: {"rows": 0, "words": 0, "seconds": Decimal(0), "segment_units": 0}
        for name in ("filtered_originals", "segments", "pairs")
    }
    seen, chunks, accepted_outliers = set(), set(), collections.Counter()
    raw_hash = hashlib.sha256()
    names = ["filtered_originals", "segments", "pairs", "quality_decisions", "pair_exclusions"]
    policy = {
        "zero_word_condition": "duration == 0",
        "long_word_condition": "duration > 5",
        "maximum_long_word_seconds": args.maximum_long_word_seconds,
        "maximum_consecutive_words": args.maximum_consecutive_words,
        "reject_at_consecutive_words": args.maximum_consecutive_words + 1,
        "scope": "entire_original_parent",
        "categories_counted_separately": True,
        "segment_max_seconds": 30,
        "alignment_failure_policy": "retain_and_exclude",
        "timestamp_policy": "unchanged_original_Qwen_predictions",
    }
    with ExitStack() as stack:
        outputs = {
            name: stack.enter_context(
                gzip.open(args.output_dir / f"{name}.jsonl.gz", "wt", encoding="utf-8", compresslevel=1)
            )
            for name in names
        }
        source = stack.enter_context(args.input.open("rb"))
        for line in source:
            raw_hash.update(line)
            old = json.loads(line)
            if old["id"] in seen:
                raise ValueError("Duplicate source parent ID")
            seen.add(old["id"])
            chunks.add(old["chunk_name"])
            counts["parents"] += 1
            row = filter_consecutive_words(
                old, maximum_consecutive=args.maximum_consecutive_words, long_seconds=args.maximum_long_word_seconds
            )
            quality = row["word_duration_filter"]
            old_passed = old["word_duration_filter"]["passed"]
            if old_passed:
                counts["previous_accepted_originals"] += 1
            counts["previous_v2_accepted_originals"] += row["previous_v2_filter_passed"]
            if row.get("alignment_error"):
                reason = "alignment_failure"
                counts["alignment_failures"] += 1
            elif not quality["complete_word_coverage"]:
                reason = "incomplete_word_coverage"
                counts["incomplete_word_coverage"] += 1
            elif not quality["passed"]:
                reason = "consecutive_word_run"
                counts["word_duration_exclusions"] += 1
                zero = quality["maximum_runs"]["zero"] > args.maximum_consecutive_words
                long = quality["maximum_runs"]["above_5"] > args.maximum_consecutive_words
                if quality["invalid_word_indices"]:
                    counts["invalid_word_timings"] += 1
                else:
                    counts[
                        "reject_both_runs" if zero and long else "reject_zero_run" if zero else "reject_long_run"
                    ] += 1
            else:
                reason = None
                # This preserves valid positive segment bounds, overlap checks and
                # exact human word coverage even when one duration outlier is allowed.
                verified = alignment.segment_row(row)
                if verified["segments"] != old["segments"] or verified["segment_metadata"] != old["segment_metadata"]:
                    raise ValueError("Re-segmentation changed the retained original predictions")
                row = verified
            if (old_passed or row["previous_v2_filter_passed"]) and not quality["passed"]:
                raise ValueError("New policy unexpectedly removed a previously accepted original")
            if not row.get("alignment_error"):
                histogram[str(quality["outlier_count"])] += 1
                update_run_statistics(stats, quality)
            alignment.write_row(
                outputs["quality_decisions"],
                {
                    "id": row["id"],
                    "audio_chunk": row["audio_chunk"],
                    "raw_alignment_path": str(args.input.resolve()),
                    "previous_filter_passed": old_passed,
                    "previous_v2_filter_passed": row["previous_v2_filter_passed"],
                    "reason": reason,
                    "alignment_error": row.get("alignment_error"),
                    "word_duration_filter": quality,
                },
            )
            own_pairs = 0
            if quality["passed"]:
                counts["filtered_originals"] += 1
                counts["newly_accepted_originals"] += not old_passed
                counts["newly_accepted_vs_v2"] += not row["previous_v2_filter_passed"]
                accepted_outliers[str(quality["outlier_count"])] += 1
                originals = {
                    k: v for k, v in row.items() if k not in {"segments", "segment_metadata", "segment_max_seconds"}
                }
                alignment.write_row(outputs["filtered_originals"], originals)
                segmented = {
                    **row,
                    "audio_path": [part[0] for part in row["segments"]],
                    "segment_reference_texts": [part[1] for part in row["segments"]],
                }
                alignment.write_row(outputs["segments"], segmented)
                words = len(row["text"].split())
                for name in ("filtered_originals", "segments"):
                    versions[name]["rows"] += 1
                    versions[name]["words"] += words
                versions["filtered_originals"]["seconds"] += Decimal(str(row["alignment_audio_duration"]))
                versions["segments"]["segment_units"] += len(row["segments"])
                versions["segments"]["seconds"] += sum(
                    (Decimal(str(m["end"])) - Decimal(str(m["start"])) for m in row["segment_metadata"]), Decimal(0)
                )
                for pair in alignment.paired_rows(row):
                    alignment.write_row(outputs["pairs"], pair)
                    own_pairs += 1
                    versions["pairs"]["rows"] += 1
                    versions["pairs"]["words"] += len(pair["text"].split())
                    versions["pairs"]["segment_units"] += 2
                    versions["pairs"]["seconds"] += sum(
                        (Decimal(str(m["end"])) - Decimal(str(m["start"])) for m in pair["segment_metadata"]),
                        Decimal(0),
                    )
                counts["pairs"] += own_pairs
                counts["pair_eligible_parents"] += bool(own_pairs)
            if not own_pairs:
                counts["pair_excluded_parents"] += 1
                counts["fewer_than_two_segment_parents"] += quality["passed"]
                alignment.write_row(
                    outputs["pair_exclusions"],
                    {"id": row["id"], "reason": reason or "fewer_than_two_segments", "word_duration_filter": quality},
                )
            if counts["parents"] % 5000 == 0:
                print(json.dumps(dict(counts)), flush=True)
    if raw_hash.hexdigest() != source_receipt["sha256"] or counts["parents"] != source_receipt["rows"]:
        raise ValueError("Source raw alignment hash/count differs from its verified receipt")
    for value in versions.values():
        value["hours"] = float(value["seconds"] / Decimal(3600))
        value["seconds"] = str(value["seconds"])
    alignment.save(
        args.output_dir / "RUN_STATISTICS.json",
        {
            "population": "complete_aligned_parents_without_alignment_error",
            "aligned_parents": counts["parents"] - counts["alignment_failures"],
            "categories": {c: {k: dict(v) for k, v in own.items()} for c, own in stats.items()},
        },
    )
    manifest_hashes = {
        f"{name}.jsonl.gz": alignment.fingerprint(args.output_dir / f"{name}.jsonl.gz") for name in names
    }
    manifest_hashes["RUN_STATISTICS.json"] = alignment.fingerprint(args.output_dir / "RUN_STATISTICS.json")
    alignment.save(
        args.output_dir / "COMPLETE.json",
        {
            "status": "locally_derived_and_source_hash_verified",
            "source_raw_alignment": str(args.input.resolve()),
            "source_raw_alignment_sha256": raw_hash.hexdigest(),
            "source_alignment_receipt_sha256": alignment.fingerprint(args.source_receipt),
            "source_alignment_runner_sha256": source_receipt["runner_sha256"],
            "refilter_runner_sha256": alignment.fingerprint(Path(__file__)),
            "policy": policy,
            "counts": dict(counts),
            "versions": versions,
            "outlier_count_histogram_aligned_parents": dict(histogram),
            "accepted_outlier_count_histogram": dict(accepted_outliers),
            "source_chunks": sorted(chunks),
            "manifest_sha256": manifest_hashes,
            "audio_files_written": False,
            "training_started": False,
        },
    )


def main():
    """Parse an explicit new policy and create a fresh, separate version directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-long-word-seconds", type=float, default=5)
    parser.add_argument("--maximum-consecutive-words", type=int, default=3)
    derive(parser.parse_args())


if __name__ == "__main__":
    main()

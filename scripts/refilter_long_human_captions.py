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


def filter_words(row, *, minimum, maximum, maximum_outliers):
    """Allow a bounded number of atypical words with strict, open duration bounds."""
    if not math.isfinite(minimum) or not math.isfinite(maximum) or not 0 <= minimum < maximum:
        raise ValueError("Invalid word-duration bounds")
    if not isinstance(maximum_outliers, int) or maximum_outliers < 0:
        raise ValueError("Maximum outlier count must be a nonnegative integer")
    timestamps = row.get("word_timestamps", [])
    complete = bool(timestamps) and len(timestamps) == len(row["text"].split())
    outliers = []
    finite = True
    for index, word in enumerate(timestamps):
        start, end = word["start"], word["end"]
        duration = end - start
        finite &= all(math.isfinite(value) for value in (start, end, duration))
        if not math.isfinite(duration) or not minimum < duration < maximum:
            outliers.append(
                {
                    "word_index": index,
                    "text": word["text"],
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "reason": "nonfinite"
                    if not math.isfinite(duration)
                    else "too_short"
                    if duration <= minimum
                    else "too_long",
                }
            )
    passed = complete and finite and not row.get("alignment_error") and len(outliers) <= maximum_outliers
    return {
        **row,
        "previous_word_duration_filter": row.get("word_duration_filter"),
        "word_duration_filter": {
            "minimum": minimum,
            "maximum": maximum,
            "bounds": "strict_open",
            "maximum_outlier_words": maximum_outliers,
            "scope": "entire_original_recording",
            "complete_word_coverage": complete,
            "passed": passed,
            "outlier_count": len(outliers),
            "outliers": outliers,
        },
    }


def derive(args):
    """Stream one source shard once, retain decisions, and hash every new manifest."""
    source_receipt = json.loads(args.source_receipt.read_text())
    if source_receipt["runner_sha256"] != alignment.fingerprint(Path(alignment.__file__)):
        raise ValueError("Original alignment runner changed from the trusted source receipt")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    counts = collections.Counter()
    histogram = collections.Counter()
    versions = {
        name: {"rows": 0, "words": 0, "seconds": Decimal(0), "segment_units": 0}
        for name in ("filtered_originals", "segments", "pairs")
    }
    seen, chunks, accepted_outliers = set(), set(), collections.Counter()
    raw_hash = hashlib.sha256()
    names = ["filtered_originals", "segments", "pairs", "quality_decisions", "pair_exclusions"]
    policy = {
        "minimum_word_seconds": args.minimum_word_seconds,
        "maximum_word_seconds": args.maximum_word_seconds,
        "bounds": "strict_open",
        "maximum_outlier_words": args.maximum_outlier_words,
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
            row = filter_words(
                old,
                minimum=args.minimum_word_seconds,
                maximum=args.maximum_word_seconds,
                maximum_outliers=args.maximum_outlier_words,
            )
            quality = row["word_duration_filter"]
            old_passed = old["word_duration_filter"]["passed"]
            if old_passed:
                counts["previous_accepted_originals"] += 1
            if row.get("alignment_error"):
                reason = "alignment_failure"
                counts["alignment_failures"] += 1
            elif not quality["complete_word_coverage"]:
                reason = "incomplete_word_coverage"
                counts["incomplete_word_coverage"] += 1
            elif not quality["passed"]:
                reason = "word_duration_outlier_count"
                counts["word_duration_exclusions"] += 1
            else:
                reason = None
                # This preserves valid positive segment bounds, overlap checks and
                # exact human word coverage even when one duration outlier is allowed.
                verified = alignment.segment_row(row)
                if verified["segments"] != old["segments"] or verified["segment_metadata"] != old["segment_metadata"]:
                    raise ValueError("Re-segmentation changed the retained original predictions")
                row = verified
            if old_passed and not quality["passed"]:
                raise ValueError("New policy unexpectedly removed a previously accepted original")
            if not row.get("alignment_error"):
                histogram[str(quality["outlier_count"])] += 1
            alignment.write_row(
                outputs["quality_decisions"],
                {
                    "id": row["id"],
                    "audio_chunk": row["audio_chunk"],
                    "raw_alignment_path": str(args.input.resolve()),
                    "previous_filter_passed": old_passed,
                    "reason": reason,
                    "alignment_error": row.get("alignment_error"),
                    "word_duration_filter": quality,
                },
            )
            own_pairs = 0
            if quality["passed"]:
                counts["filtered_originals"] += 1
                counts["newly_accepted_originals"] += not old_passed
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
    manifest_hashes = {
        f"{name}.jsonl.gz": alignment.fingerprint(args.output_dir / f"{name}.jsonl.gz") for name in names
    }
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
    parser.add_argument("--minimum-word-seconds", type=float, required=True)
    parser.add_argument("--maximum-word-seconds", type=float, required=True)
    parser.add_argument("--maximum-outlier-words", type=int, required=True)
    derive(parser.parse_args())


if __name__ == "__main__":
    main()

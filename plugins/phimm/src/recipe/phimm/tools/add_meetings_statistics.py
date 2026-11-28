# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Attach recording, dataset and overall statistics to meetings biasing JSONL."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from recipe.phimm.data.biasing import normalize_phrase


def segment_durations(row):
    """Read saved time selectors without loading or changing audio."""
    durations = []
    for path in row["audio_path"]:
        start, end = map(float, path.rsplit("#", 1)[1].split(":"))
        if not 0 <= start < end:
            raise ValueError(f"Invalid audio selector: {path}")
        durations.append(round(end - start, 6))
    if len(durations) != row["n_segments"] or not durations:
        raise ValueError(f"Segment count mismatch for {row['id']}")
    return durations


def histogram(values):
    """Count each observed value, retaining stable ascending keys."""
    return {str(key): count for key, count in sorted(Counter(values).items())}


def group_statistics(rows):
    """Summarize samples, counting a matched keyword once per recording."""
    durations = [length for row in rows for length in segment_durations(row)]
    return {
        "sample_count": len(rows),
        "segment_count": len(durations),
        "mean_segment_duration_seconds": mean(durations),
        "mean_ground_truth_keywords_per_sample": mean(len(row["keywords"]) for row in rows),
        "unique_biasing_word_count": len({normalize_phrase(word) for row in rows for word in row["biasing_words"]}),
        "unique_ground_truth_keyword_count": len({normalize_phrase(word) for row in rows for word in row["keywords"]}),
        "biasing_word_count_distribution": histogram(len(row["biasing_words"]) for row in rows),
        "ground_truth_keyword_count_distribution": histogram(len(row["keywords"]) for row in rows),
    }


def enrich_rows(rows, *, dataset_statistics, overall_statistics):
    """Attach metadata while explicitly marking segment keyword counts unknown."""
    for row in rows:
        durations = segment_durations(row)
        candidates = row["biasing_words"]
        row["statistics"] = {
            "reference_scope": "parent",
            "keyword_count_definition": (
                "unique matched candidate phrases in the complete recording, not spoken occurrence counts"
            ),
            "segments": {
                "count": len(durations),
                "durations_seconds": durations,
                "mean_duration_seconds": mean(durations),
                "min_duration_seconds": min(durations),
                "max_duration_seconds": max(durations),
                "duration_distribution": {
                    f"{start}-{end}s": sum(start <= duration < end for duration in durations)
                    for start, end in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30))
                },
                "ground_truth_keyword_counts": None,
                "ground_truth_keyword_counts_reason": (
                    "No segment-level aligned transcription; "
                    "parent keyword counts do not locate hits in individual segments"
                ),
            },
            "biasing": {
                "word_count": len(candidates),
                "ground_truth_keyword_count": len(row["keywords"]),
                "candidate_hit_ratio": len(row["keywords"]) / len(candidates) if candidates else None,
                "phrase_word_count_distribution": histogram(len(word.split()) for word in candidates),
            },
            "dataset": dataset_statistics,
            "overall": overall_statistics,
        }
    return rows


def main():
    """Write enriched copies; the supplied input set defines overall coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if len({path.name for path in args.inputs}) != len(args.inputs):
        parser.error("Input basenames must be distinct")
    datasets = [[json.loads(line) for line in path.read_text().splitlines() if line.strip()] for path in args.inputs]
    if any(not rows for rows in datasets):
        parser.error("Inputs must contain recording samples")
    overall = group_statistics([row for rows in datasets for row in rows])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, rows in zip(args.inputs, datasets, strict=True):
        enriched = enrich_rows(rows, dataset_statistics=group_statistics(rows), overall_statistics=overall)
        output = args.output_dir / path.name
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in enriched))
        print(f"{output}: {len(enriched)} samples")


if __name__ == "__main__":
    main()

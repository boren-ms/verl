#!/usr/bin/env python3
"""Convert cleaned earnings-call results_subsegments.tsv rows to training JSONL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath


REQUIRED_COLUMNS = (
    "Audio",
    "Transcription",
    "Duration_Seconds",
    "Original_Audio_URL",
    "Start_Seconds",
    "End_Seconds",
    "Parent_Audio",
    "Part_Start_Seconds",
    "Part_End_Seconds",
)


def _number(row: dict[str, str], field: str, line_number: int) -> float:
    try:
        value = float(row[field])
    except ValueError as error:
        raise ValueError(f"Invalid {field} on TSV line {line_number}: {row[field]!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field} on TSV line {line_number}: {value}")
    return value


def _audio_path(value: str, line_number: int) -> str:
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not normalized.startswith("audio/"):
        raise ValueError(f"Invalid relative audio path on TSV line {line_number}: {value!r}")
    if path.suffix.lower() != ".wav":
        raise ValueError(f"Expected WAV audio on TSV line {line_number}: {value!r}")
    return normalized


def convert(tsv_path: Path, output_path: Path, *, overwrite: bool = False) -> dict[str, object]:
    records = []
    timing_mismatches = []
    with tsv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        missing_columns = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(f"TSV is missing required columns: {missing_columns}")

        for line_number, row in enumerate(reader, start=2):
            audio_path = _audio_path(row["Audio"], line_number)
            transcription = row["Transcription"].strip()
            if not transcription:
                raise ValueError(f"Empty Transcription on TSV line {line_number}")

            duration = _number(row, "Duration_Seconds", line_number)
            start = _number(row, "Start_Seconds", line_number)
            end = _number(row, "End_Seconds", line_number)
            part_start = _number(row, "Part_Start_Seconds", line_number)
            part_end = _number(row, "Part_End_Seconds", line_number)
            if duration <= 0 or end <= start or part_end <= part_start:
                raise ValueError(f"Invalid non-positive time span on TSV line {line_number}")
            if not math.isclose(part_end - part_start, duration, abs_tol=0.02):
                timing_mismatches.append(
                    {
                        "line": line_number,
                        "audio_path": audio_path,
                        "duration_seconds": duration,
                        "part_span_seconds": part_end - part_start,
                    }
                )

            records.append(
                {
                    "id": PurePosixPath(audio_path).stem,
                    "audio_path": audio_path,
                    "display_ne_tagged": transcription,
                    "language": "English",
                    "audio_duration_seconds": duration,
                    "original_audio_url": row["Original_Audio_URL"].strip(),
                    "start_seconds": start,
                    "end_seconds": end,
                    "parent_audio": row["Parent_Audio"].strip().replace("\\", "/"),
                    "part_start_seconds": part_start,
                    "part_end_seconds": part_end,
                }
            )

    if not records:
        raise ValueError("TSV contains no data rows")
    for field in ("audio_path", "id"):
        duplicates = [value for value, count in Counter(row[field] for row in records).items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate {field} values: {duplicates[:10]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with output_path.open(mode, encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    tagged_rows = sum("<NE>" in row["display_ne_tagged"] or "<PName>" in row["display_ne_tagged"] for row in records)
    return {
        "input": str(tsv_path),
        "output": str(output_path),
        "records": len(records),
        "duration_hours": sum(row["audio_duration_seconds"] for row in records) / 3600,
        "tagged_rows": tagged_rows,
        "timing_mismatch_count": len(timing_mismatches),
        "timing_mismatch_examples": timing_mismatches[:10],
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(convert(args.tsv, args.output, overwrite=args.overwrite), indent=2))


if __name__ == "__main__":
    main()

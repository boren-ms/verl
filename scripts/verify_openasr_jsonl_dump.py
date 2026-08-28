#!/usr/bin/env python3
"""Verify OpenASR JSONL/audio dumps against source parquet metadata."""

from __future__ import annotations

import argparse
import io
import json
from typing import Any

import blobfile as bf
import soundfile as sf

from batch_dump_openasr_parquets_to_jsonl import parquet_num_rows, split_name
from dump_openasr_parquet_to_jsonl import join_path


REQUIRED_FIELDS = {"dataset", "text", "id", "audio_length_s", "audio_path", "sampling_rate", "duration"}


def add_problem(problems: list[str], problem: str, max_problems: int) -> None:
    if len(problems) < max_problems:
        problems.append(problem)


def scan_jsonl(
    jsonl_path: str,
    output_dir: str,
    expected_dataset: str,
    max_problems: int,
) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    count = 0
    first = None
    last = None
    problems: list[str] = []
    with bf.BlobFile(jsonl_path, "r") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            row_index = count
            count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                add_problem(problems, f"line {count}: invalid json: {exc}", max_problems)
                continue

            missing_fields = REQUIRED_FIELDS - record.keys()
            if missing_fields:
                add_problem(problems, f"line {count}: missing fields {sorted(missing_fields)}", max_problems)

            expected_audio_path = join_path(output_dir, "audio", f"{row_index}.wav")
            if record.get("audio_path") != expected_audio_path:
                add_problem(
                    problems,
                    f"line {count}: audio_path {record.get('audio_path')!r} != {expected_audio_path!r}",
                    max_problems,
                )
            if record.get("dataset") != expected_dataset:
                add_problem(problems, f"line {count}: dataset {record.get('dataset')!r} != {expected_dataset!r}", max_problems)
            if not isinstance(record.get("text"), str):
                add_problem(problems, f"line {count}: text is not a string", max_problems)
            if not isinstance(record.get("id"), str) or not record.get("id"):
                add_problem(problems, f"line {count}: id is missing or not a string", max_problems)
            if not isinstance(record.get("sampling_rate"), int) or record.get("sampling_rate") <= 0:
                add_problem(problems, f"line {count}: invalid sampling_rate {record.get('sampling_rate')!r}", max_problems)
            for field in ("duration", "audio_length_s"):
                value = record.get(field)
                if not isinstance(value, (int, float)) or value <= 0:
                    add_problem(problems, f"line {count}: invalid {field} {value!r}", max_problems)

            if first is None:
                first = record
            last = record
    return count, first, last, problems


def audio_readable(audio_path: str) -> bool:
    with bf.BlobFile(audio_path, "rb") as audio_file:
        info = sf.info(io.BytesIO(audio_file.read()))
    return info.frames > 0 and info.samplerate > 0


def verify_split(
    input_path: str,
    input_root: str,
    output_root: str,
    check_audio_readable: bool,
    max_problems: int,
) -> dict[str, Any]:
    dataset, split, expected_dataset = split_name(input_path, input_root)
    split_key = f"{dataset}/{split}"
    output_dir = join_path(output_root, dataset, split)
    jsonl_path = join_path(output_dir, "data.jsonl")
    expected_rows = parquet_num_rows(input_path)
    problems = []

    if not bf.exists(jsonl_path):
        return {"split": split_key, "rows": 0, "expected_rows": expected_rows, "problems": ["missing data.jsonl"]}

    count, first, last, problems = scan_jsonl(jsonl_path, output_dir, expected_dataset, max_problems)
    if count != expected_rows:
        add_problem(problems, f"row_count {count} != {expected_rows}", max_problems)

    if expected_rows:
        audio_prefix = join_path(output_dir, "audio") + "/"
        expected_audio_paths = {
            "first": audio_prefix + "0.wav",
            "last": audio_prefix + f"{expected_rows - 1}.wav",
        }
        actual_first_audio = first.get("audio_path") if first else None
        actual_last_audio = last.get("audio_path") if last else None
        if actual_first_audio != expected_audio_paths["first"]:
            add_problem(problems, f"first_audio_path {actual_first_audio!r} != {expected_audio_paths['first']!r}", max_problems)
        if actual_last_audio != expected_audio_paths["last"]:
            add_problem(problems, f"last_audio_path {actual_last_audio!r} != {expected_audio_paths['last']!r}", max_problems)

        for label, audio_path in expected_audio_paths.items():
            if not bf.exists(audio_path):
                add_problem(problems, f"missing {label} audio {audio_path}", max_problems)
            elif check_audio_readable and not audio_readable(audio_path):
                add_problem(problems, f"unreadable {label} audio {audio_path}", max_problems)

    return {"split": split_key, "rows": count, "expected_rows": expected_rows, "problems": problems}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--skip-audio-read", action="store_true")
    parser.add_argument("--max-problems-per-split", type=int, default=20)
    args = parser.parse_args()

    input_paths = sorted(bf.glob(join_path(args.input_root, "*", "*.parquet")))
    expected_jsonl_paths = set()
    for input_path in input_paths:
        dataset, split, _ = split_name(input_path, args.input_root)
        expected_jsonl_paths.add(join_path(args.output_root, dataset, split, "data.jsonl"))
    actual_jsonl_paths = set(bf.glob(join_path(args.output_root, "*", "*", "data.jsonl")))
    missing_jsonl_paths = sorted(expected_jsonl_paths - actual_jsonl_paths)
    extra_jsonl_paths = sorted(actual_jsonl_paths - expected_jsonl_paths)

    results = [
        verify_split(
            input_path=input_path,
            input_root=args.input_root,
            output_root=args.output_root,
            check_audio_readable=not args.skip_audio_read,
            max_problems=args.max_problems_per_split,
        )
        for input_path in input_paths
    ]
    failures = [result for result in results if result["problems"]]
    summary = {
        "inputs": len(input_paths),
        "generated_data_jsonl": len(actual_jsonl_paths),
        "verified_outputs": len(results) - len(failures),
        "total_rows": sum(int(result["rows"]) for result in results if not result["problems"]),
        "missing_data_jsonl": missing_jsonl_paths,
        "extra_data_jsonl": extra_jsonl_paths,
        "failures": failures,
        "first_verified": [result for result in results if not result["problems"]][:3],
        "last_verified": [result for result in results if not result["problems"]][-3:],
    }
    print(json.dumps(summary, indent=2))
    if failures or missing_jsonl_paths or extra_jsonl_paths:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
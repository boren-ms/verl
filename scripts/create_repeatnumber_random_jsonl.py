#!/usr/bin/env python3
"""Create evaluation JSONL from a RepeatNumber filenames manifest."""

import argparse
import csv
import json
from pathlib import Path


DATASET_ROOT = "az://orngwus2cresco/data/speech/am_data/gpt_tts/multi_locale_tts/tier1/en-us/txflow"
DEFAULT_DATASET_NAME = "RepeatNumber_en-US_random_TTS_DTEST_FY27Q1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", nargs="?", default=DEFAULT_DATASET_NAME)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    manifest_path = Path(f"/tmp/{dataset_name}_filenames.txt")
    output_path = Path(f"/home/boren/data/bad_cases/digits/{dataset_name}.jsonl")
    source_prefix = f"{DATASET_ROOT}/{dataset_name}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(newline="", encoding="utf-8") as manifest, output_path.open(
        "w", encoding="utf-8"
    ) as output:
        for row_number, row in enumerate(csv.reader(manifest, delimiter="\t"), start=1):
            if len(row) != 3:
                raise ValueError(f"Expected 3 tab-separated columns on line {row_number}, got {len(row)}")

            filename, transcription, text = (value.strip() for value in row)
            if not filename.endswith(".wav"):
                raise ValueError(f"Expected a WAV filename on line {row_number}: {filename!r}")

            record = {
                "id": Path(filename).stem,
                "text": text,
                "display_transcription": text,
                "transcription": transcription,
                "audio_path": f"{source_prefix}/audio/{filename}",
            }
            output.write(json.dumps(record, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
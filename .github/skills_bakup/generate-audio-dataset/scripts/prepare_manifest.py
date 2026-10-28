#!/usr/bin/env python3
"""Normalize JSONL or one-text-per-line input into a TTS JSONL manifest."""

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source JSONL or text file")
    parser.add_argument("output", type=Path, help="Output JSONL manifest")
    parser.add_argument("--text-key", default="text", help="Source field containing text")
    parser.add_argument("--id-key", default="id", help="Source field containing IDs")
    parser.add_argument("--audio-dir", default="audios", help="Relative audio directory")
    parser.add_argument("--id-prefix", default="item", help="Prefix for generated IDs")
    return parser.parse_args()


def parse_row(line: str, line_number: int) -> tuple[dict, bool]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        value = line
    if isinstance(value, str):
        return {"text": value}, False
    if not isinstance(value, dict):
        raise ValueError(f"Line {line_number}: expected a JSON object or text")
    return value, True


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not stem:
        raise ValueError(f"ID {value!r} cannot form an audio filename")
    return stem


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    rows = []
    seen_ids = set()
    seen_audio_paths = set()

    with input_path.open(encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row, is_json_object = parse_row(line, line_number)
            source_text_key = args.text_key if is_json_object else "text"
            text = row.get(source_text_key)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Line {line_number}: missing non-empty {args.text_key!r}")

            item_id = row.get(args.id_key)
            if item_id is None or str(item_id).strip() == "":
                item_id = f"{args.id_prefix}_{len(rows) + 1:06d}"
            item_id = str(item_id)
            if item_id in seen_ids:
                raise ValueError(f"Line {line_number}: duplicate ID {item_id!r}")

            audio_path = f"{args.audio_dir.rstrip('/')}/{safe_stem(item_id)}.wav"
            if audio_path in seen_audio_paths:
                raise ValueError(f"Line {line_number}: duplicate audio path {audio_path!r}")

            row["id"] = item_id
            row["text"] = text.strip()
            row["audio_path"] = audio_path
            rows.append(row)
            seen_ids.add(item_id)
            seen_audio_paths.add(audio_path)

    if not rows:
        raise ValueError("Input contains no non-empty rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
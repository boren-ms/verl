#!/usr/bin/env python3
"""Export valid Rover v1 references as a flat JSONL for the ASR loader."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def prepare(source: Path, output: Path) -> dict:
    if source.resolve() == output.resolve():
        raise ValueError("The output must differ from the tagged source")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    excluded = []
    seen = set()
    # Validate all rows before opening the output, so malformed input cannot
    # leave a partially written validation set.
    records = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            record_id = record["id"]
            if record_id in seen:
                raise ValueError(f"Duplicate ID on line {line_number}: {record_id}")
            seen.add(record_id)
            rover = record["rover_v1"]
            status = rover["updated_text_status"]
            counts["input_records"] += 1
            counts[f"status_{status}"] += 1
            if status != "valid":
                excluded.append(
                    {
                        "id": record_id,
                        "source_line_1based": line_number,
                        "updated_text_status": status,
                        "invalid_reason": rover.get("invalid_reason"),
                        "review_status": rover.get("review_status"),
                    }
                )
                continue
            text = rover["updated_text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Valid record has empty updated_text: {record_id}")
            if rover.get("review_status") != "complete":
                raise ValueError(f"Valid record has incomplete review: {record_id}")
            if not record["audio_path"].startswith("am_data/"):
                raise ValueError(f"Unexpected audio path: {record['audio_path']}")
            counts["changed_references"] += text != record["display"]
            flat = {key: value for key, value in record.items() if key != "rover_v1"}
            flat.update(
                {
                    "updated_text": text,
                    "rover_updated_text_status": status,
                    "rover_review_status": rover["review_status"],
                    "source_line_1based": line_number,
                }
            )
            records.append(flat)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    excluded_path = output.with_suffix(".excluded.json")
    excluded_path.write_text(json.dumps(excluded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output": str(output.resolve()),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "selection": "rover_v1.updated_text_status == valid; nonempty updated_text; complete review",
        "reference": "updated_text = rover_v1.updated_text; display preserves the input",
        "output_records": len(records),
        "excluded_records": len(excluded),
        "excluded_manifest": str(excluded_path.resolve()),
        **dict(counts),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))

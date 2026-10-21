#!/usr/bin/env python3
"""Add audio-matched NE-tagged TSV transcriptions to an existing earnings JSONL."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def merge_transcriptions(source_jsonl, tagged_tsv, output_jsonl):
    source_jsonl, tagged_tsv, output_jsonl = map(Path, (source_jsonl, tagged_tsv, output_jsonl))
    with tagged_tsv.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"Audio", "Display_NE_Tags_Wrapped"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"TSV must contain columns {sorted(required)}")
        transcriptions = {}
        for line_number, row in enumerate(reader, start=2):
            audio = row["Audio"]
            text = row["Display_NE_Tags_Wrapped"]
            if not audio or not text or not text.strip():
                raise ValueError(f"Empty audio/transcription on TSV line {line_number}")
            if audio in transcriptions:
                raise ValueError(f"Duplicate TSV audio: {audio}")
            transcriptions[audio] = text

    with source_jsonl.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    audio_names = [Path(record["audio_path"]).name for record in records]
    duplicates = [audio for audio, count in Counter(audio_names).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate source audio: {duplicates[:10]}")
    missing = set(audio_names) - transcriptions.keys()
    unused = transcriptions.keys() - set(audio_names)
    if missing or unused:
        raise ValueError(
            f"Audio match is not one-to-one: {len(missing)} missing TSV rows, "
            f"{len(unused)} unused TSV rows; examples: {sorted(missing)[:5]}, {sorted(unused)[:5]}"
        )
    if any("display_ne_tagged" in record for record in records):
        raise ValueError("Source already contains display_ne_tagged")
    for record, audio in zip(records, audio_names, strict=True):
        record["display_ne_tagged"] = transcriptions[audio]

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "source_jsonl": str(source_jsonl),
        "tagged_tsv": str(tagged_tsv),
        "output_jsonl": str(output_jsonl),
        "records": len(records),
        "matched_tsv_rows": len(transcriptions),
        "added_field": "display_ne_tagged",
        "sha256": hashlib.sha256(output_jsonl.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--tagged-tsv", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(merge_transcriptions(args.source_jsonl, args.tagged_tsv, args.output_jsonl), indent=2))


if __name__ == "__main__":
    main()

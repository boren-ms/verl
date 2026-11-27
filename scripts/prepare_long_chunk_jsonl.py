#!/usr/bin/env python3
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
"""Build long-form ASR JSONL from human-caption info and indexed audio chunks."""

import argparse
import hashlib
import json
import math
import os
import struct
from collections import Counter
from pathlib import Path


def read_uint(stream):
    data = stream.read(4)
    if len(data) != 4:
        raise ValueError("Truncated chunk record header")
    return struct.unpack("<I", data)[0]


def info_records(path, count):
    with Path(path).open("rb") as stream:
        if stream.read(4).lower() != b"info" or read_uint(stream) != 1:
            raise ValueError(f"Unsupported info chunk header: {path}")
        for index in range(count):
            if read_uint(stream) != index:
                raise ValueError(f"Corrupted record index {index}: {path}")
            size = read_uint(stream)
            payload = stream.read(size)
            if len(payload) != size:
                raise ValueError(f"Truncated info record {index}: {path}")
            record = json.loads(payload)
            if not isinstance(record, dict):
                raise ValueError(f"Info record is not an object: {path}:{index}")
            yield index, record
        if stream.read(1):
            raise ValueError(f"Unexpected records after declared count: {path}")


def build(file_set, info_dir, audio_root, output, min_duration=0, max_duration=None, max_wer=None):
    index = json.loads(Path(file_set).read_text())
    entries = index["fileInfo"]
    names = [entry["name"] for entry in entries]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate chunk names in file_set")
    if any(Path(name).name != name or name in ("", ".", "..") for name in names):
        raise ValueError("Chunk names must be plain filenames")
    if not audio_root.startswith("az://"):
        raise ValueError("audio_root must be an Azure Blob path")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    counts = Counter()
    total_duration = 0.0
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as stream:
            for entry in entries:
                name, count = entry["name"], int(entry["count"])
                for record_index, info in info_records(Path(info_dir) / (name + ".info"), count):
                    counts["input_rows"] += 1
                    text = info.get("tn_transcription")
                    source = info.get("transcription_source")
                    if source != "display_human_caption_adjusted":
                        raise ValueError(f"Unexpected human-caption source {source!r}: {name}:{record_index}")
                    if not isinstance(text, str):
                        raise ValueError(f"Missing normalized human caption: {name}:{record_index}")
                    if not text.strip():
                        counts["empty_reference"] += 1
                        continue
                    duration = float(info["duration"])
                    if not math.isfinite(duration) or duration <= 0:
                        raise ValueError(f"Invalid audio duration: {name}:{record_index}")
                    if duration < min_duration or (max_duration is not None and duration > max_duration):
                        counts["duration_filtered"] += 1
                        continue
                    wer = info.get("normalized_WER")
                    if wer is not None:
                        wer = float(wer)
                        if not math.isfinite(wer) or wer < 0:
                            raise ValueError(f"Invalid normalized WER: {name}:{record_index}")
                    if max_wer is not None and (wer is None or wer > max_wer):
                        counts["wer_filtered"] += 1
                        continue
                    row = {
                        "id": f"{name}:{record_index}",
                        "audio_chunk": f"{audio_root.rstrip('/')}/{name}.audio:{count}:{record_index}",
                        "text": text,
                        "audio_length": duration,
                        "language": "English",
                        "keywords": [],
                        "chunk_name": name,
                        "chunk_count": count,
                        "chunk_index": record_index,
                        "source_filename": info.get("filename", ""),
                        "transcription_source": source,
                        "normalized_wer": wer,
                    }
                    data = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode()
                    stream.write(data)
                    digest.update(data)
                    counts["output_rows"] += 1
                    total_duration += duration
                print(f"Prepared {name}: {counts['output_rows']} rows", flush=True)
        expected = sum(int(entry["count"]) for entry in entries)
        if counts["input_rows"] != expected:
            raise ValueError("Output audit does not match the declared record count")
        if not counts["output_rows"]:
            raise ValueError("No usable human captions remain")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    summary = {
        "reference_field": "tn_transcription",
        "required_transcription_source": "display_human_caption_adjusted",
        "audio_reference_format": "az://.../chunk.audio:chunk_count:record_index",
        "audio_root": audio_root,
        "chunks": len(entries),
        "counts": dict(counts),
        "filters": {"min_duration": min_duration, "max_duration": max_duration, "max_wer": max_wer},
        "audio_hours": total_duration / 3600,
        "jsonl_bytes": output.stat().st_size,
        "jsonl_sha256": digest.hexdigest(),
        "file_set_sha256": hashlib.sha256(Path(file_set).read_bytes()).hexdigest(),
    }
    output.with_suffix(".preparation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-set", type=Path, required=True)
    parser.add_argument("--info-dir", type=Path, required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-duration", type=float, default=0)
    parser.add_argument("--max-duration", type=float)
    parser.add_argument("--max-wer", type=float)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.file_set,
                args.info_dir,
                args.audio_root,
                args.output,
                args.min_duration,
                args.max_duration,
                args.max_wer,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

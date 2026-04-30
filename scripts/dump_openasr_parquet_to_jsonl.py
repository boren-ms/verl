#!/usr/bin/env python3
"""Dump an OpenASR parquet file to AMI-style JSONL plus WAV audio files."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import PurePosixPath

import blobfile as bf
import pyarrow.parquet as pq
import soundfile as sf


def join_path(base: str, *parts: str) -> str:
    return "/".join([base.rstrip("/"), *(part.strip("/") for part in parts)])


def row_id(file_name: str | None, index: int, dataset: str) -> str:
    if file_name:
        return PurePosixPath(file_name).stem
    return f"{dataset}_{index}"


def read_audio(audio_field: dict | None) -> tuple[object, int]:
    if not audio_field or not audio_field.get("bytes"):
        raise ValueError("row does not contain audio.bytes")
    return sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32", always_2d=False)


def dump_parquet(
    input_path: str,
    output_dir: str,
    dataset: str,
    batch_size: int,
    audio_path_root: str | None = None,
    progress_interval: int = 5000,
) -> int:
    audio_path_root = audio_path_root or output_dir
    audio_dir = join_path(output_dir, "audio")
    if not bf.exists(audio_dir):
        bf.makedirs(audio_dir)

    parquet_source = bf.BlobFile(input_path, "rb") if "://" in input_path else input_path
    parquet_file = pq.ParquetFile(parquet_source)
    columns = ["file_name", "audio", "duration", "text", "source_lang", "target_lang"]

    count = 0
    last_report = 0
    jsonl_path = join_path(output_dir, "data.jsonl")
    with bf.BlobFile(jsonl_path, "w") as jsonl_file:
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                audio, sampling_rate = read_audio(row.get("audio"))
                duration = row.get("duration")
                if duration is None:
                    duration = len(audio) / sampling_rate

                output_audio_path = join_path(output_dir, "audio", f"{count}.wav")
                record_audio_path = join_path(audio_path_root, "audio", f"{count}.wav")
                with bf.BlobFile(output_audio_path, "wb") as audio_file:
                    sf.write(audio_file, audio, sampling_rate, format="WAV")

                record = {
                    "dataset": dataset,
                    "text": row.get("text", ""),
                    "id": row_id(row.get("file_name"), count, dataset),
                    "audio_length_s": duration,
                    "audio_path": record_audio_path,
                    "sampling_rate": sampling_rate,
                    "duration": duration,
                }
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

            if count - last_report >= progress_interval:
                print(f"converted {count} rows", flush=True)
                last_report = count

    if hasattr(parquet_source, "close"):
        parquet_source.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input OpenASR parquet path, local or az://")
    parser.add_argument("--output-dir", required=True, help="Output directory for data.jsonl and audio/")
    parser.add_argument("--dataset", required=True, help="Value to write into the JSONL dataset field")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--progress-interval", type=int, default=5000)
    parser.add_argument(
        "--audio-path-root",
        default=None,
        help="Root to store in JSONL audio_path values. Defaults to --output-dir.",
    )
    args = parser.parse_args()

    count = dump_parquet(
        args.input,
        args.output_dir,
        args.dataset,
        args.batch_size,
        args.audio_path_root,
        args.progress_interval,
    )
    print(json.dumps({"rows": count, "output_dir": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()

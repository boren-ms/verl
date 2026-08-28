#!/usr/bin/env python3
"""Batch dump OpenASR parquet files to JSONL/audio directories."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import blobfile as bf
import pyarrow.parquet as pq

from dump_openasr_parquet_to_jsonl import dump_parquet, join_path


def split_name(input_path: str, input_root: str) -> tuple[str, str, str]:
    relative = PurePosixPath(input_path.removeprefix(input_root.rstrip("/") + "/"))
    dataset = relative.parent.as_posix()
    split = relative.stem
    if split.endswith("_test"):
        split = split[: -len("_test")]
    dataset_name = "_".join([*relative.parent.parts, split])
    return dataset, split, dataset_name


def parquet_num_rows(path: str) -> int:
    with bf.BlobFile(path, "rb") as file_obj:
        return pq.ParquetFile(file_obj).metadata.num_rows


def count_jsonl(path: str) -> int:
    count = 0
    with bf.BlobFile(path, "r") as file_obj:
        for line in file_obj:
            if line.strip():
                count += 1
    return count


def output_complete(output_dir: str, expected_rows: int) -> bool:
    jsonl_path = join_path(output_dir, "data.jsonl")
    if not bf.exists(jsonl_path):
        return False
    if count_jsonl(jsonl_path) != expected_rows:
        return False
    if expected_rows == 0:
        return True
    return bf.exists(join_path(output_dir, "audio", "0.wav")) and bf.exists(
        join_path(output_dir, "audio", f"{expected_rows - 1}.wav")
    )


def sync_dir(local_dir: Path, output_dir: str) -> None:
    subprocess.run(
        ["bbb", "sync", str(local_dir) + "/", output_dir.rstrip("/") + "/"],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def convert_one(
    input_path: str,
    input_root: str,
    output_root: str,
    work_dir: Path,
    batch_size: int,
    overwrite: bool,
) -> dict[str, object]:
    dataset, split, dataset_name = split_name(input_path, input_root)
    output_dir = join_path(output_root, dataset, split)
    expected_rows = parquet_num_rows(input_path)

    if not overwrite and output_complete(output_dir, expected_rows):
        return {"status": "skipped", "input": input_path, "output_dir": output_dir, "rows": expected_rows}

    shard_work_dir = work_dir / dataset / split
    local_parquet = shard_work_dir / PurePosixPath(input_path).name
    local_output = shard_work_dir / "out"
    if shard_work_dir.exists():
        shutil.rmtree(shard_work_dir)
    local_output.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"status": "copying", "input": input_path, "local": str(local_parquet)}), flush=True)
    bf.copy(input_path, str(local_parquet), overwrite=True)

    print(json.dumps({"status": "converting", "input": input_path, "output_dir": output_dir}), flush=True)
    rows = dump_parquet(
        str(local_parquet),
        str(local_output),
        dataset_name,
        batch_size,
        audio_path_root=output_dir,
    )
    if rows != expected_rows:
        raise RuntimeError(f"Converted {rows} rows, expected {expected_rows}: {input_path}")

    print(json.dumps({"status": "syncing", "output_dir": output_dir}), flush=True)
    sync_dir(local_output, output_dir)
    shutil.rmtree(shard_work_dir)
    return {"status": "converted", "input": input_path, "output_dir": output_dir, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--work-dir", default="/tmp/openasr_ml_jsonl_batch")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_paths = sorted(bf.glob(join_path(args.input_root, "*", "*.parquet")))
    if args.limit is not None:
        input_paths = input_paths[: args.limit]
    print(json.dumps({"inputs": len(input_paths), "input_root": args.input_root}), flush=True)

    results = []
    for index, input_path in enumerate(input_paths, start=1):
        print(json.dumps({"status": "starting", "index": index, "total": len(input_paths), "input": input_path}), flush=True)
        result = convert_one(
            input_path=input_path,
            input_root=args.input_root,
            output_root=args.output_root,
            work_dir=Path(args.work_dir),
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
        results.append(result)
        print(json.dumps(result), flush=True)

    summary = {
        "converted": sum(1 for result in results if result["status"] == "converted"),
        "skipped": sum(1 for result in results if result["status"] == "skipped"),
        "rows": sum(int(result["rows"]) for result in results),
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

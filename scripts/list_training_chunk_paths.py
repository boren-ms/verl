#!/usr/bin/env python3
"""List unique .audio ChunkFiles and missing blobs from a recipe's JSONL data.

Example:
    python scripts/list_training_chunk_paths.py \
        --config recipe/phimm/config/v2609_entity/remax_2609r2_name_en13m_s1k_bs128_n4_r256_g32.yaml \
        --output-dir /root/data/name_en13m_chunk_paths

Use --split val to check validation data (default: train).
Writes sorted chunk_paths.txt, missing_chunk_paths.txt, and summary.json. Only
selected manifests are read; audio blobs are checked with HEAD, not downloaded.
Storage errors fail the command rather than being reported as missing files.
"""

import argparse
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import blobfile as bf
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def load_training_sources(config_path, split="train"):
    if split not in ("train", "val"):
        raise ValueError(f"Unsupported data split: {split}")
    config_path = Path(config_path).resolve()
    with initialize_config_dir(config_dir=str(config_path.parent), version_base=None):
        config = compose(config_name=config_path.stem)
    data_key = f"{split}_data"
    if config.data.get(data_key) is None:
        raise ValueError(f"Recipe must configure nonempty data.{data_key}")
    sources = OmegaConf.to_container(config.data[data_key], resolve=True)
    if isinstance(sources, dict):
        sources = [sources]
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"Recipe must configure nonempty data.{data_key}")
    for source in sources:
        if source.get("dataset_name") != "jsonl":
            raise ValueError("Only JSONL datasets are supported")
    return sources


def chunk_file_path(record, path_map):
    field = path_map.get("field", "audio_path")
    value = record.get(field) or record.get("audio_chunk")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing string audio reference in {field}/audio_chunk")
    if field in record and path_map.get("src_part") and path_map.get("dst_part"):
        value = value.replace(path_map["src_part"], path_map["dst_part"])
    parts = value.rsplit(":", 2)
    if len(parts) != 3 or not parts[0].endswith(".audio"):
        raise ValueError(f"Expected .audio ChunkFile:count:index reference, got {value!r}")
    chunk_path, count, index = parts
    count, index = int(count), int(index)
    if not 0 <= index < count:
        raise ValueError(f"Invalid chunk index {index} for count {count}: {value}")
    return chunk_path


def collect_chunk_paths(sources):
    paths = set()
    manifests = []
    rows = 0
    for source in sources:
        inputs = source["jsonl_paths"]
        inputs = [inputs] if isinstance(inputs, str) else inputs
        path_map = source.get("pre_process", {}).get("path_map", {})
        for input_path in inputs:
            print(f"Reading {input_path}", flush=True)
            manifest_rows = 0
            with bf.BlobFile(input_path, "rb") as raw:
                stream = gzip.GzipFile(fileobj=raw) if input_path.endswith(".gz") else raw
                try:
                    for line_number, line in enumerate(stream, 1):
                        if not line.strip():
                            continue
                        try:
                            path = chunk_file_path(json.loads(line), path_map)
                        except (ValueError, TypeError, AttributeError) as exc:
                            raise ValueError(f"{input_path}:{line_number}: {exc}") from exc
                        paths.add(path)
                        manifest_rows += 1
                        rows += 1
                        if rows % 1_000_000 == 0:
                            print(f"Read {rows:,} rows; {len(paths):,} unique chunks", flush=True)
                finally:
                    if stream is not raw:
                        stream.close()
            manifests.append({"path": input_path, "rows": manifest_rows})
    if not paths:
        raise ValueError("No ChunkFile references found in selected data")
    return sorted(paths), rows, manifests


def find_missing_paths(paths, workers):
    if workers < 1:
        raise ValueError("workers must be positive")
    missing = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Bound pending HEAD requests even for very large path lists.
        for start in range(0, len(paths), 1024):
            batch = paths[start : start + 1024]
            for path, exists in zip(batch, executor.map(bf.exists, batch), strict=True):
                if not exists:
                    missing.append(path)
            print(f"Checked {start + len(batch):,}/{len(paths):,}; missing {len(missing):,}", flush=True)
    return missing


def write_outputs(output_dir, paths, missing, summary):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("chunk_paths.txt", paths), ("missing_chunk_paths.txt", missing)):
        destination = output_dir / name
        temporary = destination.with_suffix(".txt.tmp")
        temporary.write_text("".join(f"{path}\n" for path in values), encoding="utf-8")
        temporary.replace(destination)
    destination = output_dir / "summary.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    sources = load_training_sources(args.config, args.split)
    paths, rows, manifests = collect_chunk_paths(sources)
    missing = find_missing_paths(paths, args.workers)
    summary = {
        "config": str(args.config),
        "split": args.split,
        "manifests": manifests,
        "training_rows" if args.split == "train" else "validation_rows": rows,
        "unique_chunk_paths": len(paths),
        "duplicate_references_removed": rows - len(paths),
        "existing_chunk_paths": len(paths) - len(missing),
        "missing_chunk_paths": len(missing),
    }
    write_outputs(args.output_dir, paths, missing, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

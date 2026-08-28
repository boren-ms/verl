"""Print training-shaped samples from each dataset entry in a YAML config."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Iterator

import blobfile as bf
import yaml
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).parents[1]))

from recipe.phimm.data.dataset import augment


def expand_paths(paths: str | list[str]) -> list[str]:
    expanded = []
    for path in [paths] if isinstance(paths, str) else paths:
        matches = sorted(bf.glob(path)) if any(char in path for char in "*?[") else [path]
        for match in matches:
            if match not in expanded:
                expanded.append(match)
    return expanded


def iter_jsonl(path: str) -> Iterator[dict[str, Any]]:
    with bf.BlobFile(path, "r") as file_obj:
        for line in file_obj:
            if line.strip():
                yield json.loads(line)


def sample_paths(paths: list[str], count: int) -> list[dict[str, Any]]:
    iterators = [(path, iter_jsonl(path)) for path in paths]
    samples = []
    while iterators and len(samples) < count:
        remaining = []
        for path, records in iterators:
            try:
                record = next(records)
            except StopIteration:
                continue
            record["audio_sample_source_path"] = path
            samples.append(record)
            remaining.append((path, records))
            if len(samples) == count:
                break
        iterators = remaining
    return samples


def format_samples(config: dict[str, Any], samples: list[dict[str, Any]], model_version: int) -> Dataset:
    transform_config = copy.deepcopy(config)
    transform_config["model_version"] = model_version
    transform_config.pop("dataset_name", None)
    transform_config.pop("jsonl_paths", None)
    dataset = augment(Dataset.from_list(samples), **transform_config)
    return dataset.rename_column("audio_sample_source_path", "sample_source_path")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--model-version", type=int, default=2607)
    args = parser.parse_args()

    configs = yaml.safe_load(args.config.read_text())
    configs = configs if isinstance(configs, list) else [configs]
    for index, config in enumerate(configs):
        paths = expand_paths(config["jsonl_paths"])
        samples = sample_paths(paths, args.samples)
        dataset = format_samples(config, samples, args.model_version)
        source = dataset[0].get("data_source", f"dataset_{index}") if len(dataset) else f"dataset_{index}"
        print(f"\n### {index}: {source} ({len(paths)} unique source path(s))")
        for sample_index, sample in enumerate(dataset):
            print(json.dumps({"sample": sample_index, **sample}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
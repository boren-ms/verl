"""Cache processed phimm data configs as JSONL or parquet.

Usage:
    python -m recipe.phimm.cache_dataset \
        --config-name gen_ls_raw_rp_edge_nodigits

Each Hydra config points to a source data YAML and a destination .jsonl or .parquet path.
"""

from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import blobfile as bf
import hydra
import yaml
from datasets import Dataset, concatenate_datasets
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from recipe.phimm.data.dataset import create_datasets
from recipe.phimm.utils.audio import _is_time_chunk_spec, load_raw_audio, sf_write
from recipe.phimm.utils.shared import save_dataset


PHIMM_CONFIG_DIR = Path(__file__).parent / "config"
SourceConfig = str | dict[str, Any] | list[Any]


def _load_yaml(path: str) -> Any:
    path = _resolve_path(path, config_relative=True)
    with bf.BlobFile(path, "r") as file_obj:
        return yaml.safe_load(file_obj)


def _load_source_config(source_config: SourceConfig) -> Any:
    if isinstance(source_config, str):
        return _load_yaml(source_config)
    if isinstance(source_config, (dict, list)):
        return copy.deepcopy(source_config)
    raise TypeError(f"Unsupported source_config type: {type(source_config).__name__}")


def _resolve_path(path: str, config_relative: bool = False) -> str:
    parsed = urlparse(path)
    expanded = Path(path).expanduser()
    if parsed.scheme or expanded.is_absolute():
        return path

    absolute_path = Path(to_absolute_path(path))
    if absolute_path.exists() or not config_relative:
        return str(absolute_path)

    config_path = PHIMM_CONFIG_DIR / path
    return str(config_path)


def _strip_verl_format(config: Any) -> Any:
    if isinstance(config, list):
        return [_strip_verl_format(item) for item in config]
    if not isinstance(config, dict):
        return config
    config = copy.deepcopy(config)
    post_process = config.get("post_process")
    if isinstance(post_process, dict):
        post_process.pop("verl_format", None)
    return config


def _as_dataset(dataset: Dataset | dict[str, Dataset]) -> Dataset:
    if isinstance(dataset, Dataset):
        return dataset
    if isinstance(dataset, dict):
        if not dataset:
            raise ValueError("No datasets were created from the config.")
        return concatenate_datasets(list(dataset.values()))
    raise TypeError(f"Unsupported dataset type: {type(dataset).__name__}")


def materialize_audio_segments(
    dataset: Dataset,
    output_dir: str,
    overwrite: bool = False,
    max_workers: int = 16,
) -> Dataset:
    """Write time-range audio specs as concrete WAV files."""
    output_dir = output_dir.rstrip("/")

    def materialize(example: dict[str, Any]) -> str:
        source_field = next(
            (field for field in ("audio_path", "audio_file") if example.get(field)),
            None,
        )
        if source_field is None:
            return ""
        source = str(example[source_field])
        source_path = source.rpartition("#")[0] if _is_time_chunk_spec(source) else source
        source_stem = Path(urlparse(source_path).path).stem
        segment_id = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        output_path = f"{output_dir}/{source_stem}-{segment_id}.wav"
        if overwrite or not bf.exists(output_path):
            audio, sample_rate = load_raw_audio(example)
            sf_write(output_path, audio, sample_rate)
        return output_path

    def materialize_batch(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
        source_field = "audio_path" if "audio_path" in batch else "audio_file"
        examples = [dict(zip(batch, values, strict=True)) for values in zip(*batch.values(), strict=True)]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            output_paths = list(executor.map(materialize, examples))
        return {source_field: output_paths}

    return dataset.map(
        materialize_batch,
        batched=True,
        batch_size=max_workers,
        desc="Materializing audio segments",
    )


def cache_summary(
    source_config: SourceConfig,
    output_path: str,
    overwrite: bool,
    skipped: bool,
    reason: str | None = None,
    dataset: Dataset | None = None,
) -> dict[str, Any]:
    summary = {
        "source_config": source_config,
        "output_path": output_path,
        "overwrite": overwrite,
        "skipped": skipped,
    }
    if reason is not None:
        summary["reason"] = reason
    if dataset is not None:
        summary["num_rows"] = len(dataset)
        summary["columns"] = dataset.column_names
    print(json.dumps(summary, indent=2))
    return summary


def cache_dataset(
    source_config: SourceConfig,
    output_path: str,
    include_verl_format: bool = False,
    overwrite: bool = False,
    part_size: int | None = None,
    ext: str | None = None,
    audio_output_dir: str | None = None,
    audio_overwrite: bool = False,
) -> dict[str, Any]:
    output_path = _resolve_path(output_path)
    if bf.exists(output_path) and not overwrite:
        print(f"Output already exists, skipping: {output_path}")
        return cache_summary(source_config, output_path, overwrite, skipped=True, reason="output_exists")

    dataset_config = _load_source_config(source_config)
    if not include_verl_format:
        dataset_config = _strip_verl_format(dataset_config)

    print(f"Loading source config: {source_config}")
    if not include_verl_format:
        print("Skipping post_process.verl_format before caching.")
    dataset = _as_dataset(create_datasets(dataset_config))
    if audio_output_dir:
        dataset = materialize_audio_segments(
            dataset,
            _resolve_path(audio_output_dir),
            overwrite=audio_overwrite,
        )

    print(f"Writing to: {output_path}")
    save_dataset(dataset, output_path, overwrite=overwrite, part_size=part_size, ext=ext)
    return cache_summary(source_config, output_path, overwrite, skipped=False, dataset=dataset)


@hydra.main(config_path="config/data/cache", config_name="gen_ls_raw_rp_edge_nodigits", version_base=None)
def main(config: DictConfig) -> None:
    cfg = OmegaConf.to_container(config, resolve=True)
    part_size = cfg.get("part_size")
    cache_dataset(
        source_config=cfg["source_config"],
        output_path=str(cfg["output_path"]),
        include_verl_format=bool(cfg.get("include_verl_format", False)),
        overwrite=bool(cfg.get("overwrite", False)),
        part_size=int(part_size) if part_size is not None else None,
        ext=cfg.get("ext"),
        audio_output_dir=cfg.get("audio_output_dir"),
        audio_overwrite=bool(cfg.get("audio_overwrite", False)),
    )


if __name__ == "__main__":
    main()

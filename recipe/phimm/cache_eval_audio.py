"""Prefetch audio referenced by a JSONL evaluation dataset YAML.

Usage:
    python -m recipe.phimm.cache_eval_audio \
        recipe/phimm/config/data/val_data/earnings_aa_chunked_langhint.yaml --workers 16

Uses the evaluator's persistent Orange cache under ~/data. No YAML or JSONL
rewriting is needed; run on each evaluation node with the evaluation user's HOME.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import blobfile as bf
import yaml

from verl.audio_cache import (
    LOCAL_DATA_ROOT,
    ORANGE_DATA_PREFIX,
    _split_audio_source,
    cache_audio_source,
    local_audio_source,
)

logger = logging.getLogger(__name__)
AUDIO_FIELDS = ("audio_path", "audio_file", "audio_chunk", "url")


def _manifest_paths(config_path: str) -> list[str]:
    with bf.BlobFile(os.path.expanduser(config_path), "r") as stream:
        config = yaml.safe_load(stream)
    entries = config if isinstance(config, list) else [config]
    if not entries:
        raise ValueError("Dataset YAML contains no datasets.")
    manifests = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("dataset_name", "").lower() != "jsonl":
            raise ValueError(f"Dataset entry {index} must have dataset_name: jsonl.")
        paths = entry.get("jsonl_paths")
        paths = [paths] if isinstance(paths, str) else paths
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"Dataset entry {index} must specify nonempty jsonl_paths.")
        for path in paths:
            if not isinstance(path, str) or not path:
                raise ValueError(f"Invalid jsonl_paths value in dataset entry {index}: {path!r}")
            path = os.path.expanduser(path)
            if any(char in path for char in "*?["):
                matches = sorted(bf.glob(path))
            elif bf.isdir(path):
                matches = sorted([*bf.glob(bf.join(path, "*.jsonl")), *bf.glob(bf.join(path, "*.jsonl.gz"))])
            elif bf.exists(path):
                matches = [path]
            else:
                matches = []
            if not matches:
                raise FileNotFoundError(f"No JSONL files matched: {path}")
            manifests.update(dict.fromkeys(matches))
    return list(manifests)


def collect_audio_files(config_path: str, fields: tuple[str, ...] = AUDIO_FIELDS) -> list[str]:
    """Read each manifest once and deduplicate physical files, not segment selectors."""
    sources = {}
    for manifest in _manifest_paths(config_path):
        logger.info("Reading %s", manifest)
        with bf.BlobFile(manifest, "rb") as raw:
            stream = gzip.GzipFile(fileobj=raw) if manifest.endswith(".gz") else raw
            try:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeDecodeError) as exc:
                        raise ValueError(f"Invalid JSON at {manifest}:{line_number}") from exc
                    if not isinstance(row, dict):
                        raise ValueError(f"Expected an object at {manifest}:{line_number}")
                    found = False
                    for field in fields:
                        source = row.get(field)
                        if source is None or source == "":
                            continue
                        if not isinstance(source, str):
                            raise ValueError(f"Expected string field {field!r} at {manifest}:{line_number}")
                        found = True
                        physical_file, _ = _split_audio_source(source)
                        physical_file = os.path.expanduser(physical_file)
                        if "://" in physical_file and not physical_file.startswith(ORANGE_DATA_PREFIX):
                            raise ValueError(f"Unsupported remote audio source at {manifest}:{line_number}: {source}")
                        if "://" not in physical_file and not Path(physical_file).is_file():
                            raise FileNotFoundError(f"Missing local audio at {manifest}:{line_number}: {physical_file}")
                        sources[physical_file] = None
                    if not found:
                        raise ValueError(f"No audio reference in fields {fields} at {manifest}:{line_number}")
            finally:
                if stream is not raw:
                    stream.close()
    if not sources:
        raise ValueError("Dataset manifests contain no audio references.")
    return list(sources)


def cache_eval_audio(
    config_path: str,
    workers: int = 16,
    fields: tuple[str, ...] = AUDIO_FIELDS,
) -> dict[str, int | str]:
    """Wait for all downloads; reuse existing files and fail explicitly on errors."""
    if workers < 1:
        raise ValueError("workers must be positive.")
    if not fields or any(not field for field in fields):
        raise ValueError("At least one nonempty audio field is required.")
    sources = collect_audio_files(config_path, fields)
    pending = [
        source
        for source in sources
        if source.startswith(ORANGE_DATA_PREFIX) and not Path(local_audio_source(source)).is_file()
    ]
    summary: dict[str, int | str] = {
        "cache_root": str(LOCAL_DATA_ROOT),
        "unique_files": len(sources),
        "already_local": len(sources) - len(pending),
        "downloaded": 0,
    }
    logger.info("%d unique audio files; %d require downloading", len(sources), len(pending))
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(cache_audio_source, source): source for source in pending}
        for completed, future in enumerate(as_completed(futures), 1):
            source = futures[future]
            try:
                localized = future.result()
            except (OSError, RuntimeError) as exc:
                failures += 1
                logger.error("Failed to cache %s: %s", source, exc)
            else:
                if not Path(localized).is_file():
                    failures += 1
                    logger.error("Cache transfer did not produce a local file for %s", source)
            if completed % 100 == 0 or completed == len(pending):
                logger.info("Completed %d/%d downloads (%d failures)", completed, len(pending), failures)
    if failures:
        raise RuntimeError(f"Failed to cache {failures}/{len(pending)} audio files; rerun to retry missing files.")
    summary["downloaded"] = len(pending)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_yaml", help="Evaluation dataset YAML with dataset_name: jsonl entries.")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent downloads (default: 16).")
    parser.add_argument("--audio-fields", nargs="+", default=list(AUDIO_FIELDS), help="JSONL audio fields to cache.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    summary = cache_eval_audio(args.dataset_yaml, workers=args.workers, fields=tuple(args.audio_fields))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

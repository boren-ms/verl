"""Prepare a separate model directory excluding explicitly named archive folders.

Never edits the source. This verifies file completeness and hashes, not model
loading. Validate the resulting tokenizer/processor/model before publishing it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from filelock import FileLock


def prepare_model_directory(source, destination, exclude_dirs):
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    if not source.is_dir() or not (source / "config.json").is_file():
        raise ValueError("Source must be a model directory containing config.json")
    excluded = set(exclude_dirs)
    if not excluded or any(x in ("", ".", "..") or "/" in x or "\\" in x for x in excluded):
        raise ValueError("Provide explicit top-level directory names to exclude")
    if any(not (source / name).is_dir() for name in excluded):
        raise ValueError("Each excluded name must be an existing source directory")
    if destination.resolve().is_relative_to(source):
        raise ValueError("Destination must be outside the source directory")

    files = []
    for folder, dirs, names in os.walk(source, followlinks=False):
        if Path(folder) == source:
            dirs[:] = [d for d in dirs if d not in excluded]
        for name in dirs + names:
            if (Path(folder) / name).is_symlink():
                raise ValueError("Symlinks require manual review before preparing a model")
        for name in names:
            path = Path(folder) / name
            if not path.is_file():
                raise ValueError(f"Unsupported source entry: {path}")
            files.append(path.relative_to(source).as_posix())
    selected = set(files)
    if "verl-model-manifest.json" in selected:
        raise ValueError("Source already contains a preparation manifest; review it before repacking")
    # Preserve every indexed weight shard, including nonstandard shard names.
    for name in files:
        if name.endswith(".index.json"):
            data = json.loads((source / name).read_text())
            for shard in data.get("weight_map", {}).values():
                rel = (PurePosixPath(name).parent / shard).as_posix()
                if rel not in selected:
                    raise ValueError(f"Required weight shard would be missing: {rel}")
    # Reject explicit JSON references into an excluded tree (e.g. custom assets).
    def references(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from references(item)
        elif isinstance(value, list):
            for item in value:
                yield from references(item)

    for name in files:
        if name.endswith(".json"):
            for value in references(json.loads((source / name).read_text())):
                clean = value.removeprefix("./")
                if any(clean == x or clean.startswith(x + "/") for x in excluded):
                    raise ValueError(f"JSON references excluded directory: {name}: {value}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".prepare.lock", timeout=60):
        if destination.exists():
            raise FileExistsError(destination)
        staging = Path(tempfile.mkdtemp(prefix=destination.name + ".staging-", dir=destination.parent))
        try:
            manifest = {"schema": 1, "source": str(source), "excluded_dirs": sorted(excluded), "files": []}
            for name in sorted(files):
                src, dest = source / name, staging / name
                before = src.stat()
                dest.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with src.open("rb") as reader, dest.open("wb") as writer:
                    while chunk := reader.read(4 * 1024 * 1024):
                        digest.update(chunk)
                        writer.write(chunk)
                after = src.stat()
                if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_ino, after.st_size, after.st_mtime_ns
                ):
                    raise RuntimeError(f"Source changed while copying: {name}")
                if dest.stat().st_size != before.st_size:
                    raise RuntimeError(f"Incomplete copy: {name}")
                manifest["files"].append({"path": name, "bytes": before.st_size, "sha256": digest.hexdigest()})
            manifest["total_bytes"] = sum(f["bytes"] for f in manifest["files"])
            manifest["content_sha256"] = hashlib.sha256(
                json.dumps(manifest["files"], sort_keys=True).encode()
            ).hexdigest()
            (staging / "verl-model-manifest.json").write_text(json.dumps(manifest, indent=2))
            os.rename(staging, destination)
            return manifest
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--exclude-dir", action="append", required=True)
    args = parser.parse_args()
    result = prepare_model_directory(args.source, args.destination, args.exclude_dir)
    print(json.dumps({k: result[k] for k in ("total_bytes", "content_sha256", "excluded_dirs")}, indent=2))


if __name__ == "__main__":
    main()

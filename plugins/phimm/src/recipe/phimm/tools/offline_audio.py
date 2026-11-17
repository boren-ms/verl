"""Materialize random segments or ordered merges before task/prompt processing.

Compose an experiment preset, apply its offline_data plan to raw normalized
training rows and write JSONL, a datasource YAML and a ready-to-run training
snapshot. Audio bytes and validation sources are never rewritten.
"""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from recipe.phimm.data.dataset import _extra_info_value, jsonl_dataset, limit_ds, process_ds
from recipe.phimm.tools.random_merge_list import random_merge
from recipe.phimm.tools.random_segment import random_segment
from recipe.phimm.utils.offline_data import active_audio_transforms


def _sources(sources):
    for source in sources:
        if isinstance(source, list):
            yield from _sources(source)
        else:
            yield source


def materialize(config, output_dir, *, blob_root=None, input_paths=None, duration_cache=None):
    """Return a training snapshot which reads only the emitted, fixed manifests."""
    transforms = active_audio_transforms(config)
    if len(transforms) != 1:
        raise ValueError("Select exactly one enabled offline_data.random_segment or offline_data.random_merge")
    mode, options = next(iter(transforms.items()))
    options = OmegaConf.to_container(OmegaConf.create(options), resolve=True)
    sources = list(_sources(OmegaConf.to_container(config.data.train_data, resolve=True)))
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    output_sources, audits = [], []
    for index, source in enumerate(sources):
        source = deepcopy(source)
        if source.get("dataset_name") != "jsonl":
            raise ValueError("Offline audio preparation supports JSONL sources only")
        paths = source["jsonl_paths"]
        if input_paths:
            paths = [input_paths.get(path, path) for path in ([paths] if isinstance(paths, str) else paths)]
        ds = jsonl_dataset(paths)
        input_count = len(ds)
        post = source.get("post_process") or {}
        limit = post.pop("input_egs_limit", None)
        if mode == "random_segment" and limit:
            ds = limit_ds(ds, egs_limit=limit)
        pre = source.pop("pre_process", {}) or {}
        if set(pre) - {"path_map", "rename_fields", "clean_tagged_text", "input_egs_limit"}:
            raise ValueError("Only raw path/text normalization is supported before offline audio preparation")
        if mode == "random_merge" and "clean_tagged_text" in pre:
            clean_options = dict(pre["clean_tagged_text"] or {})
            clean_options.setdefault("preserve_existing_keywords", True)
            pre["clean_tagged_text"] = clean_options
        ds = process_ds(ds, **pre)
        prepared_count = len(ds)
        normalized_hash = hashlib.sha256()
        for row in ds:
            normalized_hash.update((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode())
        kwargs = dict(options)
        if mode == "random_merge":
            kwargs.setdefault("default_language", (source.get("add_task_info") or {}).get("language"))
            ds = random_merge(ds, **kwargs)
            # The former post-process limit counted merged parents, not inputs.
            if limit:
                ds = limit_ds(ds, egs_limit=limit)
        else:
            # Online segmentation used to follow verl_format, where missing IDs
            # were derived from the scalar source. Freeze that fallback before
            # replacing the scalar path with its segment list.
            ds = ds.map(lambda row: {"id": _extra_info_value(row, "id")})
            ds = random_segment(ds, duration_cache=duration_cache, **kwargs)
        target = root / f"source_{index:03d}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for row in ds:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        # Verify the same Arrow loader used by training can read the entire output.
        loaded = jsonl_dataset(str(target))
        if len(loaded) != len(ds) or any(original != restored for original, restored in zip(ds, loaded, strict=True)):
            raise AssertionError("JSONL roundtrip changed the prepared rows")
        source["jsonl_paths"] = f"{blob_root.rstrip('/')}/{target.name}" if blob_root else str(target)
        source["post_process"] = post
        output_sources.append(source)
        with target.open("rb") as handle:
            manifest_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        audits.append(
            {
                "source": index,
                "input_jsonl_paths": sources[index]["jsonl_paths"],
                "input_rows": input_count,
                "prepared_rows": prepared_count,
                "output_rows": len(ds),
                "normalized_input_sha256": normalized_hash.hexdigest(),
                "manifest_sha256": manifest_hash,
            }
        )
    (root / "train_data.yaml").write_text(yaml.safe_dump(output_sources, sort_keys=False, allow_unicode=True))
    snapshot = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    with open_dict(snapshot):
        snapshot.pop("offline_data", None)
        snapshot.data.train_data = output_sources
    OmegaConf.save(snapshot, root / "training.yaml")
    (root / "preparation.json").write_text(
        json.dumps({"mode": mode, "options": options, "sources": audits}, indent=2, ensure_ascii=False)
    )
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blob-root", help="URI for manifests uploaded separately; this tool only writes local files")
    parser.add_argument("--input-map", type=Path, help="JSON mapping source manifest URIs to local cached files")
    parser.add_argument("--duration-cache", type=Path, help="JSON mapping normalized audio sources to duration seconds")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, including offline_data transform settings")
    args = parser.parse_args()
    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)
    with initialize_config_dir(config_dir=str(args.config_dir.resolve()), version_base=None):
        config = compose(config_name=args.config_name, overrides=args.overrides)
    materialize(
        config,
        args.output_dir,
        blob_root=args.blob_root,
        input_paths=json.loads(args.input_map.read_text()) if args.input_map else None,
        duration_cache=json.loads(args.duration_cache.read_text()) if args.duration_cache else None,
    )
    print(f"Prepared training config: {args.output_dir.resolve() / 'training.yaml'}")


if __name__ == "__main__":
    main()

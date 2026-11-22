"""Freeze preceding segment labels in existing ordered-merge JSONL manifests.

No audio, parent ordering, references, grouping, or source metadata is changed.
Only training data is rewritten; validation keeps generated history.
"""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import blobfile as bf
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from recipe.phimm.agentic.context_prompt import context_options
from recipe.phimm.tools.offline_audio import _sources

CONTEXT_FIELD = "segment_reference_contexts"


def reference_contexts(row):
    """Recover exact preceding labels from audited per-segment provenance.

    Empty FL labels remain empty. Parent text cannot be divided heuristically:
    source_metadata_json must match the ordered audio list exactly.
    """
    paths = row.get("audio_path")
    metadata = row.get("source_metadata_json")
    if (
        not isinstance(paths, list)
        or not paths
        or any(not isinstance(path, str) or not path for path in paths)
        or not isinstance(metadata, list)
        or len(metadata) != len(paths)
    ):
        raise ValueError("Reference context requires aligned audio_path and source_metadata_json lists")
    if CONTEXT_FIELD in row:
        raise ValueError("Reference contexts are already materialized")
    references = []
    for path, value in zip(paths, metadata, strict=True):
        original = json.loads(value)
        if (
            not isinstance(original, dict)
            or original.get("audio_path") != path
            or not isinstance(original.get("text"), str)
        ):
            raise ValueError("Reference context provenance must match each segment path and string text")
        references.append(original["text"])
    if row.get("text") != " ".join(references):
        raise ValueError("Parent reference does not match the ordered segment references")
    return [references[:index] for index in range(len(references))]


def materialize(config, output_dir, *, input_paths=None, blob_root=None):
    """Emit new manifests, an audit and a standalone Hydra training snapshot."""
    options = context_options(config.asr)
    if not options.get("enable", False) or options.get("training_source") != "reference":
        raise ValueError("Select asr.context_aware.enable=true and training_source=reference")
    sources = list(_sources(OmegaConf.to_container(config.data.train_data, resolve=True)))
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    output_sources, audits = [], []
    for index, source in enumerate(sources):
        source = deepcopy(source)
        if source.get("dataset_name") != "jsonl" or source.get("pre_process"):
            raise ValueError("Reference context preparation requires already normalized JSONL merge manifests")
        paths = source["jsonl_paths"]
        paths = [paths] if isinstance(paths, str) else paths
        target = root / f"source_{index:03d}.jsonl"
        inputs, count, segments, empty_histories = [], 0, 0, 0
        with target.open("w", encoding="utf-8") as output:
            for uri in paths:
                path = (input_paths or {}).get(uri, uri)
                digest, input_count = hashlib.sha256(), 0
                with bf.BlobFile(path, "rb") as handle:
                    for line in handle:
                        digest.update(line)
                        row = json.loads(line)
                        try:
                            histories = reference_contexts(row)
                        except (ValueError, TypeError) as error:
                            raise ValueError(f"{path}:{input_count + 1}: {error}") from error
                        row[CONTEXT_FIELD] = histories
                        output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        input_count += 1
                        segments += len(histories)
                        empty_histories += sum(not any(history) for history in histories)
                count += input_count
                inputs.append({"uri": uri, "read_path": str(path), "rows": input_count, "sha256": digest.hexdigest()})
        post = source.setdefault("post_process", {})
        extra_keys = post.setdefault("verl_format", {}).setdefault("extra_keys", [])
        if CONTEXT_FIELD not in extra_keys:
            extra_keys.append(CONTEXT_FIELD)
        source["jsonl_paths"] = f"{blob_root.rstrip('/')}/{target.name}" if blob_root else str(target)
        output_sources.append(source)
        with target.open("rb") as handle:
            manifest_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        audits.append(
            {
                "inputs": inputs,
                "manifest": str(target),
                "manifest_sha256": manifest_hash,
                "parents": count,
                "segments": segments,
                "empty_histories": empty_histories,
            }
        )
    snapshot = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    with open_dict(snapshot):
        snapshot.data.train_data = output_sources
    (root / "train_data.yaml").write_text(yaml.safe_dump(output_sources, sort_keys=False, allow_unicode=True))
    OmegaConf.save(snapshot, root / "training.yaml")
    (root / "preparation.json").write_text(
        json.dumps(
            {"context_source": "preceding_reference", "validation_source": "generated", "sources": audits},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-map", type=Path, help="JSON mapping source URIs to local cached manifests")
    parser.add_argument("--blob-root", help="Manifest upload destination; uploading is a separate operation")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)
    with initialize_config_dir(config_dir=str(args.config_dir.resolve()), version_base=None):
        config = compose(config_name=args.config_name, overrides=args.overrides)
    materialize(
        config,
        args.output_dir,
        input_paths=json.loads(args.input_map.read_text()) if args.input_map else None,
        blob_root=args.blob_root,
    )
    print(f"Prepared training config: {args.output_dir.resolve() / 'training.yaml'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check the published CV11 YAML with the actual training loader on each shard."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import blobfile as bf
import yaml


def run(args):
    root = Path(args.work_dir)
    info = json.loads((root / "provenance.json").read_text())
    markers = sorted((root / "shards").glob("*.uploaded.json"))
    if len(markers) != len(info["archives"]):
        raise ValueError("Training-loader audit requires every shard to be uploaded")
    remote = info["config"]["output_dir"]
    with bf.BlobFile(remote + "/training.yaml", "rb") as stream:
        yaml_bytes = stream.read()
    config = yaml.safe_load(yaml_bytes)
    if config["jsonl_paths"] != remote + "/data.jsonl":
        raise ValueError("Training YAML does not reference the published manifest")
    sample_path = root / "loader_samples.jsonl"
    original_rows = []
    with sample_path.open("w") as output:
        for marker in markers:
            name = marker.name.removesuffix(".uploaded.json")
            checkpoint = json.loads(marker.read_text())
            indices = set(random.Random(20261106 + int(name.rsplit("_", 1)[-1])).sample(
                range(checkpoint["rows"]), min(args.samples_per_shard, checkpoint["rows"])))
            with (root / "shards" / f"{name}.jsonl").open() as stream:
                for index, line in enumerate(stream):
                    if index in indices:
                        output.write(line)
                        original_rows.append(json.loads(line))
    sys.path.insert(0, str(Path(args.loader_repo).resolve()))
    from recipe.phimm.data.dataset import create_datasets
    from recipe.phimm.data.prompts import get_task_output

    config["jsonl_paths"] = str(sample_path)
    config["num_proc"] = 1
    dataset = create_datasets(config)
    if len(dataset) != len(original_rows):
        raise ValueError("Training loader changed sample count")
    for source, loaded in zip(original_rows, dataset):
        if loaded["audio_path"] != source["audio_path"]:
            raise ValueError("Training loader changed audio path")
        if loaded["data_source"] != "noisy_cv11_en_train":
            raise ValueError("Training reward source mismatch")
        if loaded["reward_model"]["ground_truth"] != source["text"]:
            raise ValueError("Training loader changed transcript")
        expected = get_task_output(task="lang_asr_verb", lang="English", text=source["text"])
        if loaded["reward_model"]["gt_output"] != expected:
            raise ValueError("Training reference formatting mismatch")
        if loaded["extra_info"]["id"] != source["id"] or loaded["extra_info"]["language"] != "English":
            raise ValueError("Training identity/language mismatch")
        if loaded["extra_info"]["prefix"] != "":
            raise ValueError("Unexpected training prefix")
        if not loaded["prompt"] or loaded["prompt"][0]["role"] != "user" or "<audio>" not in loaded["prompt"][0]["content"]:
            raise ValueError("Invalid training audio prompt")
    result = {
        "shards": len(markers), "loader_samples": len(dataset),
        "training_yaml_sha256": hashlib.sha256(yaml_bytes).hexdigest(),
        "loader_repo_revision": subprocess.check_output(
            ["git", "-C", args.loader_repo, "rev-parse", "HEAD"], text=True).strip(),
        "build_fingerprint": original_rows[0]["build_fingerprint"],
        "audio_paths_transcripts_prompts_and_reward_references_verified": True,
        "data_source": "noisy_cv11_en_train",
    }
    path = root / "artifacts/validation_loader.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    bf.copy(str(path), remote + "/validation_loader.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--loader-repo", required=True)
    parser.add_argument("--samples-per-shard", type=int, default=8)
    run(parser.parse_args())

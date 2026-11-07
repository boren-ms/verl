#!/usr/bin/env python3
"""Audit retained CV11 samples through the training loader, scorer and WAV cache."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import blobfile as bf
import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.filter_cv11_augmented_wer import MODEL_PATH, data_source, keep_row


def run(args):
    root = Path(args.work_dir)
    summary = json.loads((root / "summary.json").read_text())
    mode = summary.get("filter_mode", "wer_ge1pct")
    expected_source = data_source(mode)
    count = summary["filtered"]["count"]
    indices = set(np.linspace(0, count - 1, min(count, args.samples), dtype=int).tolist())
    selected = []
    sample_path = root / "loader_samples.jsonl"
    with (root / "data.jsonl").open() as stream, sample_path.open("w") as output:
        for index, line in enumerate(stream):
            if index in indices:
                output.write(line)
                selected.append(json.loads(line))
    if len(selected) != len(indices):
        raise ValueError("Filtered manifest does not match its row-count summary")
    destination = args.output_dir.rstrip("/")
    with bf.BlobFile(destination + "/training.yaml", "rb") as stream:
        yaml_bytes = stream.read()
    config = yaml.safe_load(yaml_bytes)
    if config["jsonl_paths"] != destination + "/data.jsonl":
        raise ValueError("Published training YAML references a different manifest")
    config["jsonl_paths"] = str(sample_path)
    config["num_proc"] = 1
    from recipe.phimm.data.dataset import create_datasets
    from recipe.phimm.data.prompts import get_task_output
    from recipe.phimm.reward.asr_eval import openasr_en_eval
    from verl.audio_cache import localize_audio_source

    dataset = create_datasets(config)
    if len(dataset) != len(selected):
        raise ValueError("Training loader changed sample count")
    for original, loaded in zip(selected, dataset):
        if not keep_row(original, mode) or original["filter_model_path"] != MODEL_PATH:
            raise ValueError("Wrong retained threshold or baseline checkpoint")
        if loaded["audio_path"] != original["audio_path"] or loaded["data_source"] != expected_source:
            raise ValueError("Training loader changed audio pointer or reward source")
        if loaded["reward_model"]["ground_truth"] != original["text"]:
            raise ValueError("Training loader changed original transcript")
        expected = get_task_output(task="lang_asr_verb", lang="English", text=original["text"], version=2609)
        if loaded["reward_model"]["gt_output"] != expected:
            raise ValueError("Training reference formatting mismatch")
        if (loaded["extra_info"]["id"] != original["id"]
                or loaded["extra_info"]["language"] != "English"
                or loaded["extra_info"]["prefix"] != ""):
            raise ValueError("Training identity, language or prefix mismatch")
        if loaded["prompt"][0]["role"] != "user" or "<audio>" not in loaded["prompt"][0]["content"]:
            raise ValueError("Missing training audio prompt")
        score = openasr_en_eval(original["raw_response"], original["text"], version=2609)
        if score["n_err"] != original["n_err"] or score["n_ref"] != original["n_ref"]:
            raise ValueError("Recomputed baseline WER disagrees with retained measurements")
        cached = Path(localize_audio_source(original["audio_path"]))
        data = cached.read_bytes()
        if hashlib.sha256(data).hexdigest() != original["audio_sha256"] or len(data) != original["audio_num_bytes"]:
            raise ValueError("Cached source WAV checksum/size mismatch")
        audio, sr = sf.read(cached, dtype="float32")
        if sr != 16000 or audio.ndim != 1 or len(audio) != original["audio_num_samples"] or not np.isfinite(audio).all():
            raise ValueError("Cached source WAV header/signal mismatch")
    result = {
        "filtered_rows": count, "stratified_retained_samples": len(selected),
        "filter_mode": mode,
        "audio_pointers_transcripts_prompts_and_reward_references_verified": True,
        "baseline_wer_recomputed_and_threshold_verified": True,
        "cached_original_wav_sha256_headers_and_signal_verified": True,
        "training_yaml_sha256": hashlib.sha256(yaml_bytes).hexdigest(),
        "filtered_manifest_sha256": summary["filtered_manifest_sha256"],
        "model_path": MODEL_PATH,
        "loader_repo_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    path = root / "validation_loader.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    bf.copy(str(path), destination + "/validation_loader.json", overwrite=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=192)
    run(parser.parse_args())

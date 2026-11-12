#!/usr/bin/env python3
"""Audit the published RIR dataset against its dry-source manifest and WAVs."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import sys

import blobfile as bf
import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_ls_rir_dataset import load_jsonl, original_draws


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = cfg["output_dir"]
    with bf.BlobFile(root + "/data.jsonl", "rb") as stream:
        manifest_bytes = stream.read()
    rows = [json.loads(line) for line in manifest_bytes.splitlines() if line.strip()]
    source = load_jsonl(cfg["source_manifest"])
    noises = load_jsonl(cfg["noise_manifest"])
    noise_paths = [r.get("audio_path") or r.get("audio_file") or r.get("audio") for r in noises]
    rirs = load_jsonl(cfg["rir_manifest"])
    replacement = load_jsonl(cfg["replacement_noise_manifest"])
    with bf.BlobFile(root + "/summary.json", "r") as stream:
        summary = json.load(stream)
    require(len(rows) == len(source) == args.expected_rows, "Row count mismatch")
    require(len({r["id"] for r in rows}) == len(rows), "Duplicate IDs")
    require(summary["config"] == cfg, "Published configuration mismatch")
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    require(digest == summary["output_manifest_sha256"], "Published manifest SHA256 mismatch")
    expected_sizes = {}
    for row, parent in zip(rows, source, strict=True):
        for key in ("id", "text", "source_audio_path"):
            require(row[key] == parent[key], f"Source {key} mismatch: {row['id']}")
        index, speed, snr, _ = original_draws(parent, noise_paths, cfg)
        require(abs(row["speed_factor"] - speed) < 1e-8, "Speed mismatch")
        require(abs(row["snr_db"] - snr) < 1e-8, "SNR draw mismatch")
        require(row["parent_snr_db"] == parent["snr_db"], "Parent SNR mismatch")
        require(cfg["snr_range"][0] <= snr <= cfg["snr_range"][1], "SNR out of range")
        require(abs(row["measured_snr_db_before_pcm16"] - snr) < 0.001, "Measured SNR mismatch")
        rir = random.Random(cfg["rir_seed"] + index).choice(rirs)
        noise = replacement[noise_paths.index(parent["noise_path"])]
        require(row["rir_path"] == rir["audio_path"], "RIR draw mismatch")
        require(row["noise_path"] == noise["audio_path"], "Noise slot mismatch")
        require(row["rir_license"] == row["noise_license"] == cfg["resource_license"], "License mismatch")
        require(row["augmentation_order"] == ["speed", "rir", "noise"], "Augmentation order mismatch")
        require(row["build_fingerprint"] == summary["build_fingerprint"], "Build fingerprint mismatch")
        require(row["audio_path"] == root + "/audio/" + Path(parent["audio_path"]).name, "Output path mismatch")
        require(row["audio_path"] not in expected_sizes, "Duplicate audio paths")
        expected_sizes[row["audio_path"]] = 44 + 2 * row["audio_num_samples"]
    count = total_bytes = 0
    for entry in bf.scandir(root + "/audio"):
        require(entry.is_file, f"Unexpected directory: {entry.path}")
        expected = expected_sizes.pop(entry.path, None)
        require(expected is not None, f"Unexpected audio: {entry.path}")
        require(entry.stat.size == expected, f"WAV size mismatch: {entry.path}")
        count += 1
        total_bytes += entry.stat.size
    require(not expected_sizes, f"Missing {len(expected_sizes)} uploaded WAVs")
    selected = random.Random(20261111).sample(rows, min(args.samples, len(rows)))
    for row in selected:
        with bf.BlobFile(row["audio_path"], "rb") as stream:
            wav = stream.read()
        require(hashlib.sha256(wav).hexdigest() == row["audio_sha256"], "Uploaded WAV SHA256 mismatch")
        info = sf.info(io.BytesIO(wav))
        audio, sr = sf.read(io.BytesIO(wav), dtype="float32")
        require(info.subtype == "PCM_16" and audio.ndim == 1, "WAV format mismatch")
        require(sr == row["sample_rate"] and len(audio) == row["audio_num_samples"], "WAV shape mismatch")
        require(np.isfinite(audio).all() and np.mean(audio.astype(float) ** 2) > 0, "Invalid WAV signal")
        require(float(np.max(np.abs(audio))) <= 1, "WAV peak out of range")
    snrs = np.array([r["snr_db"] for r in rows])
    result = {
        "num_rows": count,
        "all_remote_paths_and_sizes_verified": True,
        "audio_bytes": total_bytes,
        "audio_hours": sum(r["duration"] for r in rows) / 3600,
        "snr_min": float(snrs.min()),
        "snr_max": float(snrs.max()),
        "snr_mean": float(snrs.mean()),
        "snr_below_zero": int((snrs < 0).sum()),
        "max_snr_error_db": max(abs(r["measured_snr_db_before_pcm16"] - r["snr_db"]) for r in rows),
        "remote_sha256_and_signal_samples": len(selected),
        "sample_ids": [r["id"] for r in selected],
        "manifest_sha256": digest,
        "build_fingerprint": summary["build_fingerprint"],
        "config": cfg,
    }
    destination = Path(args.output)
    destination.write_text(json.dumps(result, indent=2))
    bf.copy(str(destination), root + "/validation.json", overwrite=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-rows", type=int, default=20424)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())

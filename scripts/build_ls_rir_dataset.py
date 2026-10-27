#!/usr/bin/env python3
"""Regenerate the retained noisy LibriSpeech examples with an additional RIR.

python scripts/build_ls_rir_dataset.py --work-dir /path/to/build

The original manifest determines sample membership and dry audio paths. Original
per-index random draws are checked against recorded parameters. A licensed noise
pool may replace the old noise slots. RIR selection uses a separate random stream. Completed
rows are resumable locally; data.jsonl is published only after every row succeeds.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import random
import re
import sys

import blobfile as bf
import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.augment_audio import (
    linear_resample,
    mix_noise_at_snr,
    peak_normalize,
    read_audio,
    reverberate_audio,
    speed_perturb,
)

DEFAULT_CONFIG = "recipe/phimm/config/data/cache/gen_ls_raw_rp_edge_nodigits_aug_audio_rir.yaml"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_jsonl(path):
    with bf.BlobFile(str(path), "r", streaming=True, buffer_size=131072) as stream:
        return [json.loads(line) for line in stream if line.strip()]


@lru_cache(maxsize=512)
def read_resource(path):
    return read_audio(path)


def original_draws(row, noise_paths, cfg):
    match = re.match(r"(\d{8})_", Path(row["audio_path"]).name)
    if not match:
        raise ValueError(f"Cannot recover original sample index: {row['audio_path']}")
    index = int(match[1])
    rng = random.Random(cfg["original_seed"] + index)
    rng.random()  # speed probability
    speed = rng.uniform(*cfg["speed_range"])
    rng.random()  # noise probability
    noise_path = rng.choice(noise_paths)
    snr = rng.uniform(*cfg["snr_range"])
    if not np.isclose(speed, row["speed_factor"], atol=1e-8, rtol=0):
        raise ValueError(f"Original speed mismatch for {row['id']}")
    if noise_path != row["noise_path"] or not np.isclose(snr, row["snr_db"], atol=1e-8, rtol=0):
        raise ValueError(f"Original noise/SNR mismatch for {row['id']}")
    return index, speed, snr, rng


def generate_row(row, noise_paths, rirs, cfg, work_dir, fingerprint, local_rir_dir=None, replacements=None):
    index, speed, snr, noise_rng = original_draws(row, noise_paths, cfg)
    marker = work_dir / "rows" / f"{index:08d}.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["build_fingerprint"] != fingerprint:
            raise ValueError("Build inputs changed; use a fresh work directory")
        return saved
    audio, sample_rate = read_audio(row["source_audio_path"])
    sped = speed_perturb(audio, speed)
    rir_rng = random.Random(cfg["rir_seed"] + index)
    rir_record = rir_rng.choice(rirs)
    rir_path = rir_record["audio_path"]
    read_path = str(local_rir_dir / Path(rir_path).name) if local_rir_dir else rir_path
    rir, rir_sr = read_resource(read_path)
    reverberant, rir_info = reverberate_audio(sped, rir, sample_rate, rir_sr, rir_rng)
    noise_record = replacements[noise_paths.index(row["noise_path"])] if replacements else None
    noise_path = noise_record["audio_path"] if noise_record else row["noise_path"]
    noise, noise_sr = read_resource(noise_path)
    # Preserve the old cache's noise resampling convention and crop RNG.
    if noise_sr != sample_rate:
        noise = linear_resample(noise, max(1, round(len(noise) * sample_rate / noise_sr)))
    mixture = mix_noise_at_snr(reverberant, noise, snr, noise_rng)
    measured_snr = float(
        10 * np.log10(np.mean(reverberant.astype(float) ** 2) / np.mean((mixture.astype(float) - reverberant) ** 2))
    )
    if abs(measured_snr - snr) > 0.001:
        raise ValueError(f"SNR verification failed: {measured_snr} != {snr}")
    output_audio = peak_normalize(mixture)
    buffer = io.BytesIO()
    sf.write(buffer, output_audio, sample_rate, format="WAV", subtype="PCM_16")
    wav = buffer.getvalue()
    # Decode the exact bytes that are uploaded, so header and signal checks cover every file.
    decoded, decoded_sr = sf.read(io.BytesIO(wav), dtype="float32")
    if decoded_sr != sample_rate or decoded.shape != output_audio.shape or not np.isfinite(decoded).all():
        raise ValueError("Output WAV verification failed")
    path = f"{cfg['output_dir']}/audio/{Path(row['audio_path']).name}"
    with bf.BlobFile(path, "wb") as stream:
        stream.write(wav)
    # Decodes and error scores inherited from the dry-source mining stage are
    # provenance, not predictions on the new acoustic simulation.
    obsolete = ("response", "raw_response", "wer", "edge_wer")
    result = {key: value for key, value in row.items() if key not in obsolete}
    result.update(
        {
            "audio_path": path,
            "parent_cached_audio_path": row["audio_path"],
            "parent_noise_path": row["noise_path"],
            "noise_path": noise_path,
            "noise_sha256": noise_record.get("sha256") if noise_record else None,
            "noise_license": noise_record.get("license") if noise_record else None,
            "noise_room": noise_record.get("room") if noise_record else None,
            "source_manifest": cfg["source_manifest"],
            "source_generation": {key: row[key] for key in obsolete if key in row},
            "speed_factor": speed,
            "snr_db": snr,
            "measured_snr_db_before_pcm16": measured_snr,
            "rir_path": rir_path,
            "rir_sha256": rir_record["sha256"],
            "rir_license": rir_record.get("license"),
            "rir_room": rir_record.get("room"),
            **rir_info,
            "rir_seed": cfg["rir_seed"],
            "augmentation_order": ["speed", "rir", "noise"],
            "sample_rate": sample_rate,
            "duration": len(output_audio) / sample_rate,
            "audio_num_samples": len(output_audio),
            "audio_sha256": sha256(wav),
            "build_fingerprint": fingerprint,
        }
    )
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False))
    temporary.replace(marker)
    return result


def run(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg["rir_prob"] != 1.0:
        raise ValueError("This matched augmentation requires rir_prob=1")
    if not args.max_examples and bf.exists(f"{cfg['output_dir']}/data.jsonl"):
        raise FileExistsError("Output manifest already exists; refusing to overwrite")
    work_dir = Path(args.work_dir)
    (work_dir / "rows").mkdir(parents=True, exist_ok=True)
    source_path = Path(args.source_local) if args.source_local else work_dir / "source.jsonl"
    if not source_path.exists():
        bf.copy(cfg["source_manifest"], str(source_path), parallel=True)
    rows = load_jsonl(source_path)
    noise_rows = load_jsonl(args.noise_local or cfg["noise_manifest"])
    noise_paths = [r.get("audio_path") or r.get("audio_file") or r.get("audio") for r in noise_rows]
    rirs = load_jsonl(args.rir_manifest_local or cfg["rir_manifest"])
    replacements = load_jsonl(cfg["replacement_noise_manifest"]) if cfg.get("replacement_noise_manifest") else None
    if replacements is not None:
        if len(replacements) != len(noise_paths):
            raise ValueError("Replacement noise pool must match the original slot count")
        if any(r.get("license") != cfg["resource_license"] for r in [*rirs, *replacements]):
            raise ValueError("Every RIR and noise resource must carry the configured license")
    source_digest = sha256(source_path.read_bytes())
    fingerprint = sha256(
        json.dumps(
            {
                "config": cfg,
                "source_sha256": source_digest,
                "rirs": rirs,
                "noise_paths": noise_paths,
                "replacement_noises": replacements,
                "code_sha256": sha256(Path(__file__).read_bytes()),
                "dsp_sha256": sha256(Path("scripts/augment_audio.py").read_bytes()),
            },
            sort_keys=True,
        ).encode()
    )
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source IDs")
    if len({r["source_audio_path"] for r in rows}) != len(rows):
        raise ValueError("Duplicate dry source paths")
    for row in rows:
        original_draws(row, noise_paths, cfg)
    print(json.dumps({"rows": len(rows), "rirs": len(rirs), "fingerprint": fingerprint}), flush=True)
    if args.max_examples:
        rows = rows[: args.max_examples]
    local_rir_dir = Path(args.rir_local_dir) if args.rir_local_dir else None
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers or cfg["workers"]) as executor:
        futures = {
            executor.submit(
                generate_row, row, noise_paths, rirs, cfg, work_dir, fingerprint, local_rir_dir, replacements
            ): i
            for i, row in enumerate(rows)
        }
        for done, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            if done % 100 == 0 or done == len(rows):
                print(f"Completed {done}/{len(rows)}", flush=True)
    ordered = [results[i] for i in range(len(rows))]
    local_manifest = work_dir / ("smoke.jsonl" if args.max_examples else "data.jsonl")
    local_manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered))
    summary = {
        "num_rows": len(rows),
        "audio_hours": sum(r["duration"] for r in ordered) / 3600,
        "rir_files_used": len({r["rir_path"] for r in ordered}),
        "rir_pool_size": len(rirs),
        "source_manifest_sha256": source_digest,
        "build_fingerprint": fingerprint,
        "output_manifest_sha256": sha256(local_manifest.read_bytes()),
        "config": cfg,
        "max_snr_error_db": max(abs(r["snr_db"] - r["measured_snr_db_before_pcm16"]) for r in ordered),
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if not args.max_examples:
        if bf.exists(f"{cfg['output_dir']}/data.jsonl"):
            raise FileExistsError("Output manifest already exists; refusing to overwrite")
        bf.copy(str(local_manifest), f"{cfg['output_dir']}/data.jsonl")
        bf.copy(str(work_dir / "summary.json"), f"{cfg['output_dir']}/summary.json", overwrite=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--source-local")
    parser.add_argument("--noise-local")
    parser.add_argument("--rir-manifest-local")
    parser.add_argument("--rir-local-dir")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-examples", type=int, help="Smoke run; never publishes the final manifest")
    run(parser.parse_args())

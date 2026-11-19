#!/usr/bin/env python3
"""Rebuild retained LS rows from dry audio with seeded uniform speed only.

No selection, noise or RIR is applied. Publish the final manifest
only after every ordered source row has a verified, nonempty PCM16 WAV.
"""

import argparse
import concurrent.futures
import hashlib
import io
import json
import math
import random
from pathlib import Path

import blobfile as bf
import numpy as np
import soundfile as sf


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def speed_only(audio, factor):
    """Match the historical scripts/augment_audio.py linear interpolation."""
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError(f"Invalid speed factor: {factor}")
    output_len = max(1, int(round(len(audio) / factor)))
    if len(audio) == output_len:
        return audio.copy()
    old = np.linspace(0.0, 1.0, num=len(audio), endpoint=True)
    new = np.linspace(0.0, 1.0, num=output_len, endpoint=True)
    if audio.ndim == 1:
        return np.interp(new, old, audio).astype(np.float32)
    return np.stack([np.interp(new, old, audio[:, c]) for c in range(audio.shape[1])], axis=1).astype(np.float32)


def build_row(index, row, output, fingerprint, work):
    marker = work / "rows" / f"{index:08d}.json"
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior["build_fingerprint"] != fingerprint:
            raise ValueError("Resume receipt has different input/config fingerprint")
        if bf.stat(prior["audio_path"]).size != prior["audio_bytes"]:
            raise ValueError("Previously uploaded WAV size changed")
        return prior
    with bf.BlobFile(row["source_audio_path"], "rb") as stream:
        source = stream.read()
    audio, sr = sf.read(io.BytesIO(source), dtype="float32")
    if not audio.size or not np.isfinite(audio).all() or not np.any(audio):
        raise ValueError(f"Invalid dry source: {row['source_audio_path']}")
    augmented = speed_only(audio, float(row["speed_factor"]))
    peak = float(np.max(np.abs(augmented)))
    peak_scale = min(1.0, 0.99 / peak) if peak else 1.0
    augmented *= peak_scale
    buffer = io.BytesIO()
    sf.write(buffer, augmented, sr, format="WAV", subtype="PCM_16")
    wav = buffer.getvalue()
    decoded, decoded_sr = sf.read(io.BytesIO(wav), dtype="float32")
    if decoded_sr != sr or decoded.shape != augmented.shape or not np.isfinite(decoded).all():
        raise ValueError("PCM16 decode/shape verification failed")
    # Keep labels and sample order. Inherited predictions describe old audio.
    stale = ("response", "raw_response", "wer", "edge_wer", "noise_path", "snr_db", "rir_path")
    result = {k: v for k, v in row.items() if k not in stale}
    path = f"{output}/audio/{index:08d}.wav"
    with bf.BlobFile(path, "wb") as stream:
        stream.write(wav)
    if bf.stat(path).size != len(wav):
        raise ValueError("Uploaded WAV length mismatch")
    result.update(
        audio_path=path,
        parent_cached_audio_path=row["audio_path"],
        parent_augmentation={k: row[k] for k in stale if k in row},
        augmentation_order=["speed"],
        noise_path=None,
        rir_path=None,
        snr_db=None,
        source_audio_sha256=sha256(source),
        audio_sha256=sha256(wav),
        audio_bytes=len(wav),
        sample_rate=sr,
        audio_num_samples=len(decoded),
        duration=len(decoded) / sr,
        peak_scale=peak_scale,
        source_num_samples=len(audio),
        source_duration=len(audio) / sr,
        build_fingerprint=fingerprint,
    )
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False))
    temporary.replace(marker)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest", required=True)
    p.add_argument("--source-local", help="Already downloaded exact source manifest")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--expected-rows", type=int, default=20424)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--max-examples", type=int, help="Smoke only; does not publish data.jsonl")
    p.add_argument("--speed-min", type=float, default=0.5)
    p.add_argument("--speed-max", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if not (math.isfinite(args.speed_min) and math.isfinite(args.speed_max) and 0 < args.speed_min <= args.speed_max):
        p.error("speed range must be finite, positive and ordered")
    output = args.output_dir.rstrip("/")
    if bf.exists(output + "/data.jsonl"):
        raise FileExistsError("Immutable destination manifest already exists")
    work = Path(args.work_dir)
    (work / "rows").mkdir(parents=True, exist_ok=True)
    if args.source_local:
        raw = Path(args.source_local).read_bytes()
    else:
        with bf.BlobFile(args.source_manifest, "rb") as f:
            raw = f.read()
    (work / "original_manifest.jsonl").write_bytes(raw)
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} rows, found {len(rows)}")
    if len({r["source_audio_path"] for r in rows}) != len(rows):
        raise ValueError("Duplicate dry source pointers")
    for r in rows:
        if not math.isfinite(float(r["speed_factor"])) or not 1 <= float(r["speed_factor"]) <= 2:
            raise ValueError("Invalid recorded speed")
    rng = random.Random(args.seed)
    for r in rows:
        r["parent_speed_factor"] = r["speed_factor"]
        r["speed_factor"] = rng.uniform(args.speed_min, args.speed_max)
        r["speed_distribution"] = {"kind": "uniform", "min": args.speed_min, "max": args.speed_max, "seed": args.seed}
    policy = json.dumps({"min": args.speed_min, "max": args.speed_max, "seed": args.seed}, sort_keys=True).encode()
    fingerprint = sha256(raw + Path(__file__).read_bytes() + output.encode() + policy)
    selected = rows[: args.max_examples] if args.max_examples else rows
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_row, i, r, output, fingerprint, work) for i, r in enumerate(selected)]
        ordered = [f.result() for f in futures]
    manifest = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered).encode()
    summary = dict(
        rows=len(ordered), source_rows=len(rows), source_manifest=args.source_manifest,
        source_manifest_sha256=sha256(raw), output_manifest_sha256=sha256(manifest),
        build_fingerprint=fingerprint, augmentation_order=["speed"],
        speed_policy="Seeded independent uniform draw per source row, frozen offline; historical linear interpolation",
        configured_speed_range=[args.speed_min, args.speed_max], seed=args.seed,
        speed_range=[min(r["speed_factor"] for r in ordered), max(r["speed_factor"] for r in ordered)],
        speed_mean=float(np.mean([r["speed_factor"] for r in ordered])),
        speed_histogram=np.histogram([r["speed_factor"] for r in ordered], bins=10, range=(args.speed_min,args.speed_max))[0].tolist(),
        max_duration=max(r["duration"] for r in ordered),
        durations_over_40_seconds=sum(r["duration"] > 40 for r in ordered),
        noise_prob=0, rir_prob=0, original_samples_and_order_preserved=True,
        pcm16_verified=True, uploaded_file_sizes_verified=True,
        hours=sum(r["duration"] for r in ordered) / 3600,
        total_audio_bytes=sum(r["audio_bytes"] for r in ordered), smoke_only=bool(args.max_examples),
    )
    (work / "data.jsonl").write_bytes(manifest)
    (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.max_examples:
        for name in ["summary.json", "original_manifest.jsonl"]:
            bf.copy(str(work / name), output + "/" + name)
        if bf.exists(output + "/data.jsonl"):
            raise FileExistsError("Manifest was concurrently published")
        bf.copy(str(work / "data.jsonl"), output + "/data.jsonl")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

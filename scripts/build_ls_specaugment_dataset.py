#!/usr/bin/env python3
"""Generate deterministic per-audio randomized SpecAugment WAVs on a remote CPU node.

Reads dry source_audio_path, retaining every source row, label and its order.
Only publishes data.jsonl after decoding and verifying every uploaded WAV.
"""

import argparse
import concurrent.futures
import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path

import blobfile as bf
import numpy as np
import soundfile as sf
import yaml

try:
    from scripts.specaugment_wav import SpecAugmentConfig, augment_waveform
except ModuleNotFoundError:
    from specaugment_wav import SpecAugmentConfig, augment_waveform


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def sample_seed(global_seed, sample_id, source_path):
    """Independent of thread scheduling, Python hash salt and manifest order."""
    identity = json.dumps([global_seed, sample_id, source_path], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")


def build_row(index, row, output, fingerprint, work, policy):
    marker = work / "rows" / f"{index:08d}.json"
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior["build_fingerprint"] != fingerprint:
            raise ValueError("Resume receipt has a different input/config/code fingerprint")
        if bf.stat(prior["audio_path"]).size != prior["audio_bytes"]:
            raise ValueError("Previously uploaded WAV size changed")
        return prior
    source_path = row[policy["source_audio_field"]]
    with bf.BlobFile(source_path, "rb") as stream:
        source = stream.read()
    audio, rate = sf.read(io.BytesIO(source), dtype="float64")
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"Expected finite nonempty mono speech: {source_path}")
    if len(audio) / rate > 40:
        raise ValueError("Audio exceeds unchanged training loader duration limit; refusing to filter")
    seed = sample_seed(policy["global_seed"], row["id"], source_path)
    config = SpecAugmentConfig(**policy["specaugment"])
    augmented, sampled = augment_waveform(audio, rate, config, seed=seed, return_metadata=True)
    if augmented.shape != audio.shape or not np.isfinite(augmented).all():
        raise ValueError("Augmentation changed shape or produced nonfinite samples")
    buffer = io.BytesIO()
    sf.write(buffer, augmented, rate, format="WAV", subtype="FLOAT")
    wav = buffer.getvalue()
    decoded, decoded_rate = sf.read(io.BytesIO(wav), dtype="float64")
    if decoded_rate != rate or decoded.shape != audio.shape or not np.isfinite(decoded).all():
        raise ValueError("Generated WAV failed decode/shape verification")
    path = f"{output}/audio/{index:08d}.wav"
    with bf.BlobFile(path, "wb") as stream:
        stream.write(wav)
    # Verify the stored bytes, not just the pre-upload buffer.
    with bf.BlobFile(path, "rb") as stream:
        uploaded = stream.read()
    if sha256(uploaded) != sha256(wav):
        raise ValueError("Uploaded WAV SHA256 mismatch")
    stale = ("response", "raw_response", "wer", "edge_wer", "speed_factor", "noise_path", "snr_db", "rir_path")
    result = {k: v for k, v in row.items() if k not in stale}
    result.update(
        audio_path=path,
        audio_chunk=None,
        parent_cached_audio_path=row["audio_path"],
        parent_augmentation={k: row[k] for k in stale if k in row},
        augmentation_order=["specaugment"],
        speed_factor=1.0,
        noise_path=None,
        rir_path=None,
        snr_db=None,
        specaugment_config=asdict(config),
        specaugment_seed=seed,
        specaugment_sampled=sampled,
        source_audio_sha256=sha256(source),
        audio_sha256=sha256(wav),
        audio_bytes=len(wav),
        sample_rate=rate,
        audio_num_samples=len(decoded),
        duration=len(decoded) / rate,
        audio_format="WAV",
        audio_subtype="FLOAT",
        peak_scale=1.0,
        build_fingerprint=fingerprint,
    )
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n")
    temporary.replace(marker)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=20424)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--max-examples", type=int, help="Smoke only; does not publish data.jsonl")
    args = parser.parse_args()
    output = args.output_dir.rstrip("/")
    if bf.exists(output + "/data.jsonl"):
        raise FileExistsError("Immutable destination manifest already exists")
    policy = yaml.safe_load(args.config.read_text())
    SpecAugmentConfig(**policy["specaugment"])
    with bf.BlobFile(args.source_manifest, "rb") as stream:
        raw = stream.read()
    if sha256(raw) != args.source_sha256:
        raise ValueError("Source manifest hash does not match the baseline cohort")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != args.expected_rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Unexpected cohort row count or duplicate sample IDs")
    if len({r[policy["source_audio_field"]] for r in rows}) != len(rows):
        raise ValueError("Duplicate dry source pointers")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "rows").mkdir(exist_ok=True)
    (args.work_dir / "original_manifest.jsonl").write_bytes(raw)
    (args.work_dir / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    core = Path(__file__).with_name("specaugment_wav.py").read_bytes()
    fingerprint = sha256(raw + args.config.read_bytes() + Path(__file__).read_bytes() + core + output.encode())
    selected = rows[: args.max_examples] if args.max_examples else rows
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(build_row, i, row, output, fingerprint, args.work_dir, policy) for i, row in enumerate(selected)
        ]
        ordered = []
        for i, future in enumerate(futures):
            ordered.append(future.result())
            if (i + 1) % 500 == 0:
                print(json.dumps({"completed_ordered_rows": i + 1, "total": len(selected)}), flush=True)
    assert all(a["id"] == b["id"] and a["text"] == b["text"] for a, b in zip(ordered, selected, strict=True))
    seeds = [r["specaugment_seed"] for r in ordered]
    assert len(set(seeds)) == len(seeds), "Per-audio seed collision"
    manifest = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered).encode()
    summary = dict(
        status="PASSED",
        rows=len(ordered),
        source_rows=len(rows),
        source_manifest=args.source_manifest,
        source_manifest_sha256=sha256(raw),
        output_manifest_sha256=sha256(manifest),
        build_fingerprint=fingerprint,
        policy=policy,
        original_samples_labels_and_order_preserved=True,
        independent_per_audio_seeds=True,
        waveform_shape_and_rate_preserved=True,
        all_audio_decoded_finite=True,
        uploaded_wav_sha256_verified=True,
        hours=sum(r["duration"] for r in ordered) / 3600,
        total_audio_bytes=sum(r["audio_bytes"] for r in ordered),
        smoke_only=bool(args.max_examples),
    )
    (args.work_dir / "data.jsonl").write_bytes(manifest)
    (args.work_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.max_examples:
        for name in ["original_manifest.jsonl", "policy.json", "summary.json"]:
            bf.copy(str(args.work_dir / name), output + "/" + name)
        if bf.exists(output + "/data.jsonl"):
            raise FileExistsError("Destination manifest was concurrently published")
        bf.copy(str(args.work_dir / "data.jsonl"), output + "/data.jsonl")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

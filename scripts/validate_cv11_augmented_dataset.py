#!/usr/bin/env python3
"""Independently verify every uploaded CV11 WAV's existence/size and sample bytes.

Can run alongside the builder: it audits only completed upload checkpoints, then
checks the final manifest's SHA256 once all shards have been published.
"""

import argparse
import asyncio
import hashlib
import io
import json
from pathlib import Path
import random
import time

import blobfile as bf
from boostedblob.listing import scantree
from boostedblob.read import read_single
import numpy as np
import soundfile as sf


async def audit_shard(root, name, samples):
    rows = [json.loads(line) for line in (root / "shards" / f"{name}.jsonl").open()]
    expected = {r["audio_path"]: r["audio_num_bytes"] for r in rows}
    if len(expected) != len(rows):
        raise ValueError("Duplicate manifest paths")
    remote = rows[0]["audio_path"].rsplit("/", 1)[0] + "/"
    count = 0
    total_bytes = 0
    async for entry in scantree(remote):
        if not entry.is_file:
            continue
        path = str(entry.path)
        expected_size = expected.pop(path, None)
        if expected_size is None or entry.stat.size != expected_size:
            raise ValueError(f"Unexpected path/size: {path}")
        count += 1
        total_bytes += entry.stat.size
    if expected:
        raise ValueError(f"Missing {len(expected)} remote WAVs in {name}")
    selected = random.Random(20261106 + int(name.rsplit("_", 1)[-1])).sample(rows, min(samples, len(rows)))

    async def verify(row):
        data = await read_single(row["audio_path"])
        if hashlib.sha256(data).hexdigest() != row["audio_sha256"]:
            raise ValueError(f"Remote SHA256 mismatch: {row['id']}")
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        if sr != row["sample_rate"] or len(audio) != row["audio_num_samples"] or audio.ndim != 1:
            raise ValueError(f"Remote WAV header mismatch: {row['id']}")
        if not np.isfinite(audio).all() or np.max(np.abs(audio)) > 0.991 or np.mean(audio.astype(float)**2) <= 0:
            raise ValueError(f"Remote signal validation failed: {row['id']}")
        return row["id"]

    ids = await asyncio.gather(*(verify(row) for row in selected))
    result = {"shard": name, "all_remote_wav_count": count, "all_remote_wav_bytes": total_bytes,
              "sha256_and_signal_samples": len(ids), "sample_ids": ids}
    print(json.dumps(result), flush=True)
    return result


def run(args):
    root = Path(args.work_dir)
    output = root / "remote_validation"
    output.mkdir(exist_ok=True)
    while True:
        for marker in sorted((root / "shards").glob("*.uploaded.json")):
            name = marker.name.removesuffix(".uploaded.json")
            result_path = output / f"{name}.json"
            checkpoint = json.loads(marker.read_text())
            if result_path.exists():
                previous = json.loads(result_path.read_text())
                if previous["checkpoint"] != checkpoint:
                    raise ValueError("Upload checkpoint changed after audit")
                continue
            result = asyncio.run(audit_shard(root, name, args.samples_per_shard))
            result["checkpoint"] = checkpoint
            result_path.write_text(json.dumps(result, indent=2))
        provenance = root / "provenance.json"
        if provenance.exists():
            info = json.loads(provenance.read_text())
            checks = list(output.glob("en_train_*.json"))
            if len(checks) == len(info["archives"]):
                remote = info["config"]["output_dir"]
                if bf.exists(remote + "/data.jsonl"):
                    digest = hashlib.sha256()
                    with bf.BlobFile(remote + "/data.jsonl", "rb", streaming=True) as stream:
                        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                            digest.update(block)
                    summary = json.loads((root / "artifacts/summary.json").read_text())
                    if digest.hexdigest() != summary["manifest_sha256"]:
                        raise ValueError("Remote final manifest SHA256 mismatch")
                    results = [json.loads(p.read_text()) for p in checks]
                    result = {"num_rows": sum(r["all_remote_wav_count"] for r in results),
                              "audio_bytes": sum(r["all_remote_wav_bytes"] for r in results),
                              "remote_sha256_and_signal_samples": sum(r["sha256_and_signal_samples"] for r in results),
                              "remote_manifest_sha256": digest.hexdigest(), "shards": len(checks),
                              "build_fingerprint": summary["build_fingerprint"],
                              "all_remote_paths_and_sizes_verified": True}
                    if result["num_rows"] != info["config"]["expected_rows"]:
                        raise ValueError("Final remote row count mismatch")
                    destination = root / "artifacts/validation.json"
                    destination.write_text(json.dumps(result, indent=2))
                    bf.copy(str(destination), remote + "/validation.json")
                    print(json.dumps(result, indent=2), flush=True)
                    return
        if args.once:
            return
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--samples-per-shard", type=int, default=8)
    parser.add_argument("--once", action="store_true")
    run(parser.parse_args())

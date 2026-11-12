"""Prune only this case's private, uniquely and durably decoded audio cache.

Run on EACH dedicated generation pod, dry-run first. Keep the newest output
part as an in-flight margin. Never touch original blobs, references, models,
the other OSS case or shared caches. Bitmaps reject duplicate sample indices.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import blobfile as bf

CASE = "oss_gap_2611r2_errors_20261112"
PREFIX = "az://orngwus2cresco/data/"
BLOB = PREFIX + "boren/data/" + CASE
PRIVATE_ROOT = Path("/root/data/" + CASE + "_cache")
LEDGER = Path("/tmp/" + CASE + "_prune_cursor.json")
SOURCE_SHA256 = "99db23934dc56bbb8ac7b21996845a14cce7890bc6f5aaf6f29ab27e6f54d133"


def add_record(masks, chunks, record):
    audio, n, idx = record["audio_chunk"].rsplit(":", 2)
    n, idx = int(n), int(idx)
    if audio not in chunks or n != chunks[audio] or not 0 <= idx < n:
        raise ValueError(f"Unknown/out-of-bounds source selector: {record['audio_chunk']}")
    previous = masks.get(audio, 0)
    bit = 1 << idx
    if previous & bit:
        raise ValueError(f"Duplicate generation sample: {record['audio_chunk']}")
    masks[audio] = previous | bit


def private_audio_path(audio, root):
    if not audio.startswith(PREFIX) or not audio.endswith((".audio", ".audios")):
        raise ValueError(f"Not an eligible audio blob: {audio}")
    if root.resolve() != root:
        raise ValueError(f"Private cache root must not resolve through a symlink: {root}")
    path = (root / audio.removeprefix(PREFIX)).resolve()
    path.relative_to(root)
    return path


def main(apply=False):
    with bf.BlobFile(BLOB + "/source_chunks.json", "rb") as f:
        raw = f.read()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Frozen source identity changed")
    sources = json.loads(raw)
    chunks = {x["audio_key"]: x["count"] for x in sources}
    if len(chunks) != 20538 or sum(chunks.values()) != 36993182:
        raise ValueError("Unexpected source cardinality")
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {
        "case_id": CASE, "source_sha256": SOURCE_SHA256, "parts": {}, "masks": {}, "pruned": []}
    if ledger.get("case_id") != CASE or ledger.get("source_sha256") != SOURCE_SHA256:
        raise ValueError("Prune ledger belongs to a different source/case")
    masks = {a: int(v, 16) for a, v in ledger["masks"].items()}
    for audio, bits in masks.items():
        if audio not in chunks or bits < 0 or bits.bit_length() > chunks[audio]:
            raise ValueError("Corrupt prune bitmap")
    parts = sorted(bf.glob(BLOB + "/generation/part-*.jsonl"), key=lambda x: int(re.search(r"part-(\d+)\.jsonl$", x).group(1)))
    for i, part in enumerate(parts):
        if int(re.search(r"part-(\d+)\.jsonl$", part).group(1)) != i:
            raise ValueError("Non-contiguous durable generation prefix")
    if not set(ledger["parts"]).issubset(parts):
        raise ValueError("Previously recorded durable generation parts disappeared")
    # Validate all new eligible output records before deleting any file.
    for part in parts[:-1]:
        if part in ledger["parts"]:
            continue
        digest, rows = hashlib.sha256(), 0
        with bf.BlobFile(part, "rb") as f:
            for line in f:
                digest.update(line)
                record = json.loads(line)
                if record.get("selection_scorer") != "oss_gap_error_mining_v1":
                    raise ValueError("Wrong scorer in generation output")
                add_record(masks, chunks, record)
                rows += 1
        if not rows:
            raise ValueError("Empty generation part")
        ledger["parts"][part] = {"sha256": digest.hexdigest(), "rows": rows}
    actions = []
    pruned = set(ledger["pruned"])
    for audio, mask in masks.items():
        if mask.bit_count() != chunks[audio] or audio in pruned:
            continue
        path = private_audio_path(audio, PRIVATE_ROOT)
        if path.is_file():
            actions.append({"path": str(path), "bytes": path.stat().st_size})
            if apply:
                path.unlink()
        if apply:
            pruned.add(audio)
    if apply:
        ledger["masks"] = {a: hex(v) for a, v in masks.items()}
        ledger["pruned"] = sorted(pruned)
        temporary = LEDGER.with_suffix(".tmp")
        temporary.write_text(json.dumps(ledger, indent=2))
        temporary.replace(LEDGER)
    print(json.dumps({"case_id": CASE, "apply": apply, "private_root": str(PRIVATE_ROOT),
                      "durable_parts_processed": len(ledger["parts"]), "newest_part_retained": parts[-1] if parts else None,
                      "unique_durable_samples": sum(v.bit_count() for v in masks.values()),
                      "files": len(actions), "bytes": sum(a["bytes"] for a in actions), "actions": actions[:10]}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true")
    main(p.parse_args().apply)

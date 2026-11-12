"""Remove only pipeline-private audio chunks durably decoded in an old part.

Keep the newest generation part as an additional in-flight margin. Never prune
the default/shared ~/data cache, source blobs, model files, or transcript chunks.
Run on the dedicated generation node with --apply after inspecting the plan.
"""
import argparse
import collections
import json
from pathlib import Path

import blobfile as bf

PREFIX = "az://orngwus2cresco/data/"
PRIVATE_ROOT = Path("/root/data/oss_2611r2_generation_cache")
BLOB = PREFIX + "boren/data/oss_refine_2611r2_20261112"


def main(apply=False):
    with bf.BlobFile(BLOB + "/source_chunks.json") as f:
        chunks = {r["audio_key"]: r["count"] for r in json.load(f)}
    ledger_path = Path("/tmp/oss_2611r2_cache_prune_cursor.json")
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"parts": [], "counts": {}, "pruned": []}
    parts = sorted(bf.glob(BLOB + "/generation/part-*.jsonl"), key=lambda p: int(Path(p).stem.split("-")[-1]))
    counts = collections.Counter(ledger["counts"])
    for part in parts[:-1]:
        if part in ledger["parts"]:
            continue
        with bf.BlobFile(part) as f:
            for line in f:
                r = json.loads(line)
                audio, n, idx = r["audio_chunk"].rsplit(":", 2)
                if audio not in chunks or int(n) != chunks[audio]:
                    raise ValueError(f"Unknown chunk in output: {audio}")
                counts[audio] += 1
                if counts[audio] > chunks[audio]:
                    raise ValueError(f"Duplicate generation coverage: {audio}")
        ledger["parts"].append(part)
    actions = []
    for audio, count in counts.items():
        if count != chunks[audio] or audio in ledger["pruned"]:
            continue
        if not audio.startswith(PREFIX) or not audio.endswith((".audio", ".audios")):
            raise ValueError(audio)
        path = (PRIVATE_ROOT / audio.removeprefix(PREFIX)).resolve()
        path.relative_to(PRIVATE_ROOT)
        if path.is_file():
            actions.append({"path": str(path), "bytes": path.stat().st_size})
            if apply:
                path.unlink()
        if apply:
            ledger["pruned"].append(audio)
    if apply:
        ledger["counts"] = dict(counts)
        temp = ledger_path.with_suffix(".tmp")
        temp.write_text(json.dumps(ledger, indent=2))
        temp.replace(ledger_path)
    print(json.dumps({"apply": apply, "files": len(actions), "bytes": sum(a["bytes"] for a in actions), "private_root": str(PRIVATE_ROOT), "actions": actions[:10]}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true")
    main(p.parse_args().apply)

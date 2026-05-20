#!/usr/bin/env python3
"""Convert phi-fastllm eval result `.txt` files (JSON arrays) into JSONL.

Each input `.txt` is a JSON list of objects with parallel-list fields
`generated_texts`, `audio_ids`, `label`. We flatten each list element into one
JSON line with fields `audio_id`, `label`, `hyp` (first generation) and
`generated_texts` (full n-best). Output is written next to the input as
`<same-stem>.jsonl` on the same blob path.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import re
import sys
from typing import Iterable

import blobfile as bf

ROOT = (
    "az://orngwus2cresco/data/speech/projects/phi-fastllm-2605/amlt-results/"
    "fast-llm-2605-qwen3-5-9b-s2-st-example/90000/eval_output"
)

AUX_SUFFIXES = ("_eer", "_ewer", "_disfluencytolerant_ter")


def is_main_result(path: str) -> bool:
    if not path.endswith(".txt"):
        return False
    name = path.rsplit("/", 1)[-1]
    if not name.startswith("generate_"):
        return False
    stem = name[:-4]
    return not any(stem.endswith(s) for s in AUX_SUFFIXES)


def iter_inputs(root: str, workers: int = 16) -> Iterable[str]:
    # Structure: <root>/<dataset>/<seed>/generate_*.txt
    # Use parallel listdir at each level; faster than bf.glob.
    datasets = [d.rstrip("/") for d in bf.listdir(root)]

    def list_seeds(ds: str) -> list[str]:
        return [s.rstrip("/") for s in bf.listdir(f"{root}/{ds}")]

    def list_files(ds_seed: tuple[str, str]) -> list[str]:
        ds, seed = ds_seed
        base = f"{root}/{ds}/{seed}"
        return [f"{base}/{n}" for n in bf.listdir(base) if is_main_result(f"{base}/{n}")]

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        ds_seeds: list[tuple[str, str]] = []
        for ds, seeds in zip(datasets, ex.map(list_seeds, datasets)):
            for s in seeds:
                ds_seeds.append((ds, s))
        for files in ex.map(list_files, ds_seeds):
            yield from files


def convert_one(src: str, *, overwrite: bool = False) -> tuple[str, int, str]:
    dst = src[:-4] + ".jsonl"
    if not overwrite and bf.exists(dst):
        return src, -1, "skip-exists"
    try:
        with bf.BlobFile(src, "rb") as f:
            data = json.load(io.TextIOWrapper(f, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return src, 0, f"read-error: {e!r}"

    lines: list[str] = []
    for entry in data:
        gens = entry.get("generated_texts") or []
        aids = entry.get("audio_ids") or []
        labels = entry.get("label") or []
        n = max(len(gens), len(aids), len(labels))
        for i in range(n):
            aid = aids[i] if i < len(aids) else None
            if isinstance(aid, list) and len(aid) == 1:
                aid = aid[0]
            rec = {
                "audio_id": aid,
                "label": labels[i] if i < len(labels) else None,
                "hyp": gens[i] if i < len(gens) else None,
            }
            if i < len(gens) and isinstance(gens[i], list):
                rec["hyp"] = gens[i][0] if gens[i] else None
                rec["generated_texts"] = gens[i]
            lines.append(json.dumps(rec, ensure_ascii=False))

    payload = ("\n".join(lines) + "\n").encode("utf-8")
    with bf.BlobFile(dst, "wb") as f:
        f.write(payload)
    return src, len(lines), "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--filter", default="", help="regex to match input path")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pat = re.compile(args.filter) if args.filter else None
    inputs = [p for p in iter_inputs(args.root, workers=args.workers) if not pat or pat.search(p)]
    print(f"found {len(inputs)} input files", file=sys.stderr)
    if args.dry_run:
        for p in inputs[:20]:
            print(p)
        return 0

    ok = skipped = errored = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(convert_one, p, overwrite=args.overwrite) for p in inputs]
        for fut in cf.as_completed(futs):
            src, n, status = fut.result()
            short = src.rsplit("/eval_output/", 1)[-1]
            if status == "ok":
                ok += 1
            elif status == "skip-exists":
                skipped += 1
            else:
                errored += 1
            print(f"[{status}] n={n} {short}")
    print(f"done. ok={ok} skipped={skipped} errored={errored}", file=sys.stderr)
    return 0 if errored == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Convert phi-fastllm ASR eval `.txt` results directly to the final layout.

Source:
    <root>/eval_output/<dataset>/<seed>/generate_<name>.txt    # JSON array

Target:
    <root>/jsonl_results/<dataset>/<name>.jsonl                # unique
    <root>/jsonl_results/<dataset>/<name>_1.jsonl, _2.jsonl    # extra seeds

Within each ``<dataset>/<name>`` group the first seed (alphabetical) gets the
unsuffixed name; additional seeds get ``_1``, ``_2``, ... — no suffix is added
when only one seed exists.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import re
import sys

import blobfile as bf

DEFAULT_ROOT = (
    "az://orngwus2cresco/data/speech/projects/phi-fastllm-2605/amlt-results/"
    "fast-llm-2605-qwen3-5-9b-s2-st-example/90000"
)
DEFAULT_SRC_SUBDIR = "eval_output"
DEFAULT_DST_SUBDIR = "jsonl_results"

AUX_SUFFIXES = ("_eer", "_ewer", "_disfluencytolerant_ter")


def is_main_result(name: str) -> bool:
    if not name.endswith(".txt") or not name.startswith("generate_"):
        return False
    stem = name[:-4]
    return not any(stem.endswith(s) for s in AUX_SUFFIXES)


def base_name(fname: str) -> str:
    """``generate_foo.txt`` -> ``foo``."""
    n = fname[:-4] if fname.endswith(".txt") else fname
    if n.startswith("generate_"):
        n = n[len("generate_") :]
    return n


def discover(src_root: str, workers: int) -> list[tuple[str, str, str, str]]:
    """Return [(dataset, base, seed, src_full_path), ...]."""
    datasets = sorted(d.rstrip("/") for d in bf.listdir(src_root))

    def list_seeds(ds: str) -> list[tuple[str, str]]:
        return [(ds, s.rstrip("/")) for s in bf.listdir(f"{src_root}/{ds}")]

    def list_files(ds_seed: tuple[str, str]) -> list[tuple[str, str, str, str]]:
        ds, seed = ds_seed
        base = f"{src_root}/{ds}/{seed}"
        try:
            entries = bf.listdir(base)
        except (NotADirectoryError, FileNotFoundError):
            return []
        return [(ds, base_name(n), seed, f"{base}/{n}") for n in entries if is_main_result(n)]

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        ds_seeds: list[tuple[str, str]] = []
        for seeds in ex.map(list_seeds, datasets):
            ds_seeds.extend(seeds)
        out: list[tuple[str, str, str, str]] = []
        for batch in ex.map(list_files, ds_seeds):
            out.extend(batch)
    return out


def plan_targets(
    items: list[tuple[str, str, str, str]], dst_root: str
) -> list[tuple[str, str]]:
    """Group by (dataset, base); assign ``_1``/``_2``/... to extra seeds."""
    groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for ds, base, seed, src in items:
        groups.setdefault((ds, base), []).append((seed, src))

    plan: list[tuple[str, str]] = []
    for (ds, base), members in groups.items():
        members.sort(key=lambda x: x[0])
        for i, (_seed, src) in enumerate(members):
            name = f"{base}.jsonl" if i == 0 else f"{base}_{i}.jsonl"
            plan.append((src, f"{dst_root}/{ds}/{name}"))
    return plan


def convert_one(src: str, dst: str, *, overwrite: bool) -> tuple[str, str, int, str]:
    if not overwrite and bf.exists(dst):
        return src, dst, -1, "skip-exists"
    try:
        with bf.BlobFile(src, "rb") as f:
            data = json.load(io.TextIOWrapper(f, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return src, dst, 0, f"read-error: {e!r}"

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
    return src, dst, len(lines), "ok"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--src-subdir", default=DEFAULT_SRC_SUBDIR)
    ap.add_argument("--dst-subdir", default=DEFAULT_DST_SUBDIR)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--filter", default="", help="regex matched against `<dataset>/<base>`")
    args = ap.parse_args()

    src_root = f"{args.root}/{args.src_subdir}"
    dst_root = f"{args.root}/{args.dst_subdir}"

    items = discover(src_root, args.workers)
    if args.filter:
        pat = re.compile(args.filter)
        items = [it for it in items if pat.search(f"{it[0]}/{it[1]}")]
    print(f"discovered {len(items)} source .txt files", file=sys.stderr)

    plan = plan_targets(items, dst_root)
    print(f"planned {len(plan)} writes", file=sys.stderr)

    src_marker = f"/{args.src_subdir}/"
    dst_marker = f"/{args.dst_subdir}/"

    if args.dry_run:
        for s, d in plan[:40]:
            print(f"{s.split(src_marker, 1)[-1]} -> {d.split(dst_marker, 1)[-1]}")
        if len(plan) > 40:
            print(f"... ({len(plan)} total)")
        return 0

    counts: dict[str, int] = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(convert_one, s, d, overwrite=args.overwrite) for s, d in plan]
        for fut in cf.as_completed(futs):
            _src, dst, n, status = fut.result()
            key = "ok" if status == "ok" else ("skip-exists" if status == "skip-exists" else "error")
            counts[key] = counts.get(key, 0) + 1
            print(f"[{status}] n={n} {dst.split(dst_marker, 1)[-1]}")
    print(f"done. {counts}", file=sys.stderr)
    return 0 if counts.get("error", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

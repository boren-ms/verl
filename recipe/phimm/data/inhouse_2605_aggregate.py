"""Aggregate per-segment hyps into per-parent DTER for InhouseASR_2605_seg eval.

The SVAD-segmented eval writes one verl ``val_data_gen/<data_source>/<step>.jsonl``
row per segment (with the FULL parent reference duplicated on every row as
``gts``). Per-row DTER is meaningless; the real metric is
``DTER(concat(hyp_seg) vs full_ref)`` per parent wav, then aggregated per corpus.

Each row contains at least: ``input, output, gts, score, data_source, audio_path``.
``audio_path`` has the form ``<wav>#<start>:<end>`` so we group by stripping the
``#start:end`` suffix and sort segments by start.

Usage:
    python -m recipe.phimm.data.inhouse_2605_aggregate \\
        --val-data-dir az://.../val_data_gen \\
        [--step 0]
"""

from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict

import blobfile as bf

from recipe.phimm.reward.asr_inhouse_measure import (
    _clean_ref,
    _compute_dter,
    ensure_pack_dir,
)
from recipe.phimm.utils.shared import parse_asr_response


def _parent_key(audio_path: str) -> str:
    return audio_path.split("#", 1)[0]


def _seg_start(audio_path: str) -> float:
    if "#" not in audio_path:
        return 0.0
    rng = audio_path.split("#", 1)[1]
    try:
        return float(rng.split(":", 1)[0])
    except Exception:
        return 0.0


def _iter_rows(val_data_dir: str, step: int | None):
    sources = [p.rstrip("/") for p in bf.glob(os.path.join(val_data_dir, "*/"))]
    print(f"Found {len(sources)} data_source dirs under {val_data_dir}", flush=True)
    for src_dir in sources:
        files = list(bf.glob(os.path.join(src_dir, "*.jsonl")))
        if not files:
            continue
        if step is not None:
            files = [f for f in files if os.path.basename(f) == f"{step}.jsonl"]
        else:
            def _step_of(p):
                try:
                    return int(os.path.basename(p).removesuffix(".jsonl"))
                except ValueError:
                    return -1
            files = [max(files, key=_step_of)]
        for f in files:
            with bf.BlobFile(f, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    ap = d.get("audio_path") or ""
                    yield (
                        d.get("data_source", os.path.basename(src_dir)),
                        ap,
                        d.get("gts") or "",
                        d.get("output") or "",
                    )


def aggregate(val_data_dir: str, step: int | None = None) -> dict:
    ensure_pack_dir(None)

    by_parent: dict[str, dict] = {}
    rows_seen = 0
    for corpus, ap, gts, output in _iter_rows(val_data_dir, step):
        if not ap:
            continue
        rows_seen += 1
        pk = _parent_key(ap)
        d = by_parent.setdefault(pk, {"ref": gts, "segs": [], "corpus": corpus})
        hyp = (parse_asr_response(output) or {}).get("text") or ""
        d["segs"].append((_seg_start(ap), hyp))
    print(f"Loaded {rows_seen} segment rows -> {len(by_parent)} parents", flush=True)

    corp_err: dict[str, int] = defaultdict(int)
    corp_ref: dict[str, int] = defaultdict(int)
    corp_parents: dict[str, int] = defaultdict(int)
    per_parent_rows = []

    for pk, d in by_parent.items():
        segs = sorted(d["segs"], key=lambda x: x[0])
        hyp_full = " ".join(s for _, s in segs if s).strip()
        ref_full = _clean_ref(d["ref"] or "")
        n_err, n_ref, dter, _ = _compute_dter(ref_full, hyp_full)
        corp = d["corpus"]
        corp_err[corp] += n_err
        corp_ref[corp] += n_ref
        corp_parents[corp] += 1
        per_parent_rows.append({
            "parent": pk,
            "corpus": corp,
            "n_segments": len(segs),
            "n_err": n_err,
            "n_ref": n_ref,
            "dter": dter,
        })

    print(f"\n{'Corpus':<70} {'N_parent':>8} {'N_err':>8} {'N_ref':>8} {'DTER%':>8}")
    print("-" * 110)
    total_err = total_ref = total_parents = 0
    for c in sorted(corp_err):
        e, r, p = corp_err[c], corp_ref[c], corp_parents[c]
        total_err += e
        total_ref += r
        total_parents += p
        print(f"{c:<70} {p:>8} {e:>8} {r:>8} {(e / max(r, 1) * 100):>8.2f}")
    print("-" * 110)
    print(f"{'OVERALL':<70} {total_parents:>8} {total_err:>8} {total_ref:>8} "
          f"{(total_err / max(total_ref, 1) * 100):>8.2f}")

    tag = step if step is not None else "latest"
    summary = {
        "val_data_dir": val_data_dir,
        "step": step,
        "per_corpus": {
            c: {"n_parent": corp_parents[c], "n_err": corp_err[c],
                 "n_ref": corp_ref[c], "dter": corp_err[c] / max(corp_ref[c], 1)}
            for c in corp_err
        },
        "overall": {"n_parent": total_parents, "n_err": total_err, "n_ref": total_ref,
                    "dter": total_err / max(total_ref, 1)},
    }
    out_summary = os.path.join(val_data_dir, f"aggregate_summary_{tag}.json")
    with bf.BlobFile(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    out_details = os.path.join(val_data_dir, f"aggregate_per_parent_{tag}.jsonl")
    with bf.BlobFile(out_details, "w") as f:
        for row in per_parent_rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {out_summary}\nWrote {out_details}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val-data-dir", required=True,
                   help="Verl trainer val_data_gen dir (az:// or local).")
    p.add_argument("--step", type=int, default=None,
                   help="Specific eval step jsonl to aggregate (default: latest per data_source).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    aggregate(args.val_data_dir, args.step)

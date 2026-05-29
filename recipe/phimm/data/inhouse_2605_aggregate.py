"""Aggregate per-segment hyps into per-parent DTER for InhouseASR_2605_seg eval.

The SVAD-segmented eval writes one parquet row per segment (with the FULL parent
reference duplicated on every row). Per-row DTER is meaningless; the real metric
is DTER(concat(hyp_seg) vs full_ref) per parent wav, then aggregated per corpus.

Usage:
    python -m recipe.phimm.data.inhouse_2605_aggregate \\
        --eval-dir az://orngwus2cresco/data/boren/data/Evaluation/InhouseASR_2605_seg_eval/<model>/
"""

from __future__ import annotations
import argparse
import io
import json
import os
import re
from collections import defaultdict

import blobfile as bf

from recipe.phimm.reward.asr_inhouse_measure import _clean_ref, _compute_dter, ensure_pack_dir


_CORPUS_RE = re.compile(r"/InhouseASR_2605/(?:en-US/)?([^/]+)/")


def _parent_key(audio_path: str) -> str:
    """Strip the ``#start:end`` time-range suffix to get the parent wav path."""
    return audio_path.split("#", 1)[0]


def _seg_start(audio_path: str) -> float:
    if "#" not in audio_path:
        return 0.0
    rng = audio_path.split("#", 1)[1]
    try:
        return float(rng.split(":", 1)[0])
    except Exception:
        return 0.0


def _corpus_from_path(audio_path: str) -> str:
    m = _CORPUS_RE.search(audio_path)
    return m.group(1) if m else "unknown"


def _iter_rows(eval_dir: str):
    import pyarrow.parquet as pq
    parts = sorted(bf.glob(os.path.join(eval_dir, "data_*.parquet")))
    if not parts:
        parts = sorted(bf.glob(os.path.join(eval_dir, "part-*.parquet")))
    print(f"Found {len(parts)} parquet parts", flush=True)
    for pf in parts:
        with bf.BlobFile(pf, "rb") as f:
            t = pq.ParquetFile(f).read(columns=["audio_path", "text", "response"]).to_pandas()
        for _, r in t.iterrows():
            yield r["audio_path"], r["text"], r["response"]


def aggregate(eval_dir: str) -> dict:
    ensure_pack_dir(None)  # warm up dotnet/TER
    # parent_key -> {"ref": str, "segs": [(start, hyp)], "corpus": str}
    by_parent: dict[str, dict] = {}
    for ap, text, resp in _iter_rows(eval_dir):
        pk = _parent_key(ap)
        d = by_parent.setdefault(pk, {"ref": text, "segs": [], "corpus": _corpus_from_path(ap)})
        d["segs"].append((_seg_start(ap), resp or ""))

    # Per-corpus accumulators.
    corp_err: dict[str, int] = defaultdict(int)
    corp_ref: dict[str, int] = defaultdict(int)
    corp_parents: dict[str, int] = defaultdict(int)

    per_parent_rows = []
    for pk, d in by_parent.items():
        segs = sorted(d["segs"], key=lambda x: x[0])
        hyp_full = " ".join(s for _, s in segs).strip()
        ref_full = _clean_ref(d["ref"] or "")
        n_err, n_ref, dter = _compute_dter(ref_full, hyp_full)
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

    # Print per-corpus.
    print(f"\n{'Corpus':<70} {'N_parent':>8} {'N_err':>8} {'N_ref':>8} {'DTER%':>8}")
    print("-" * 110)
    total_err = total_ref = total_parents = 0
    for c in sorted(corp_err):
        e, r, p = corp_err[c], corp_ref[c], corp_parents[c]
        total_err += e
        total_ref += r
        total_parents += p
        print(f"{c:<70} {p:>8} {e:>8} {r:>8} {(e/max(r,1)*100):>8.2f}")
    print("-" * 110)
    print(f"{'OVERALL':<70} {total_parents:>8} {total_err:>8} {total_ref:>8} {(total_err/max(total_ref,1)*100):>8.2f}")

    # Write summary.
    summary = {
        "eval_dir": eval_dir,
        "per_corpus": {c: {"n_parent": corp_parents[c], "n_err": corp_err[c],
                            "n_ref": corp_ref[c], "dter": corp_err[c] / max(corp_ref[c], 1)}
                       for c in corp_err},
        "overall": {"n_parent": total_parents, "n_err": total_err, "n_ref": total_ref,
                    "dter": total_err / max(total_ref, 1)},
    }
    out_summary = os.path.join(eval_dir, "aggregate_summary.json")
    with bf.BlobFile(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    out_details = os.path.join(eval_dir, "aggregate_per_parent.jsonl")
    with bf.BlobFile(out_details, "w") as f:
        for row in per_parent_rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {out_summary}\nWrote {out_details}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", required=True,
                   help="Eval output dir containing data_*.parquet (az:// or local).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    aggregate(args.eval_dir)

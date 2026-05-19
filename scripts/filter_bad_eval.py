#!/usr/bin/env python3
"""Filter eval rollout utterances using openasr_eval.

For each utterance in an eval result jsonl, run `openasr_eval(output, gts, extra_info={...})`
and keep utterances where any of {p_fmt, p_lang, p_bracket, p_repeat} is not 1.0.

The output jsonl has the original record plus a `source` field naming the original
dataset directory, and a `metrics` field with the openasr_eval result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.phimm.reward.asr_edge import openasr_eval  # noqa: E402

# For each metric, the value indicating a *failure* (problematic utterance).
# p_fmt / p_lang: 1.0 = pass, 0.0 = fail
# p_bracket / p_repeat: 0.0 = pass (clean), 1.0 = fail (problem detected)
FAIL_VALUE = {
    "p_fmt": 0.0,
    "p_lang": 0.0,
    "p_bracket": 1.0,
    "p_repeat": 1.0,
}
GATING_KEYS = tuple(FAIL_VALUE.keys())


def is_problematic(metrics: dict) -> dict:
    """Return per-key failure flags; an utt is problematic if any flag is True."""
    return {k: float(metrics.get(k, 0.0)) == FAIL_VALUE[k] for k in GATING_KEYS}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def process_eval_dir(root: Path, out_path: Path) -> dict:
    """Scan root/<source>/0.jsonl files; write filtered records to out_path."""
    total = 0
    kept = 0
    per_source: dict[str, dict[str, int]] = {}
    fail_counts = {k: 0 for k in GATING_KEYS}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out_f:
        for src_jsonl in sorted(root.glob("*/0.jsonl")):
            source = src_jsonl.parent.name
            per_source.setdefault(source, {"total": 0, "kept": 0})
            for rec in iter_jsonl(src_jsonl):
                total += 1
                per_source[source]["total"] += 1
                solution = rec.get("output", "") or ""
                gt = rec.get("gts", "") or ""
                language = rec.get("language", "English")
                try:
                    metrics = openasr_eval(
                        solution, gt, extra_info={"language": language}
                    )
                except Exception as exc:  # pragma: no cover
                    metrics = {"error": repr(exc)}
                    flags = {k: True for k in GATING_KEYS}
                else:
                    flags = is_problematic(metrics)
                if not any(flags.values()):
                    continue
                for k, v in flags.items():
                    if v:
                        fail_counts[k] += 1
                out_rec = dict(rec)
                out_rec["source"] = source
                out_rec["metrics"] = {
                    k: metrics.get(k) for k in (*GATING_KEYS, "score", "n_err", "n_ref")
                }
                out_rec["fail_flags"] = flags
                out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                kept += 1
                per_source[source]["kept"] += 1

    return {
        "total": total,
        "kept": kept,
        "fail_counts": fail_counts,
        "per_source": per_source,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "specs",
        nargs="+",
        help="NAME=DIR pairs. DIR contains <source>/0.jsonl. Output written to OUT_DIR/NAME.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="tmp/nonspeech_analysis/filtered",
        help="Directory to write <NAME>.jsonl files into.",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional path for a JSON summary across all specs.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for spec in args.specs:
        if "=" not in spec:
            raise SystemExit(f"Spec must be NAME=DIR, got: {spec}")
        name, _, raw = spec.partition("=")
        in_dir = Path(raw)
        if not in_dir.is_dir():
            raise SystemExit(f"Not a directory: {in_dir}")
        out_path = out_dir / f"{name}.jsonl"
        print(f"[{name}] {in_dir} -> {out_path}", flush=True)
        result = process_eval_dir(in_dir, out_path)
        summary[name] = {"out": str(out_path), **result}
        print(
            f"[{name}] kept {result['kept']}/{result['total']} utterances",
            flush=True,
        )
        print(f"    fail counts: {result['fail_counts']}")
        for src, s in sorted(result["per_source"].items()):
            print(f"    {src}: {s['kept']}/{s['total']}")

    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2))
        print(f"summary written to {args.summary}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

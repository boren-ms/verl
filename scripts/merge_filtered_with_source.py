#!/usr/bin/env python3
"""Merge original input data fields into filtered eval jsonls.

Takes the filtered jsonls produced by `filter_bad_eval.py` (which contain rollout
output fields only) and joins each record back to its original source
`data.jsonl` by `id`, producing records whose schema matches the legacy
`bad_fmt_lang.jsonl` files:

  <original input fields: dataset, text, id, audio_length_s, audio_path,
   sampling_rate, duration>
  <rollout output fields: input, output, gts, score, step, clean_output,
   data_source, reward, n_err, n_ref, [n_edge,] p_fmt, p_lang, p_bracket,
   keywords>

Drops the analysis-only fields (`source`, `metrics`, `fail_flags`) that
filter_bad_eval.py adds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# data_source -> local original data.jsonl path
# (matches openasr.yaml / openasr_ml.yaml).
SRC_EN_ROOT = Path("tmp/nonspeech_analysis/bad_case_build/en/src")
SRC_ML_ROOT = Path("tmp/nonspeech_analysis/bad_case_build/ml/src")

# audio_path prefix to prepend for EN sources (relative `audio/...` -> az://...).
EN_AUDIO_BASE = "az://orngwus2cresco/data/boren/data/openasr_jsonl"
# Dir-name overrides for EN data_source where the blob path differs.
EN_DIR_OVERRIDE = {
    "ls_clean": "ls-clean",
    "ls_other": "ls-other",
}


def en_source_jsonl(data_source: str) -> Path | None:
    name = EN_DIR_OVERRIDE.get(data_source, data_source)
    p = SRC_EN_ROOT / name / "data.jsonl"
    return p if p.exists() else None


def en_audio_full(data_source: str, rel: str) -> str:
    name = EN_DIR_OVERRIDE.get(data_source, data_source)
    if rel.startswith("audio/"):
        return f"{EN_AUDIO_BASE}/{name}/{rel}"
    return rel


def ml_source_jsonl(data_source: str) -> Path | None:
    # data_source like "fr_mls" -> mls__fr
    if "_" not in data_source:
        return None
    lang, _, subset = data_source.partition("_")
    p = SRC_ML_ROOT / f"{subset}__{lang}" / "data.jsonl"
    return p if p.exists() else None


# Rollout fields to keep in the merged record (in canonical order).
ROLLOUT_KEYS = [
    "input",
    "output",
    "gts",
    "score",
    "step",
    "clean_output",
    "data_source",
    "reward",
    "n_err",
    "n_ref",
    "n_edge",
    "p_fmt",
    "p_lang",
    "p_bracket",
    "keywords",
]

# Original input keys to keep (in canonical order).
ORIG_KEYS = [
    "dataset",
    "text",
    "id",
    "audio_length_s",
    "audio_path",
    "sampling_rate",
    "duration",
]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_src_index(jsonl_path: Path) -> dict:
    idx = {}
    for rec in iter_jsonl(jsonl_path):
        idx[rec["id"]] = rec
    return idx


def merge_one(filtered_path: Path, kind: str, out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src_cache: dict[str, dict] = {}
    total = 0
    written = 0
    missing_src = 0
    missing_id = 0
    per_source: dict[str, dict[str, int]] = {}

    with out_path.open("w", encoding="utf-8") as out_f:
        for rec in iter_jsonl(filtered_path):
            total += 1
            ds = rec.get("data_source")
            per_source.setdefault(ds, {"total": 0, "written": 0})
            per_source[ds]["total"] += 1
            if ds not in src_cache:
                src_path = en_source_jsonl(ds) if kind == "en" else ml_source_jsonl(ds)
                if src_path is None:
                    print(f"  [warn] no source jsonl for data_source={ds}", file=sys.stderr)
                    src_cache[ds] = {}
                else:
                    src_cache[ds] = build_src_index(src_path)
            src_idx = src_cache[ds]
            if not src_idx:
                missing_src += 1
                continue
            orig = src_idx.get(rec.get("id"))
            if orig is None:
                missing_id += 1
                continue
            # Build canonical merged record.
            merged = {}
            for k in ORIG_KEYS:
                if k in orig:
                    merged[k] = orig[k]
            if kind == "en" and "audio_path" in merged:
                merged["audio_path"] = en_audio_full(ds, merged["audio_path"])
            for k in ROLLOUT_KEYS:
                if k in rec:
                    merged[k] = rec[k]
            out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
            written += 1
            per_source[ds]["written"] += 1

    return {
        "total": total,
        "written": written,
        "missing_src": missing_src,
        "missing_id": missing_id,
        "per_source": per_source,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--en-filtered",
        default="tmp/nonspeech_analysis/filtered/eval_openasr_qwen.jsonl",
    )
    parser.add_argument(
        "--ml-filtered",
        default="tmp/nonspeech_analysis/filtered/eval_openasr_ml_qwen.jsonl",
    )
    parser.add_argument(
        "--out-dir", default="tmp/nonspeech_analysis/filtered_merged"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for kind, src in [("en", args.en_filtered), ("ml", args.ml_filtered)]:
        src_path = Path(src)
        if not src_path.is_file():
            print(f"[{kind}] skip (missing): {src_path}", file=sys.stderr)
            continue
        out_path = out_dir / src_path.name
        print(f"[{kind}] {src_path} -> {out_path}", flush=True)
        stats = merge_one(src_path, kind, out_path)
        print(
            f"[{kind}] wrote {stats['written']}/{stats['total']} "
            f"(missing_src={stats['missing_src']} missing_id={stats['missing_id']})"
        )
        for ds, s in sorted(stats["per_source"].items()):
            print(f"    {ds}: {s['written']}/{s['total']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Count <nonspeech> occurrences in ASR responses, binned by predicted language.

For each utterance:
  - response       = the model output (jsonl 'output' field)
  - raw_response   = same; we parse the predicted language from the `<lang=...>` tag
  - language bin   = the predicted language parsed from raw_response (falls back
                     to the top-level 'language' field, then 'unknown')

Reports per language: total utterances, # with <nonspeech>, ratio, and total
<nonspeech> tag occurrences.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

NONSPEECH_RE = re.compile(r"<nonspeech\b[^>]*>", re.IGNORECASE)
LANG_TAG_RE = re.compile(r"<lang=([^>]+)>", re.IGNORECASE)


def parse_pred_lang(raw_response: str) -> str | None:
    m = LANG_TAG_RE.search(raw_response or "")
    if not m:
        return None
    return m.group(1).strip()


def analyze(jsonl_paths: list[Path]) -> dict:
    # bin -> {"total": int, "with_ns": int, "ns_count": int}
    bins: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "with_ns": 0, "ns_count": 0}
    )
    overall = {"total": 0, "with_ns": 0, "ns_count": 0}

    for p in jsonl_paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                raw = rec.get("output", "") or ""
                pred_lang = parse_pred_lang(raw)
                if not pred_lang:
                    pred_lang = rec.get("language") or "unknown"

                ns_matches = NONSPEECH_RE.findall(raw)
                n_ns = len(ns_matches)

                b = bins[pred_lang]
                b["total"] += 1
                overall["total"] += 1
                if n_ns > 0:
                    b["with_ns"] += 1
                    overall["with_ns"] += 1
                    b["ns_count"] += n_ns
                    overall["ns_count"] += n_ns

    return {"by_lang": dict(bins), "overall": overall}


def print_report(name: str, result: dict) -> None:
    by_lang = result["by_lang"]
    overall = result["overall"]
    print(f"\n=== {name} ===")
    print(f"{'language':<20} {'total':>8} {'with_ns':>8} {'ratio':>8} {'ns_tags':>8}")
    print("-" * 56)
    # sort by descending ratio
    rows = sorted(
        by_lang.items(),
        key=lambda kv: (kv[1]["with_ns"] / kv[1]["total"]) if kv[1]["total"] else 0.0,
        reverse=True,
    )
    for lang, b in rows:
        ratio = (b["with_ns"] / b["total"] * 100.0) if b["total"] else 0.0
        print(
            f"{lang:<20} {b['total']:>8d} {b['with_ns']:>8d} {ratio:>7.2f}% {b['ns_count']:>8d}"
        )
    print("-" * 56)
    o_ratio = (overall["with_ns"] / overall["total"] * 100.0) if overall["total"] else 0.0
    print(
        f"{'OVERALL':<20} {overall['total']:>8d} {overall['with_ns']:>8d} "
        f"{o_ratio:>7.2f}% {overall['ns_count']:>8d}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Inputs in NAME=PATH form (or just PATH; name defaults to basename). "
            "PATH may be a directory or a .jsonl file."
        ),
    )
    ap.add_argument("--json-out", default=None, help="Optional JSON output path.")
    args = ap.parse_args(argv)

    pairs: list[tuple[str, str]] = []
    for spec in args.inputs:
        if "=" in spec:
            name, _, path = spec.partition("=")
        else:
            path = spec
            name = Path(spec).name
        pairs.append((name, path))

    all_results: dict[str, dict] = {}
    for name, inp in pairs:
        p = Path(inp)
        if p.is_dir():
            jsonls = sorted(p.rglob("*.jsonl"))
        else:
            jsonls = [p]
        if not jsonls:
            print(f"[warn] no jsonl found under {inp}", file=sys.stderr)
            continue
        result = analyze(jsonls)
        all_results[name] = result
        print_report(name, result)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

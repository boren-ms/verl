#!/usr/bin/env python3
"""Recompute mechanism evidence from the fixed retained-lineage snapshots.

No cluster calls and no job mutations. Parser demonstration executes the saved
source functions, with imports of the current workspace package removed.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import sys
import types
from importlib.metadata import version
from collections import defaultdict
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INPUT = ROOT / "reports/bcr_error_analysis_20261022T1811Z/sources"


def load(name):
    return json.loads((INPUT / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def stats(rows, key):
    vals = [r["metrics"][key] for r in rows if key in r["metrics"]]
    return dict(n=len(vals), mean=mean(vals) if vals else None,
                min=min(vals) if vals else None, max=max(vals) if vals else None)


def finite_sum(rows, key):
    vals = [r["metrics"][key] for r in rows]
    assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)
    return int(sum(vals))


def zero_ranges(rows, key):
    ranges, start, previous = [], None, None
    for row in rows:
        step = row["step"]
        zero = row["metrics"].get(key) == 0
        if start is not None and (not zero or step != previous + 1):
            ranges.append([start, previous])
            start = None
        if zero and start is None:
            start = step
        previous = step
    if start is not None:
        ranges.append([start, previous])
    return ranges


def isolated_source(label, suffix, ns=None, strip_recipe_imports=False):
    path = HERE / "sources" / f"{label}__{suffix}"
    tree = ast.parse(path.read_text(), filename=str(path))
    if strip_recipe_imports:
        tree.body = [node for node in tree.body if not
                     (isinstance(node, ast.ImportFrom) and
                      (node.module or "").startswith("recipe."))]
    module = types.ModuleType(f"_frozen_{label}_{suffix}")
    module.__dict__.update(ns or {})
    sys.modules[module.__name__] = module
    ns = module.__dict__
    exec(compile(tree, str(path), "exec"), ns)
    return ns


def parser_demonstration():
    response = isolated_source("bcr16_modern", "recipe__phimm__reward__asr_response.py")
    language = isolated_source("bcr16_modern", "recipe__phimm__utils__languages.py")
    measure = isolated_source(
        "bcr16_modern", "recipe__phimm__reward__asr_measure.py",
        {"get_hyp_text": response["get_hyp_text"],
         "parse_task_output": response["parse_task_output"],
         "get_language_code": language["get_language_code"]}, True)
    vote = isolated_source(
        "bcr16_modern", "recipe__phimm__reward__asr_vote.py",
        {name: measure[name] for name in ["_parse_response", "clip", "compute_delta", "reduce_scores"]}, True)
    body = "Guten Tag, meine Damen und Herren."
    # Synthetic mechanism demonstrations. These strings are not observed rows.
    strings = {
        "correct_header_with_verbatim": f"<src=German><tgt=German>\n<VERBATIM>\n{body}\n</VERBATIM>",
        "correct_header_without_verbatim": f"<src=German><tgt=German>\n{body}",
        "missing_gt_and_tgt": f"<src=German\n{body}",
        "wrong_tgt_language": f"<src=German><tgt=English>\n{body}",
    }
    examples = []
    for name, text in strings.items():
        parsed = response["parse_task_output"](text, version=2609)
        lang = measure["check_lang"](parsed, "German")
        hyp = response["get_hyp_text"](text, version=2609)
        examples.append(dict(name=name, synthetic=True, raw=text, parsed=parsed,
                             lang=lang, cleaned_hyp=hyp))
    assert all(x["cleaned_hyp"] == body for x in examples)
    assert [x["lang"]["acc"] for x in examples] == [1, 1, 0, 0]
    cfg = dict(delta="abs", beta=0.2, vote="delta", clip=[-10, 10])
    outcomes = []
    for sampled, greedy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        weighted, delta, clipped = vote["compute_measure_vote"]("lang", sampled, greedy, cfg)
        outcomes.append(dict(sampled=sampled, greedy=greedy, delta=delta,
                             clipped_vote=clipped, beta_weighted=weighted,
                             reduced_scalar_contribution=weighted / 1.6))
    return dict(examples=examples, lang_reward_cases=outcomes,
                interpretation="lang scores parsed target tags, not the language of body text; equal-zero sampled and greedy tags give zero differential contribution, but exploration can recover")


def main():
    lineage, r64 = load("r16_fix_lineage.json"), load("r64_history.json")
    rows = {"Fix": lineage["runs"]["baseline"]["training_rows"],
            "R16": lineage["runs"]["target"]["training_rows"]}
    segments64 = r64["segments"]
    rows["R64"] = sorted(
        [r for r in segments64["raysubmit_a5BkBYyJeRuPPHft"]["optimizer_rows"].values()
         if r["step"] <= 390] +
        list(segments64["bcr64-recover390-20261022T1535Z"]["optimizer_rows"].values()),
        key=lambda r: r["step"])
    for run, items in rows.items():
        assert [r["step"] for r in items] == list(range(1, len(items) + 1)), run
    budgets = []
    for cutoff in [440, 580]:
        for run, items in rows.items():
            chosen = [r for r in items if r["step"] <= cutoff]
            if len(chosen) < cutoff:
                budgets.append(dict(cutoff=cutoff, run=run, status="N/A", available_through=len(items)))
                continue
            fresh = cutoff * 128 if run == "Fix" else finite_sum(chosen, "replay/fresh_count")
            replayed = 0 if run == "Fix" else finite_sum(chosen, "replay/replayed")
            budgets.append(dict(cutoff=cutoff, run=run, status="complete", optimizer_rows=len(chosen),
                                fresh_presentations=fresh, replay_presentations=replayed,
                                raw_prompt_presentations=fresh + replayed,
                                fresh_vs_fix_percent=100 * fresh / (cutoff * 128),
                                basis="configured fresh128; complete optimizer coverage" if run == "Fix" else "sum retained replay/fresh_count and replay/replayed"))
    latest_budgets = {}
    segment_budgets = {}
    for run, items in rows.items():
        if run == "Fix":
            continue
        groups = defaultdict(list)
        for r in items:
            groups[r.get("segment_id", "original" if r["step"] <= 390 else "recovered390")].append(r)
        segment_budgets[run] = [dict(segment=k, first_step=min(r["step"] for r in v),
                                     last_step=max(r["step"] for r in v), rows=len(v),
                                     fresh=finite_sum(v, "replay/fresh_count"),
                                     replayed=finite_sum(v, "replay/replayed"),
                                     cold_steps=[r["step"] for r in v if r["metrics"]["replay/replayed"] == 0])
                                for k, v in groups.items()]
        latest_budgets[run] = dict(last_step=items[-1]["step"],
                                  fresh=finite_sum(items, "replay/fresh_count"),
                                  replayed=finite_sum(items, "replay/replayed"))
    keys = ["reward-core/word/mean", "reward-core/score/mean", "actor/kl_loss",
            "reward-core/lang/mean", "reward-aux/lang/min", "reward-aux/lang/max",
            "reward-delta/lang/mean", "reward-aux/d_lang/min", "reward-aux/d_lang/max",
            "reward-vote/lang/mean", "reward-aux/v_lang/min", "reward-aux/v_lang/max",
            "train/gen_kept_frac", "train/num_gen_batches", "replay/score_mean",
            "replay/reinsert_rate", "replay/buffer_size"]
    window = {}
    for run, items in rows.items():
        chosen = [r for r in items if 401 <= r["step"] <= 440]
        assert len(chosen) == 40
        counts = {source: sum(r["metrics"].get(f"data_source/{source}/num_samples", 0)
                              for r in chosen)
                  for source in ["openml", "ls_rare_verb", "earnings_fy27"]}
        total = sum(counts.values())
        info = dict(optimizer_steps=40, stats={k: stats(chosen, k) for k in keys},
                    final_rollout_source_rows=counts, final_rollout_rows=total,
                    final_rollout_source_percent={k: v / total * 100 for k, v in counts.items()})
        if run != "Fix":
            replayed, reinserted = (finite_sum(chosen, k) for k in ["replay/replayed", "replay/reinserted"])
            info.update(fresh=finite_sum(chosen, "replay/fresh_count"), replayed=replayed,
                        reinserted=reinserted, weighted_retention=reinserted / replayed,
                        deduplicated=finite_sum(chosen, "replay/deduplicated"))
        window[run] = info
    lang = {run: dict(n_rows=len(items), zero_mean_rows=sum(r["metrics"].get("reward-core/lang/mean") == 0 for r in items),
                      zero_mean_step_ranges=zero_ranges(items, "reward-core/lang/mean"))
            for run, items in rows.items()}
    inputs = {name: dict(path=str(INPUT / name), sha256=sha(INPUT / name))
              for name in ["r16_fix_lineage.json", "r64_history.json", "manifest.json"]}
    out = dict(input_sources=inputs, source_manifest_sha256=sha(HERE / "sources/code_manifest.json"),
               reproduction_runtime=dict(python=sys.version, jiwer=version("jiwer"),
                                         script_sha256=sha(Path(__file__))),
               generation_batch_metric_coverage={run: stats(items, "train/num_gen_batches")
                                                 for run, items in rows.items()},
               fixed_cutoff_budgets=budgets, latest_retained_budgets=latest_budgets,
               segment_budgets=segment_budgets, common_window_401_440=window,
               language_metric_coverage=lang, parser_demonstration=parser_demonstration(),
               limitations=["presentations are not unique audio or equal compute",
                            "source counts are final repeated rollout rows, not buffer composition",
                            "score_mean is current generation batch score, not resident buffer summary",
                            "R16 first50 retained updates used custom ReplayLossNormalizer",
                            "R64 replay size and capacity changed together relative to R16",
                            "unseeded grouping/shuffle means numeric loader cursor does not prove identical audio sequence",
                            "validation malformed header does not prove identical training raw output shape",
                            "current source file hash equality does not prove historical in-memory module identity"])
    write_json("mechanism_analysis.json", out)
    with (HERE / "prompt_budgets.csv").open("w") as f:
        fields = ["cutoff", "run", "status", "optimizer_rows", "fresh_presentations",
                  "replay_presentations", "raw_prompt_presentations", "fresh_vs_fix_percent", "available_through"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(budgets)
    print(json.dumps(dict(output=str(HERE / "mechanism_analysis.json"),
                          budgets=budgets, rows={k: len(v) for k, v in rows.items()},
                          language_zero_ranges=lang["R16"]), ensure_ascii=False))


if __name__ == "__main__":
    main()

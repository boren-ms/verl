"""Audit complete longform entity outputs and write dataset and corpus EER/DTER.

Run only after the exact Ray submission succeeds. Reads saved generations;
does not launch decoding, alter hypotheses, or substitute a scoring backend.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import blobfile as bf

from recipe.phimm.data.inhouse_dtest_enus import DATA_SOURCES, OUTPUT_ROOTS, SUITES, WORKBOOK_S2
from recipe.phimm.data.inhouse_entity import OUTPUT_ROOT


def read_json(uri):
    with bf.BlobFile(uri, "r") as stream:
        return json.load(stream)


def audit_rows(rows, expected, *, context_mode="independent", include_eer=True, expected_prompt=None, prefix=None):
    """Check every original parent/clip/reference before aggregating counts."""
    if context_mode not in ("independent", "text_h1"):
        raise ValueError(f"Unsupported context audit mode: {context_mode}")
    expected = {row["id"]: row for row in expected}
    seen = set()
    metric_names = ("eer", "dter") if include_eer else ("dter",)
    count_keys = [f"{metric}_n_{kind}" for metric in metric_names for kind in ("err", "ref")]
    counts = dict.fromkeys(("parents", "segments", *count_keys), 0)
    for row in rows:
        detail = json.loads(row["asr_detail"])
        info = detail.get("extra_info", detail.get("parent_info", {}))
        uid = info["id"]
        if uid not in expected or uid in seen:
            raise ValueError(f"Unexpected or duplicate result parent: {uid}")
        source = expected[uid]
        if row["asr_data_source"] != source["data_source"]:
            raise ValueError(f"Changed datasource: {uid}")
        if row["gts"] != source["text"] or info["entity_transcription"] != source["entity_transcription"]:
            raise ValueError(f"Changed full-parent reference: {uid}")
        segments = detail.get("asr_segments", detail.get("segment_rows", []))
        paths = [segment["audio_path"] for segment in segments]
        if paths != source["audio_path"] or len(detail["segment_outputs"]) != len(paths):
            raise ValueError(f"Incomplete or reordered decoded clips: {uid}")
        if expected_prompt is not None:
            messages = [{"role": "user", "content": expected_prompt}]
            if any(segment.get("prompt", detail.get("prompt")) != messages for segment in segments):
                raise ValueError(f"Changed or missing base prompt/language hint: {uid}")
        contexts = [""] * len(paths)
        if context_mode == "text_h1":
            from recipe.phimm.agentic.context_prompt import ContextPrompt

            context = ContextPrompt("text", prefix=prefix, max_history=1)
            contexts = []
            for text in detail["segment_outputs"]:
                contexts.append(context.prompt)
                context.observe(text)
        if detail["previous_contexts"] != contexts:
            raise ValueError(f"Unexpected decoding context: {uid}")
        seen.add(uid)
        counts["parents"] += 1
        counts["segments"] += len(paths)
        for key in count_keys:
            value = row[key]
            if not isinstance(value, int | float) or not math.isfinite(value) or value < 0 or value != int(value):
                raise ValueError(f"Invalid count {key}: {uid}")
            counts[key] += int(value)
        if include_eer and row["eer_n_err"] > row["eer_n_ref"]:
            raise ValueError(f"Entity errors exceed reference entities: {uid}")
    if seen != set(expected):
        raise ValueError(f"Missing results for {len(set(expected) - seen)} parents")
    return counts


def ratios(counts):
    return {
        metric: counts[f"{metric}_n_err"] / counts[f"{metric}_n_ref"] if counts[f"{metric}_n_ref"] else None
        for metric in ("eer", "dter")
        if f"{metric}_n_ref" in counts
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True, help="Run's val_data_gen directory (local or Blob)")
    parser.add_argument("--manifest-root")
    parser.add_argument("--suite", choices=("eer", "ter", "legacy_entity_v3"), default="eer")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--context-mode", choices=("independent", "text_h1"), default="independent")
    parser.add_argument("--context-prefix", help="Custom context instruction; omitted uses the default text prefix")
    parser.add_argument("--model-label", default="2611 V1")
    parser.add_argument("--prompt-style", choices=("verb", "lex", "display"), default="display")
    parser.add_argument("--language-hint", choices=("English", "none"), help="Audit every segment's saved base prompt")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expected_prompt = None
    if args.language_hint is not None:
        from recipe.phimm.data.prompts import get_task_prompt

        expected_prompt = get_task_prompt(
            style=args.prompt_style, language="English", lang_hint=args.language_hint == "English"
        )
    workbook = args.suite != "legacy_entity_v3"
    include_eer = args.suite != "ter"
    metric_names = ("eer", "dter") if include_eer else ("dter",)
    root = args.manifest_root or (OUTPUT_ROOTS[args.suite] if workbook else OUTPUT_ROOT)
    paths = [f"{root}/{name}.jsonl" for name in SUITES[args.suite]] if workbook else sorted(bf.glob(f"{root}/*.jsonl"))
    metrics = read_json(f"{args.result_root}/{args.step}.metrics.json")["metrics"]
    result = []
    totals = None
    for path in paths:
        with bf.BlobFile(path, "r") as stream:
            expected = [json.loads(line) for line in stream if line.strip()]
        source = expected[0]["data_source"]
        uri = f"{args.result_root}/{source}/{args.step}.jsonl"
        with bf.BlobFile(uri, "r") as stream:
            counts = audit_rows(
                (json.loads(line) for line in stream if line.strip()),
                expected,
                context_mode=args.context_mode,
                include_eer=include_eer,
                expected_prompt=expected_prompt,
                prefix=args.context_prefix,
            )
        values = ratios(counts)
        for metric in metric_names:
            saved = metrics[f"val-core/{source}/{metric}_p_err/mean@1"]
            if values[metric] is None or not math.isclose(values[metric], saved, abs_tol=1e-12):
                raise ValueError(f"Persisted metrics differ from detail counts: {source}/{metric}")
        result.append({"dataset": source, **counts, **values, "detail_path": uri, "manifest": path})
        print(f"Audited {source}: {counts['parents']} recordings / {counts['segments']} clips", flush=True)
        if totals is None:
            totals = dict.fromkeys(counts, 0)
        for key, value in counts.items():
            totals[key] += value
    if workbook:
        if [row["dataset"] for row in result] != list(DATA_SOURCES[args.suite].values()):
            raise ValueError("Exact five-dataset workbook coverage was not verified")
    elif (len(result), totals["parents"], totals["segments"]) != (31, 755, 25978):
        raise ValueError("Legacy en-US-entity-v3 coverage was not verified")
    summary = {
        "result_root": args.result_root,
        "step": args.step,
        "model_label": args.model_label,
        "suite": args.suite,
        "context_mode": args.context_mode,
        "context_prefix": args.context_prefix,
        "language_hint": args.language_hint,
        "prompt_style": args.prompt_style,
        "datasets": result,
        "total": {**totals, **ratios(totals)},
        "macro": {metric: sum(row[metric] for row in result) / len(result) for metric in metric_names},
        "audit": (
            f"{totals['parents']} original references, ordered {totals['segments']} clips, "
            + (
                "empty previous contexts"
                if args.context_mode == "independent"
                else "exact previous-output text h1 contexts"
            )
            + ", saved metric agreement"
            + (f", every segment base prompt verified ({args.language_hint} language hint)" if expected_prompt else "")
        ),
    }
    if workbook:
        primary_metric = "eer" if include_eer else "dter"
        summary["primary_metric"] = primary_metric
        summary["primary_" + primary_metric] = summary["macro"][primary_metric]
        summary["primary_aggregation"] = "Equal-weight mean of exactly five workbook datasets"
    if args.suite == "eer":
        summary["workbook_2611_s2"] = {
            "datasets": dict(zip(DATA_SOURCES["eer"].values(), WORKBOOK_S2, strict=True)),
            "macro_eer_percent": sum(WORKBOOK_S2) / len(WORKBOOK_S2),
            "reported_macro_eer_percent": 23.15,
            "source": "User-provided EER worksheet, 2611-s2 / shadow run column",
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.output_dir / "datasets.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    lines = [
        f"# {args.model_label} — Inhouse en-US {args.suite} evaluation",
        "",
        f"Language hint audited: {args.language_hint or 'unspecified (not checked)'}.",
        "Uses the existing VAD audio clips and concatenates segment outputs in recording order.",
        (
            "Clips decode independently with empty preceding contexts."
            if args.context_mode == "independent"
            else "Text context uses only the preceding clip's cleaned generated output; history resets per recording."
        ),
        (
            "Workbook EER uses original DisplayTranscription and SpeechInsight --nondisfluency."
            if args.suite == "eer"
            else (
                "Workbook TER uses original DisplayTranscription and disfluencytolerant normalization."
                if args.suite == "ter"
                else "Legacy verbatim EER uses original tagged Transcription."
            )
        ),
        "DTER uses DisplayTranscription. Corpus rates divide summed counts; macro rates weight subsets equally.",
        "",
        "| Dataset | Recordings | Segments | Errors / reference units | "
        + ("EER | DTER |" if include_eer else "DTER |"),
        "|---|---:|---:|---:|---:|" + ("---:|" if include_eer else ""),
    ]
    for row in [*result, {"dataset": "TOTAL (count weighted)", **summary["total"]}]:
        primary_metric = "eer" if include_eer else "dter"
        lines.append(
            f"| {row['dataset']} | {row['parents']} | {row['segments']} | "
            f"{row[primary_metric + '_n_err']} / {row[primary_metric + '_n_ref']} | "
            + (f"{row['eer']:.2%} | " if include_eer else "")
            + f"{row['dter']:.2%} |"
        )
    if args.suite == "ter":
        lines += ["", f"Primary five-dataset mean DTER: **{summary['primary_dter']:.4%}**."]
    if args.suite == "eer":
        lines += [
            "",
            "## Workbook comparison",
            "",
            "Primary result: equal-weight mean across these five datasets.",
            "",
            "| Dataset | Workbook 2611 S2 EER | This run EER | Difference (pp) |",
            "|---|---:|---:|---:|",
        ]
        for row, baseline in zip(result, WORKBOOK_S2, strict=True):
            lines.append(
                f"| {row['dataset']} | {baseline:.2f}% | {row['eer']:.4%} | {100 * row['eer'] - baseline:+.4f} |"
            )
        baseline = sum(WORKBOOK_S2) / len(WORKBOOK_S2)
        current = summary["primary_eer"]
        lines.append(
            f"| **Five-dataset mean** | **{baseline:.3f}%** | **{current:.4%}** | **{100 * current - baseline:+.4f}** |"
        )
    lines.extend(["", f"Audit: {summary['audit']}", "", f"Results: `{args.result_root}`", ""])
    (args.output_dir / "report.md").write_text("\n".join(lines))
    print(json.dumps({"total": summary["total"], "macro": summary["macro"]}, indent=2))


if __name__ == "__main__":
    main()

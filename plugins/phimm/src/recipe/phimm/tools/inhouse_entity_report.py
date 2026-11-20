"""Audit complete longform entity outputs and write count-weighted EER/DTER.

Run only after the exact Ray submission succeeds. Reads saved generations;
does not launch decoding, alter hypotheses, or substitute a scoring backend.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import blobfile as bf
from recipe.phimm.data.inhouse_entity import OUTPUT_ROOT


def read_json(uri):
    with bf.BlobFile(uri, "r") as stream:
        return json.load(stream)


def audit_rows(rows, expected, *, context_mode="independent"):
    """Check every original parent/clip/reference before aggregating counts."""
    if context_mode not in ("independent", "text_h1"):
        raise ValueError(f"Unsupported context audit mode: {context_mode}")
    expected = {row["id"]: row for row in expected}
    seen = set()
    counts = dict.fromkeys(("parents", "segments", "eer_n_err", "eer_n_ref", "dter_n_err", "dter_n_ref"), 0)
    for row in rows:
        detail = json.loads(row["asr_detail"])
        info = detail["parent_info"]
        uid = info["id"]
        if uid not in expected or uid in seen:
            raise ValueError(f"Unexpected or duplicate result parent: {uid}")
        source = expected[uid]
        if row["asr_data_source"] != source["data_source"]:
            raise ValueError(f"Changed datasource: {uid}")
        if row["gts"] != source["text"] or info["entity_transcription"] != source["entity_transcription"]:
            raise ValueError(f"Changed full-parent reference: {uid}")
        paths = [segment["audio_path"] for segment in detail["segment_rows"]]
        if paths != source["audio_path"] or len(detail["segment_outputs"]) != len(paths):
            raise ValueError(f"Incomplete or reordered decoded clips: {uid}")
        contexts = [""] * len(paths)
        if context_mode == "text_h1":
            from recipe.phimm.reward.asr_response import get_hyp_text

            contexts[1:] = [get_hyp_text(text, version=2609) for text in detail["segment_outputs"][:-1]]
        if detail["previous_contexts"] != contexts:
            raise ValueError(f"Unexpected decoding context: {uid}")
        seen.add(uid)
        counts["parents"] += 1
        counts["segments"] += len(paths)
        for key in ("eer_n_err", "eer_n_ref", "dter_n_err", "dter_n_ref"):
            value = row[key]
            if not isinstance(value, int | float) or not math.isfinite(value) or value < 0 or value != int(value):
                raise ValueError(f"Invalid count {key}: {uid}")
            counts[key] += int(value)
        if row["eer_n_err"] > row["eer_n_ref"]:
            raise ValueError(f"Entity errors exceed reference entities: {uid}")
    if seen != set(expected):
        raise ValueError(f"Missing results for {len(set(expected) - seen)} parents")
    return counts


def ratios(counts):
    return {
        "eer": counts["eer_n_err"] / counts["eer_n_ref"] if counts["eer_n_ref"] else None,
        "dter": counts["dter_n_err"] / counts["dter_n_ref"] if counts["dter_n_ref"] else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True, help="Run's val_data_gen directory (local or Blob)")
    parser.add_argument("--manifest-root", default=OUTPUT_ROOT)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--context-mode", choices=("independent", "text_h1"), default="independent")
    parser.add_argument("--model-label", default="2611 V1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = read_json(f"{args.result_root}/{args.step}.metrics.json")["metrics"]
    result = []
    totals = None
    for path in sorted(bf.glob(f"{args.manifest_root}/*.jsonl")):
        with bf.BlobFile(path, "r") as stream:
            expected = [json.loads(line) for line in stream if line.strip()]
        source = expected[0]["data_source"]
        uri = f"{args.result_root}/{source}/{args.step}.jsonl"
        with bf.BlobFile(uri, "r") as stream:
            counts = audit_rows(
                (json.loads(line) for line in stream if line.strip()), expected, context_mode=args.context_mode
            )
        values = ratios(counts)
        for metric in ("eer", "dter"):
            saved = metrics[f"val-core/{source}/{metric}_p_err/mean@1"]
            if values[metric] is None or not math.isclose(values[metric], saved, abs_tol=1e-12):
                raise ValueError(f"Persisted metrics differ from detail counts: {source}/{metric}")
        result.append({"dataset": source, **counts, **values, "detail_path": uri, "manifest": path})
        print(f"Audited {source}: {counts['parents']} recordings / {counts['segments']} clips", flush=True)
        if totals is None:
            totals = dict.fromkeys(counts, 0)
        for key, value in counts.items():
            totals[key] += value
    if (len(result), totals["parents"], totals["segments"]) != (31, 755, 25978):
        raise ValueError("Full en-US-entity-v3 coverage was not verified")
    summary = {
        "result_root": args.result_root,
        "step": args.step,
        "model_label": args.model_label,
        "context_mode": args.context_mode,
        "datasets": result,
        "total": {**totals, **ratios(totals)},
        "macro": {metric: sum(row[metric] for row in result) / len(result) for metric in ("eer", "dter")},
        "audit": (
            "755 original references, ordered 25978 clips, "
            + (
                "empty previous contexts"
                if args.context_mode == "independent"
                else "exact previous-output text h1 contexts"
            )
            + ", saved metric agreement"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.output_dir / "datasets.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    lines = [
        f"# {args.model_label} — Inhouse en-US-entity-v3 longform evaluation",
        "",
        "Uses the existing VAD audio clips and concatenates segment outputs in recording order.",
        (
            "Clips decode independently with empty preceding contexts."
            if args.context_mode == "independent"
            else "Text context uses only the preceding clip's cleaned generated output; history resets per recording."
        ),
        "Official SpeechInsight verbatim EER uses original tagged Transcription; DTER uses DisplayTranscription.",
        "Corpus rates divide summed errors by summed reference units; macro rates weight the 31 subsets equally.",
        "",
        "| Dataset | Recordings | Segments | Entity errors / entities | EER | DTER |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in [*result, {"dataset": "TOTAL (count weighted)", **summary["total"]}]:
        lines.append(
            f"| {row['dataset']} | {row['parents']} | {row['segments']} | "
            f"{row['eer_n_err']} / {row['eer_n_ref']} | {row['eer']:.2%} | {row['dter']:.2%} |"
        )
    lines.extend(["", f"Audit: {summary['audit']}", "", f"Results: `{args.result_root}`", ""])
    (args.output_dir / "report.md").write_text("\n".join(lines))
    print(json.dumps({"total": summary["total"], "macro": summary["macro"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build independent segment subsets from grouped OpenLong evaluation dumps.

References are inferred from the evaluated hypotheses; they are not timestamped
ground truth. Never treat the resulting selected set as an unbiased evaluation.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from recipe.phimm.reward.asr_eval import measure_openasr_en_wer
from scripts.align_long_audio_transcript import assign_reference_words


DATASETS = ("earnings21", "earnings22", "coraal", "earnings22_cleaned_aa")
EXPECTED = {"earnings21": (5552, 44), "earnings22": (17099, 125),
            "coraal": (22041, 271), "earnings22_cleaned_aa": (273, 6)}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset(source, destination, dataset, limit, verify_parent_scores=False):
    groups = defaultdict(list)
    references = {}
    count = 0
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("data_source") != dataset:
                raise ValueError(f"Unexpected source in {source}")
            parent = row["parent_audio_path"]
            ref = row.pop("gts")
            if parent in references and references[parent] != ref:
                raise ValueError(f"Inconsistent parent reference: {parent}")
            references[parent] = ref
            groups[parent].append(row)
            count += 1
    if (count, len(groups)) != EXPECTED[dataset]:
        raise ValueError(f"Incomplete {dataset}: {count} segments, {len(groups)} parents")

    candidates = []
    for parent, rows in groups.items():
        rows.sort(key=lambda r: (r["seg_index"], r["seg_start"]))
        n = len(rows)
        if [r["seg_index"] for r in rows] != list(range(n)) or any(r["n_segments"] != n for r in rows):
            raise ValueError(f"Incomplete or duplicate segments: {parent}")
        hyps = [r["clean_output"] for r in rows]
        if not all(isinstance(h, str) for h in hyps):
            raise ValueError(f"Missing clean hypothesis: {parent}")
        ref = references[parent]
        aligned = assign_reference_words(ref, hyps)
        parent_n_err = [r["n_err"] for r in rows if r.get("n_err") is not None]
        if len(parent_n_err) != 1:
            raise ValueError(f"Missing or duplicate grouped scores: {parent}")
        if verify_parent_scores:
            parent_score = measure_openasr_en_wer("\n".join(hyps), ref)
            if parent_n_err[0] != parent_score["n_err"]:
                raise ValueError(f"Grouped score mismatch: {parent}")
        for row, transcript, hyp in zip(rows, aligned, hyps):
            score = measure_openasr_en_wer(hyp, transcript)
            candidates.append({
                "id": f"{dataset}:{hashlib.sha256(parent.encode()).hexdigest()[:16]}:{row['seg_index']}",
                "audio_path": row["audio_path"],
                "text": transcript,
                "language": row.get("language") or "English",
                "source_id": row["id"],
                "parent_audio_path": parent,
                "seg_index": row["seg_index"],
                "seg_start": row["seg_start"],
                "seg_end": row["seg_end"],
                "n_segments": n,
                "baseline_hypothesis": hyp,
                "baseline_n_err": score["n_err"],
                "baseline_n_ref": score["n_ref"],
                "baseline_wer": score["wer"],
            })
    candidates.sort(key=lambda r: (-r["baseline_n_err"], -r["baseline_n_ref"], r["audio_path"]))
    if len(candidates) < limit:
        raise ValueError(f"Not enough segments in {dataset}")
    selected = candidates[:limit]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"source_sha256": sha256(source), "output_sha256": sha256(destination),
            "segments": count, "parents": len(groups), "selected": len(selected),
            "empty_inferred_references": sum(not row["text"].strip() for row in selected),
            "min_selected_errors": selected[-1]["baseline_n_err"],
            "max_selected_errors": selected[0]["baseline_n_err"]}


def summarize_existing(source, destination, dataset, limit):
    with destination.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != limit or len({row["id"] for row in rows}) != limit:
        raise ValueError(f"Invalid existing output: {destination}")
    if rows != sorted(rows, key=lambda r: (-r["baseline_n_err"], -r["baseline_n_ref"], r["audio_path"])):
        raise ValueError(f"Output is not ranked: {destination}")
    count, parents = EXPECTED[dataset]
    return {"source_sha256": sha256(source), "output_sha256": sha256(destination),
            "segments": count, "parents": parents, "selected": len(rows),
            "empty_inferred_references": sum(not row["text"].strip() for row in rows),
            "min_selected_errors": rows[-1]["baseline_n_err"],
            "max_selected_errors": rows[0]["baseline_n_err"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--verify-parent-scores", action="store_true",
                        help="Recompute each full-recording score (computationally expensive).")
    parser.add_argument("--reuse-existing", action="store_true",
                        help="Summarize already-built outputs instead of rebuilding them.")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    summary = {"_provenance": {
        "evaluation_config": "recipe/phimm/config/v2611_dev/eval/eval_2609r2b_openlong_seg_verb_langhint_zip.yaml",
        "evaluation_outputs": "az://orngwus2cresco/data/boren/outputs/v2611_eval/eval_2609r2b_openlong_seg_verb_langhint_zip/val_data_gen/",
        "reference_method": "Monotonic hypothesis-based alignment of each full parent transcript to its segment outputs; segment boundaries are approximate.",
        "empty_reference_warning": "An empty inferred reference is an alignment outcome, not proof of silence; inspect those clips before treating them as labeled data.",
        "ranking": "Descending compound-aware OpenASR English word errors; then descending reference length and audio path.",
        "parent_scores_recomputed": args.verify_parent_scores,
        "selection_bias": "Selected on the evaluated model's own errors; unsuitable as an unbiased evaluation set.",
    }}
    for dataset in DATASETS:
        print(f"Processing {dataset}...", flush=True)
        source = args.input_dir / f"{dataset}.jsonl"
        destination = args.output_dir / f"{dataset}.jsonl"
        if args.reuse_existing:
            summary[dataset] = summarize_existing(source, destination, dataset, args.limit)
        else:
            summary[dataset] = build_dataset(source, destination, dataset, args.limit,
                                             verify_parent_scores=args.verify_parent_scores)
        print(json.dumps({dataset: summary[dataset]}), flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    examples = {}
    for dataset in DATASETS:
        with (args.output_dir / f"{dataset}.jsonl").open(encoding="utf-8") as handle:
            examples[dataset] = [json.loads(next(handle)) for _ in range(min(2, args.limit))]
    (args.output_dir / "examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

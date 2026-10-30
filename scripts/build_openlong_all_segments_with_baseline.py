#!/usr/bin/env python3
"""Attach 2609r2b hypotheses to every OpenLong segment.

Segment references are inferred by aligning parent text to model hypotheses;
they are not time-aligned ground truth and require review before use as labels.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from recipe.phimm.reward.asr_eval import measure_openasr_en_wer
from scripts.align_long_audio_transcript import assign_reference_words
from scripts.build_openlong_high_error_segments import DATASETS, EXPECTED, sha256


def build_dataset(source: Path, destination: Path, dataset: str) -> dict:
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
            reference = row.pop("gts")
            if parent in references and references[parent] != reference:
                raise ValueError(f"Inconsistent parent reference: {parent}")
            references[parent] = reference
            groups[parent].append(row)
            count += 1
    if (count, len(groups)) != EXPECTED[dataset]:
        raise ValueError(f"Incomplete {dataset}: {count} segments, {len(groups)} parents")

    parent_scores = {}
    for parent, rows in groups.items():
        rows.sort(key=lambda row: row["seg_index"])
        n = len(rows)
        if [row["seg_index"] for row in rows] != list(range(n)) or any(row["n_segments"] != n for row in rows):
            raise ValueError(f"Incomplete or duplicate segments: {parent}")
        scores = [(row["n_err"], row["n_ref"]) for row in rows if row.get("n_err") is not None]
        if len(scores) != 1:
            raise ValueError(f"Missing or duplicate grouped score: {parent}")
        parent_scores[parent] = scores[0]
    ranked = sorted(groups, key=lambda p: (-parent_scores[p][0], -parent_scores[p][1], p))
    ranks = {parent: index for index, parent in enumerate(ranked, 1)}

    destination.parent.mkdir(parents=True, exist_ok=True)
    empty_reference_count = 0
    written = 0
    with destination.open("w", encoding="utf-8") as handle:
        for parent in sorted(groups):
            rows = groups[parent]
            hypotheses = [row["clean_output"] for row in rows]
            if not all(isinstance(hyp, str) for hyp in hypotheses):
                raise ValueError(f"Missing hypothesis: {parent}")
            aligned = assign_reference_words(references[parent], hypotheses)
            parent_n_err, parent_n_ref = parent_scores[parent]
            for row, transcript, hypothesis in zip(rows, aligned, hypotheses, strict=True):
                score = measure_openasr_en_wer(hypothesis, transcript)
                parent_hash = hashlib.sha256(parent.encode()).hexdigest()[:16]
                record = {
                    "id": f"{dataset}:{parent_hash}:{row['seg_index']}",
                    "audio_path": row["audio_path"], "text": transcript,
                    "language": row.get("language") or "English", "source_id": row["id"],
                    "parent_audio_path": parent, "parent_rank": ranks[parent],
                    "parent_baseline_n_err": parent_n_err, "parent_baseline_n_ref": parent_n_ref,
                    "seg_index": row["seg_index"], "seg_start": row["seg_start"],
                    "seg_end": row["seg_end"], "n_segments": len(rows),
                    "baseline_hypothesis": hypothesis, "baseline_n_err": score["n_err"],
                    "baseline_n_ref": score["n_ref"], "baseline_wer": score["wer"],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                empty_reference_count += int(not transcript.strip())
    if written != count:
        raise ValueError(f"Output count mismatch: {written} != {count}")
    return {"input_sha256": sha256(source), "output_sha256": sha256(destination),
            "segments": written, "parents": len(groups),
            "empty_inferred_references": empty_reference_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = {"_provenance": {
        "evaluation_outputs": "az://orngwus2cresco/data/boren/outputs/v2611_eval/eval_2609r2b_openlong_seg_verb_langhint_zip/val_data_gen/",
        "selection": "Every segment in every source dataset, ordered by parent and segment index.",
        "reference_warning": "Segment text is inferred from baseline hypotheses; empty inferred references are not proof of silence.",
        "parent_score_warning": "Saved full-recording errors can be inflated by normalization across segment boundaries.",
    }}
    for dataset in DATASETS:
        print(f"Processing {dataset}...", flush=True)
        summary[dataset] = build_dataset(args.input_dir / f"{dataset}.jsonl",
                                         args.output_dir / f"{dataset}.jsonl", dataset)
        print(json.dumps({dataset: summary[dataset]}), flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

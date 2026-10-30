#!/usr/bin/env python3
"""Keep every OpenLong clip from the two highest-error parents per dataset.

Parent ranking uses saved full-recording n_err. Clip references are inferred
from the same model's hypotheses and should be reviewed before training.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from recipe.phimm.reward.asr_eval import measure_openasr_en_wer
from recipe.phimm.utils.open_asr_normalizer.hf_english_normalizer import _HFEnglishTextNormalizer
from scripts.align_long_audio_transcript import assign_reference_words
from scripts.build_openlong_high_error_segments import DATASETS, EXPECTED, sha256


def build(source, destination, dataset):
    parents = defaultdict(lambda: {"indexes": set(), "expected": None, "scores": []})
    count = 0
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("data_source") != dataset:
                raise ValueError(f"Unexpected source in {source}")
            parent = parents[row["parent_audio_path"]]
            index = row["seg_index"]
            if index in parent["indexes"]:
                raise ValueError("Duplicate segment index")
            parent["indexes"].add(index)
            if parent["expected"] is not None and parent["expected"] != row["n_segments"]:
                raise ValueError("Inconsistent parent segment count")
            parent["expected"] = row["n_segments"]
            if row.get("n_err") is not None:
                parent["scores"].append((row["n_err"], row["n_ref"]))
            count += 1
    if (count, len(parents)) != EXPECTED[dataset]:
        raise ValueError(f"Incomplete {dataset}: {count} segments, {len(parents)} parents")
    for path, group in parents.items():
        if group["indexes"] != set(range(group["expected"])) or len(group["scores"]) != 1:
            raise ValueError(f"Incomplete parent group or score: {path}")
    ranked = sorted(parents, key=lambda p: (-parents[p]["scores"][0][0],
                                            -parents[p]["scores"][0][1], p))[:2]
    groups = {parent: [] for parent in ranked}
    refs = {}
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            parent = row["parent_audio_path"]
            if parent not in groups:
                continue
            ref = row.pop("gts")
            if parent in refs and refs[parent] != ref:
                raise ValueError(f"Inconsistent reference: {parent}")
            refs[parent] = ref
            groups[parent].append(row)

    candidates = []
    normalization = []
    normalizer = _HFEnglishTextNormalizer()
    for rank, parent in enumerate(ranked, 1):
        rows = sorted(groups[parent], key=lambda r: r["seg_index"])
        hyps = [r["clean_output"] for r in rows]
        if not all(isinstance(h, str) for h in hyps):
            raise ValueError(f"Missing hypothesis: {parent}")
        whole_words = len(normalizer("\n".join(hyps)).split())
        segment_words = sum(len(normalizer(h).split()) for h in hyps)
        normalization.append({"whole_hyp_words": whole_words, "sum_per_segment_hyp_words": segment_words,
                              "large_cross_segment_normalization_loss": whole_words < 0.8 * segment_words})
        aligned = assign_reference_words(refs[parent], hyps)
        n_err, n_ref = parents[parent]["scores"][0]
        group = []
        for row, text, hyp in zip(rows, aligned, hyps, strict=True):
            score = measure_openasr_en_wer(hyp, text)
            group.append({
                "id": f"{dataset}:{hashlib.sha256(parent.encode()).hexdigest()[:16]}:{row['seg_index']}",
                "audio_path": row["audio_path"], "text": text,
                "language": row.get("language") or "English", "source_id": row["id"],
                "parent_audio_path": parent, "parent_rank": rank,
                "parent_baseline_n_err": n_err, "parent_baseline_n_ref": n_ref,
                "seg_index": row["seg_index"], "seg_start": row["seg_start"],
                "seg_end": row["seg_end"], "n_segments": len(rows),
                "baseline_hypothesis": hyp, "baseline_n_err": score["n_err"],
                "baseline_n_ref": score["n_ref"], "baseline_wer": score["wer"],
            })
        candidates.append(group)

    selected = candidates[0] + candidates[1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    details = [{"parent": p, "parent_rank": i + 1,
                "n_err": parents[p]["scores"][0][0], "n_ref": parents[p]["scores"][0][1],
                "total_segments": len(groups[p]), "selected_segments": len(candidates[i]),
                "normalization_check": normalization[i]}
               for i, p in enumerate(ranked)]
    examples = {str(i + 1): sorted(candidates[i],
                 key=lambda r: (-r["baseline_n_err"], -r["baseline_n_ref"], r["audio_path"]))[:2]
                for i in range(2)}
    return {"input_sha256": sha256(source), "output_sha256": sha256(destination),
            "selected": len(selected), "empty_inferred_references": sum(not r["text"].strip() for r in selected),
            "parents": details}, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = {"_provenance": {
        "evaluation_outputs": "az://orngwus2cresco/data/boren/outputs/v2611_eval/eval_2609r2b_openlong_seg_verb_langhint_zip/val_data_gen/",
        "parent_ranking": "saved grouped full-recording n_err descending; ties by n_ref descending and parent path",
        "segment_selection": "all segments from each of the two selected parents, in segment order",
        "reference_warning": "Segment references are inferred from model hypotheses, not time-aligned ground truth; empty references do not establish silence.",
        "parent_score_warning": "Saved full-recording errors can be inflated when unmatched markup causes normalization to remove text across segment boundaries; inspect normalization_check.",
    }}
    examples = {}
    for dataset in DATASETS:
        print(f"Processing {dataset}...", flush=True)
        summary[dataset], examples[dataset] = build(args.input_dir / f"{dataset}.jsonl",
                                                    args.output_dir / f"{dataset}.jsonl", dataset)
        print(json.dumps({dataset: summary[dataset]}), flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

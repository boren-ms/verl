#!/usr/bin/env python3
"""Reconcile OpenLong segment text using test reference, baseline and Qwen.

Preserves every original reference token exactly once, in original order.
Baseline exact matches act as segment-local anchors; Qwen placement is accepted
only near those anchors or the prior seed, and never from pathological windows.
These are machine-generated candidates, not verified ground truth.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from recipe.phimm.reward.asr_eval import measure_openasr_en_wer
from scripts.align_long_audio_transcript import canonicalize
from scripts.build_openlong_high_error_segments import DATASETS, EXPECTED, sha256


UNRELIABLE_QWEN_FLAGS = {
    "collapsed_qwen_timestamps", "nonmonotonic_across_windows", "window_edge",
    "timestamp_outside_window", "single_segment_retry", "empty_seed", "unalignable_seed",
}


def match_reference_words(reference_words, target_words, target_owners):
    """Return target segment owners for exact lexical matches in a global edit alignment."""
    ref = [canonicalize(word) for word in reference_words]
    target = [canonicalize(word) for word in target_words]
    owners = [None] * len(ref)
    for op in Levenshtein.opcodes(ref, target):
        if op.tag == "equal":
            for source, dest in zip(range(op.src_start, op.src_end), range(op.dest_start, op.dest_end), strict=True):
                owners[source] = target_owners[dest]
    return owners


def monotone_owners(proposals, weights):
    """Weighted isotonic projection, rounded to monotonic segment indexes."""
    blocks = []
    for index, (value, weight) in enumerate(zip(proposals, weights, strict=True)):
        blocks.append([index, index + 1, value * weight, weight])
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    result = [0] * len(proposals)
    for start, end, value_sum, weight_sum in blocks:
        value = int(math.floor(value_sum / weight_sum + 0.5))
        result[start:end] = [value] * (end - start)
    return result


def reconcile_parent(rows, original_reference):
    """Return conservative segment candidates and a parent-level QA summary."""
    rows = sorted(rows, key=lambda row: row["seg_index"])
    if not rows or [r["seg_index"] for r in rows] != list(range(len(rows))) or any(r["n_segments"] != len(rows) for r in rows):
        raise ValueError("Incomplete parent segment group")
    ref_words = original_reference.split()
    seed_words = [word for row in rows for word in row["prior_inferred_text"].split()]
    if seed_words != ref_words:
        raise ValueError("Prior partition does not reconstruct the original test reference")
    seed_owners = [i for i, row in enumerate(rows) for _ in row["prior_inferred_text"].split()]
    hyp_words = [word for row in rows for word in row["baseline_hypothesis"].split()]
    hyp_owners = [i for i, row in enumerate(rows) for _ in row["baseline_hypothesis"].split()]
    qwen_words = [word for row in rows for word in row["text"].split()]
    qwen_owners = [i for i, row in enumerate(rows) for _ in row["text"].split()]
    baseline_match = match_reference_words(ref_words, hyp_words, hyp_owners)
    qwen_match = match_reference_words(ref_words, qwen_words, qwen_owners)

    qwen_outliers = set()
    for index, row in enumerate(rows):
        q = len(row["text"].split())
        seed = len(row["prior_inferred_text"].split())
        hyp = len(row["baseline_hypothesis"].split())
        duration = max(0.0, row["seg_end"] - row["seg_start"])
        if q > max(seed + 30, 2 * seed + 10, 2 * hyp + 30, 4 * duration + 20):
            qwen_outliers.add(index)

    proposals, weights, evidence = [], [], []
    for index, (word, seed, baseline, qwen) in enumerate(zip(ref_words, seed_owners, baseline_match, qwen_match, strict=True)):
        marker = any(ch in word for ch in "<>[]") or "inaudible" in word.lower() or "unintelligible" in word.lower()
        baseline_near = baseline is not None and abs(baseline - seed) <= 1
        qwen_near = qwen is not None and abs(qwen - seed) <= 1
        qwen_reliable = (qwen_near and qwen not in qwen_outliers and
                         not (set(rows[qwen].get("alignment_flags", [])) & UNRELIABLE_QWEN_FLAGS) and not marker)
        owner = baseline if baseline_near else seed
        basis = "baseline_exact" if baseline_near else "seed_fallback"
        weight = 8 if baseline_near else 1
        if qwen_reliable and baseline_near and qwen == baseline:
            basis = "baseline_qwen_consensus"
            weight = 12
        elif qwen_reliable and not baseline_near:
            owner, basis, weight = qwen, "qwen_near_seed", 2
        proposals.append(owner)
        weights.append(weight)
        evidence.append({"basis": basis, "marker": marker,
                         "qwen_rejected": qwen is not None and not qwen_reliable,
                         "qwen_zero_duration_support": basis == "qwen_near_seed" and
                                                       "zero_duration" in rows[qwen].get("alignment_flags", []),
                         "baseline_far": baseline is not None and not baseline_near})
    owners = monotone_owners(proposals, weights)
    by_segment = [[] for _ in rows]
    evidence_by_segment = [Counter() for _ in rows]
    for i, (word, owner, proposal, item) in enumerate(zip(ref_words, owners, proposals, evidence, strict=True)):
        if not 0 <= owner < len(rows):
            raise ValueError("Invalid projected segment owner")
        by_segment[owner].append((i, word))
        stats = evidence_by_segment[owner]
        stats[item["basis"]] += 1
        stats["reference_markers"] += int(item["marker"])
        stats["qwen_rejected_words"] += int(item["qwen_rejected"])
        stats["qwen_zero_duration_words"] += int(item["qwen_zero_duration_support"])
        stats["baseline_far_words"] += int(item["baseline_far"])
        stats["monotonic_adjusted_words"] += int(owner != proposal)

    output = []
    for index, row in enumerate(rows):
        words = by_segment[index]
        stats = evidence_by_segment[index]
        flags = []
        if index in qwen_outliers:
            flags.append("qwen_text_outlier_rejected")
        if stats["reference_markers"]:
            flags.append("reference_marker")
        if stats["monotonic_adjusted_words"]:
            flags.append("monotonic_adjustment")
        if stats["baseline_far_words"]:
            flags.append("baseline_anchor_far_from_seed")
        if stats["qwen_rejected_words"]:
            flags.append("qwen_evidence_rejected")
        if stats["qwen_zero_duration_words"]:
            flags.append("qwen_zero_duration_support")
        if stats["seed_fallback"] > max(15, len(words) * 0.4):
            flags.append("many_unanchored_reference_words")
        if len(words) > max(160, 6 * (row["seg_end"] - row["seg_start"])):
            flags.append("unusually_long_reference")
        if not words:
            flags.append("empty_candidate")
        record = dict(row)
        record["previous_qwen_text"] = row["text"]
        record["qwen_alignment_flags"] = row.get("alignment_flags", [])
        record["qwen_alignment_status"] = row.get("alignment_status")
        record["text"] = " ".join(word for _, word in words)
        record["reference_word_start"] = words[0][0] if words else None
        record["reference_word_end"] = words[-1][0] + 1 if words else None
        record["alignment_diagnostics"] = {key: stats[key] for key in (
            "baseline_exact", "baseline_qwen_consensus", "qwen_near_seed", "seed_fallback",
            "reference_markers", "qwen_rejected_words", "qwen_zero_duration_words", "baseline_far_words", "monotonic_adjusted_words")}
        metric = measure_openasr_en_wer(record["baseline_hypothesis"], record["text"])
        record.update({"baseline_n_err": metric["n_err"], "baseline_n_ref": metric["n_ref"],
                       "baseline_wer": metric["wer"], "baseline_n_ins": metric["n_ins"],
                       "baseline_n_del": metric["n_del"], "baseline_n_sub": metric["n_sub"],
                       "alignment_method": "test_reference_baseline_qwen_consensus_v1",
                       "alignment_status": "needs_review" if flags else "hybrid_candidate",
                       "alignment_flags": flags})
        output.append(record)
    if [word for row in output for word in row["text"].split()] != ref_words:
        raise ValueError("Output does not preserve the original reference in order")
    return output, {"segments": len(output), "reference_words": len(ref_words),
                    "qwen_outlier_segments": len(qwen_outliers),
                    "needs_review": sum(row["alignment_status"] == "needs_review" for row in output),
                    "changed_from_qwen": sum(row["text"] != row["previous_qwen_text"] for row in output),
                    "flags": dict(Counter(flag for row in output for flag in row["alignment_flags"]))}


def load_original_references(source, dataset):
    references, count = {}, 0
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["data_source"] != dataset:
                raise ValueError(f"Unexpected source: {dataset}")
            parent = row["parent_audio_path"]
            if parent in references and references[parent] != row["gts"]:
                raise ValueError(f"Inconsistent original test reference: {parent}")
            references[parent] = row["gts"]
            count += 1
    if (count, len(references)) != EXPECTED[dataset]:
        raise ValueError(f"Unexpected original coverage: {dataset}")
    return references


def build_dataset(source, original_source, destination, dataset, expected_qwen_sha, expected_original_sha):
    if sha256(source) != expected_qwen_sha or sha256(original_source) != expected_original_sha:
        raise ValueError(f"Source checksum mismatch: {dataset}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    groups = defaultdict(list)
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                groups[row["parent_audio_path"]].append(row)
    refs = load_original_references(original_source, dataset)
    if (sum(map(len, groups.values())), len(groups)) != EXPECTED[dataset] or set(groups) != set(refs):
        raise ValueError(f"Input coverage mismatch: {dataset}")
    counts = Counter()
    flags = Counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for parent in sorted(groups):
            output, qa = reconcile_parent(groups[parent], refs[parent])
            for row in output:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            for field in ("segments", "reference_words", "qwen_outlier_segments", "needs_review", "changed_from_qwen"):
                counts[field] += qa[field]
            flags.update(qa["flags"])
    return {**counts, "parents": len(groups), "flags": dict(flags), "source_sha256": expected_qwen_sha,
            "original_test_sha256": expected_original_sha, "output_sha256": sha256(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-dir", type=Path, required=True)
    parser.add_argument("--qwen-summary", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--original-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    qwen_summary = json.loads(args.qwen_summary.read_text(encoding="utf-8"))
    original_summary = json.loads(args.original_summary.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"_provenance": {
        "original_test": "Original OpenLong evaluation gts, verified by checksum and parent coverage.",
        "baseline": "2609r2b step400 clean_output per segment; exact lexical matches anchor test words.",
        "qwen": "Prior Qwen timed candidates used only near the seed and outside flagged or pathological windows.",
        "reference_policy": "Original test tokens preserved exactly once and in original order; ambiguous words remain flagged.",
        "overlap_warning": "Each reference token is assigned once; overlapping audio may repeat speech across segment boundaries.",
        "quality_warning": "Automatic consensus is not human verification and cannot guarantee correct alignment.",
    }}
    for dataset in DATASETS:
        print(f"Reconciling {dataset}...", flush=True)
        summary[dataset] = build_dataset(args.qwen_dir / f"{dataset}.jsonl", args.original_dir / f"{dataset}.jsonl",
                                         args.output_dir / f"{dataset}.jsonl", dataset,
                                         qwen_summary[dataset]["output_sha256"], original_summary[dataset]["input_sha256"])
        print(dataset, json.dumps(summary[dataset]), flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

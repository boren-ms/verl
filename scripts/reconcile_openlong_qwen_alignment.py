#!/usr/bin/env python3
"""Convert Qwen window timestamps into reviewable OpenLong segment candidates.

Each source word is processed once via a core window. Words whose midpoint lies
in overlapping segment intervals are assigned to both clips. Untimed or
out-of-interval words retain their provisional seed owner and are flagged.
Candidates are machine-generated and require validation before use as labels.
"""

import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

from recipe.phimm.reward.asr_eval import measure_openasr_en_wer
from scripts.build_openlong_high_error_segments import DATASETS, EXPECTED, sha256


def _is_cjk(character):
    code = ord(character)
    return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or
            0x20000 <= code <= 0x2A6DF or 0x2A700 <= code <= 0x2B73F or
            0x2B740 <= code <= 0x2B81F or 0x2B820 <= code <= 0x2CEAF or
            0xF900 <= code <= 0xFAFF)


def qwen_english_tokens(word):
    """Match qwen-asr 0.0.6 tokenize_space_lang for one whitespace word."""
    kept = [ch for ch in word if ch == "'" or unicodedata.category(ch)[0] in ("L", "N")]
    tokens, buffer = [], []
    for ch in kept:
        if _is_cjk(ch):
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
            tokens.append(ch)
        else:
            buffer.append(ch)
    if buffer:
        tokens.append("".join(buffer))
    return tokens


def reconcile_parent(rows, results):
    """Return segment candidates and QA for one complete parent group."""
    rows = sorted(rows, key=lambda row: row["seg_index"])
    if [r["seg_index"] for r in rows] != list(range(len(rows))) or any(r["n_segments"] != len(rows) for r in rows):
        raise ValueError("Incomplete parent segment group")
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate segment id")
    assignments = defaultdict(list)
    flags = defaultdict(set)
    covered = set()
    unassigned_words = overlap_words = 0
    previous_midpoint = None
    for result in sorted(results, key=lambda r: r["window"]["core_segment_indexes"][0]):
        window = result["window"]
        core_ids = window["core_segment_ids"]
        if any(sid not in by_id or sid in covered for sid in core_ids):
            raise ValueError("Unknown or duplicate core segment")
        covered.update(core_ids)
        core_rows = [by_id[sid] for sid in core_ids]
        if [r["seg_index"] for r in core_rows] != window["core_segment_indexes"]:
            raise ValueError("Core id/index mismatch")
        seed_words = window["seed_text"].split()
        start, end = window["core_word_start"], window["core_word_end"]
        core_words = [(r, pos, word) for r in core_rows for pos, word in enumerate(r["text"].split())]
        if seed_words[start:end] != [item[2] for item in core_words]:
            raise ValueError("Core transcript differs from source seed")
        if result["status"] not in ("ok", "empty_seed", "unalignable_seed"):
            raise ValueError(f"Window inference failed: {result['status']}")
        token_groups = [qwen_english_tokens(word) for word in seed_words]
        expected = [token for tokens in token_groups for token in tokens]
        words = result["words"]
        if result["status"] == "ok" and [w["text"] for w in words] != expected:
            raise ValueError("Qwen tokenization differs from the original seed")
        if result["status"] != "ok" and (words or expected):
            raise ValueError("Window has missing Qwen timestamps")
        if window.get("retry_reason"):
            for sid in core_ids:
                flags[sid].add("single_segment_retry")
        if result.get("quality", {}).get("collapsed"):
            for seed_row, word_pos, word in core_words:
                assignments[seed_row["id"]].append((seed_row["seg_index"], word_pos, word))
            for sid in core_ids:
                flags[sid].add("collapsed_qwen_timestamps")
            unassigned_words += len(core_words)
            continue
        offsets = [0]
        for tokens in token_groups:
            offsets.append(offsets[-1] + len(tokens))
        for offset, (seed_row, word_pos, word) in enumerate(core_words):
            index = start + offset
            timings = words[offsets[index]:offsets[index + 1]]
            word_flags = set()
            if any(ch in word for ch in "<>[]") or "inaudible" in word.lower():
                word_flags.add("transcript_marker")
            if not timings:
                targets = [seed_row]
                word_flags.add("untimed_token")
                unassigned_words += 1
            else:
                begin = min(t["parent_start"] for t in timings)
                finish = max(t["parent_end"] for t in timings)
                midpoint = (begin + finish) / 2
                if finish <= begin:
                    word_flags.add("zero_duration")
                if any(t.get("clipped_to_window") for t in timings):
                    word_flags.add("timestamp_outside_window")
                if midpoint <= window["audio_start"] + 0.15 or midpoint >= window["audio_end"] - 0.15:
                    word_flags.add("window_edge")
                if previous_midpoint is not None and midpoint + 0.05 < previous_midpoint:
                    word_flags.add("nonmonotonic_across_windows")
                previous_midpoint = midpoint
                targets = [r for r in rows if r["seg_start"] <= midpoint < r["seg_end"]]
                if not targets:
                    targets = [seed_row]
                    word_flags.add("timestamp_outside_segments")
                    unassigned_words += 1
                elif len(targets) > 1:
                    overlap_words += 1
            for target in targets:
                assignments[target["id"]].append((seed_row["seg_index"], word_pos, word))
                flags[target["id"]].update(word_flags)
        if result["status"] != "ok":
            for sid in core_ids:
                flags[sid].add(result["status"])
    if covered != set(by_id):
        raise ValueError("Missing core segment windows")
    output = []
    for row in rows:
        record = dict(row)
        record["prior_inferred_text"] = row["text"]
        record["text"] = " ".join(w for _, _, w in sorted(assignments[row["id"]]))
        if not record["text"].strip():
            flags[row["id"]].add("empty_candidate")
        for name in ("baseline_n_err", "baseline_n_ref", "baseline_wer"):
            record[f"previous_{name}"] = record[name]
        metric = measure_openasr_en_wer(record["baseline_hypothesis"], record["text"])
        record.update({"baseline_n_err": metric["n_err"], "baseline_n_ref": metric["n_ref"],
                       "baseline_wer": metric["wer"], "alignment_method": "qwen3_forced_aligner_0.6b",
                       "alignment_status": "needs_review" if flags[row["id"]] else "qwen_candidate",
                       "alignment_flags": sorted(flags[row["id"]])})
        output.append(record)
    return output, {"segments": len(output), "needs_review": sum(r["alignment_status"] == "needs_review" for r in output),
                    "untimed_or_outside_words": unassigned_words, "overlap_words": overlap_words}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    results = defaultdict(list)
    for path in args.results:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    result = json.loads(line)
                    w = result["window"]
                    results[(w["dataset"], w["parent_audio_path"])].append(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"_warning": "Qwen-aligned candidates are not human-verified; flagged rows require review."}
    for dataset in DATASETS:
        source = args.input_dir / f"{dataset}.jsonl"
        groups = defaultdict(list)
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    groups[row["parent_audio_path"]].append(row)
        if (sum(map(len, groups.values())), len(groups)) != EXPECTED[dataset]:
            raise ValueError(f"Incomplete input: {dataset}")
        destination = args.output_dir / f"{dataset}.jsonl"
        stats = {"segments": 0, "parents": len(groups), "needs_review": 0,
                 "untimed_or_outside_words": 0, "overlap_words": 0}
        with destination.open("x", encoding="utf-8") as handle:
            for parent in sorted(groups):
                key = (dataset, parent)
                if key not in results:
                    raise ValueError(f"Missing Qwen results: {key}")
                rows, qa = reconcile_parent(groups[parent], results.pop(key))
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                for name in ("segments", "needs_review", "untimed_or_outside_words", "overlap_words"):
                    stats[name] += qa[name]
        stats.update({"source_sha256": sha256(source), "output_sha256": sha256(destination)})
        summary[dataset] = stats
        print(dataset, json.dumps(stats), flush=True)
    if results:
        raise ValueError("Unexpected Qwen result parents remain")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

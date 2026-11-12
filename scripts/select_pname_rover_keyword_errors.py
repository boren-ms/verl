"""Verify complete pname generation and retain source rows with keyword errors."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re

import blobfile as bf
import yaml

from recipe.phimm.data.dataset import create_datasets
from recipe.phimm.reward.asr_eval import openasr_en_eval


def select(source, validation_yaml, generation_dir, output_dir, expected_count=8930):
    source = Path(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in source.read_text().splitlines()]
    originals = {row["id"]: row for row in records}
    if len(originals) != len(records) or len(records) != expected_count:
        raise ValueError("Source count or ID uniqueness does not match the declared universe")
    config = copy.deepcopy(yaml.safe_load(Path(validation_yaml).read_text()))
    config[0]["jsonl_paths"] = str(source)
    dataset = create_datasets(config)["dataset_0"]
    expected = {row["extra_info"]["id"]: row for row in dataset}
    if set(expected) != set(originals):
        raise ValueError("Actual validation loader changed source membership")

    parts = []
    for path in bf.glob(generation_dir.rstrip("/") + "/part-*.jsonl"):
        match = re.fullmatch(r"part-(\d+)\.jsonl", bf.basename(path))
        if match:
            parts.append((int(match[1]), path))
    parts.sort()
    if not parts or [index for index, _ in parts] != list(range(len(parts))):
        raise ValueError("Generation requires a complete contiguous output-part sequence")

    scores = {}
    part_audit = []
    for _, path in parts:
        digest = hashlib.sha256()
        count = 0
        with bf.BlobFile(path, "rb") as stream:
            for line in stream:
                digest.update(line)
                row = json.loads(line)
                record_id = row["id"]
                if record_id in scores or record_id not in expected:
                    raise ValueError(f"Duplicate or unexpected generated ID: {record_id}")
                example = expected[record_id]
                if row["text"] != example["reward_model"]["ground_truth"]:
                    raise ValueError(f"Generation reference differs from updated_text: {record_id}")
                if row["audio_path"] != example["audio_path"]:
                    raise ValueError(f"Generation audio selector differs: {record_id}")
                if row["keywords"] != example["extra_info"]["keywords"]:
                    raise ValueError(f"Generation keyword list differs: {record_id}")
                canonical = openasr_en_eval(
                    row["raw_response"], row["text"], version=2609, extra_info=example["extra_info"]
                )
                for key in ("nb_err", "nb_ref", "n_err", "n_ref", "wer"):
                    if not math.isclose(row[key], canonical[key], abs_tol=1e-12, rel_tol=1e-9):
                        raise ValueError(f"Canonical score mismatch for {record_id}: {key}")
                scores[record_id] = row
                count += 1
        if not count:
            raise ValueError(f"Empty generation part: {path}")
        part_audit.append({"path": path, "rows": count, "sha256": digest.hexdigest()})
    if set(scores) != set(originals):
        raise ValueError(f"Incomplete coverage: {len(scores)}/{len(originals)} source IDs")

    selected = [row for row in records if scores[row["id"]]["nb_err"] > 0]
    selected_path = output_dir / "keyword_errors.jsonl"
    selected_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    audit_path = output_dir / "keyword_errors.audit.jsonl"
    audit_path.write_text("".join(json.dumps(scores[row["id"]], ensure_ascii=False) + "\n" for row in selected))
    counts = {key: sum(row[key] for row in scores.values()) for key in ("nb_err", "nb_ref", "n_err", "n_ref")}
    summary = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "validation_yaml": str(Path(validation_yaml).resolve()),
        "generation_dir": generation_dir,
        "expected_records": expected_count,
        "generated_unique_ids": len(scores),
        "selected_records": len(selected),
        "selection": "canonical openasr_en_eval(version=2609), nb_err > 0",
        "source_split": "validation",
        "role": "biased baseline-error diagnostic; not training data",
        "references": "Source rows unchanged; updated_text remains the reference",
        "response_at_limit": sum(bool(row.get("response_at_limit")) for row in scores.values()),
        "selected_response_at_limit": sum(bool(scores[row["id"]].get("response_at_limit")) for row in selected),
        "selected_audio_hours": sum(scores[row["id"]].get("audio_duration_seconds", 0) for row in selected) / 3600,
        "all_audio_hours": sum(row.get("audio_duration_seconds", 0) for row in scores.values()) / 3600,
        "counts": counts,
        "wer": counts["n_err"] / max(counts["n_ref"], 1),
        "keyword_error_rate": counts["nb_err"] / max(counts["nb_ref"], 1),
        "selected_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        "membership_sha256": hashlib.sha256("\n".join(row["id"] for row in selected).encode()).hexdigest(),
        "parts": part_audit,
    }
    (output_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    review = [{"source": row, "generation": scores[row["id"]]} for row in selected[:2]]
    (output_dir / "keyword_errors.review_2.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--validation-yaml", required=True)
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-count", type=int, default=8930)
    args = parser.parse_args()
    select(args.source, args.validation_yaml, args.generation_dir, args.output_dir, args.expected_count)

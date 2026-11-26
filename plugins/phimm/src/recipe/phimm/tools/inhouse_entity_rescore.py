"""Recompute EER from audited saved segment outputs, without loading a model or audio."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path

import blobfile as bf
from recipe.phimm.reward import asr_inhouse_measure as measure
from recipe.phimm.reward.asr_response import get_hyp_text


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def rescore_row(row, expected, pack_dir):
    detail = json.loads(row["asr_detail"])
    info = detail.get("extra_info", detail.get("parent_info", {}))
    if info["id"] != expected["id"] or row["asr_data_source"] != expected["data_source"]:
        raise ValueError("Changed recording ID or datasource")
    if row["gts"] != expected["text"] or info["entity_transcription"] != expected["entity_transcription"]:
        raise ValueError("Changed original full-parent reference")
    paths = [segment["audio_path"] for segment in detail.get("asr_segments", detail.get("segment_rows", []))]
    if paths != expected["audio_path"] or len(detail["segment_outputs"]) != len(paths):
        raise ValueError("Missing or reordered segment outputs")
    # The original evaluator cleans each generated clip and joins in recording order.
    hypothesis = get_hyp_text(detail["segment_outputs"])
    n_err, n_ref, eer = measure._compute_eer(
        row["gts"], hypothesis, pack_dir, strict=True, mode="nondisfluency", locale="en-us"
    )
    return {
        "id": info["id"],
        "dataset": row["asr_data_source"],
        "segments": len(paths),
        "reference_sha256": _digest(row["gts"]),
        "hypothesis_sha256": _digest(hypothesis),
        "segment_outputs_sha256": _digest(detail["segment_outputs"]),
        "previous_contexts_sha256": _digest(detail["previous_contexts"]),
        "old_eer_n_err": row["eer_n_err"],
        "old_eer_n_ref": row["eer_n_ref"],
        "eer_n_err": n_err,
        "eer_n_ref": n_ref,
        "eer": eer,
    }


def _read_json(uri):
    with bf.BlobFile(str(uri), "r") as stream:
        return json.load(stream)


def _read_rows(uri, cache_dir=None):
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / (hashlib.sha256(uri.encode()).hexdigest() + ".jsonl")
        if not cached.exists():
            bf.copy(uri, str(cached))
        uri = str(cached)
    with bf.BlobFile(uri, "r") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Original audited EER summary.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pack-dir", default=measure.DEFAULT_PACK_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, help="Cache original manifests and saved outputs locally")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    original = _read_json(args.summary)
    if original["suite"] != "eer":
        raise ValueError("Rescoring requires an audited EER suite")
    pack = measure.ensure_pack_dir(args.pack_dir)
    datasets, details = [], []
    for dataset in original["datasets"]:
        expected = _read_rows(dataset["manifest"], args.cache_dir)
        references = {row["id"]: row for row in expected}
        rows = _read_rows(dataset["detail_path"], args.cache_dir)
        parsed_details = (json.loads(row["asr_detail"]) for row in rows)
        ids = [detail.get("extra_info", detail.get("parent_info", {}))["id"] for detail in parsed_details]
        if len(references) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(references):
            raise ValueError("Duplicate, missing or unexpected saved recording")
        with ThreadPoolExecutor(args.workers) as executor:
            rescored = list(executor.map(rescore_row, rows, [references[uid] for uid in ids], repeat(pack)))
        counts = {
            key: sum(row[key] for row in rescored)
            for key in ("segments", "old_eer_n_err", "old_eer_n_ref", "eer_n_err", "eer_n_ref")
        }
        if len(rescored) != dataset["parents"] or counts["segments"] != dataset["segments"]:
            raise ValueError("Saved recording/segment coverage differs from original audit")
        if (counts["old_eer_n_err"], counts["old_eer_n_ref"]) != (dataset["eer_n_err"], dataset["eer_n_ref"]):
            raise ValueError("Original audited EER differs from saved detail counts")
        values = {
            "dataset": dataset["dataset"],
            "parents": len(rescored),
            **counts,
            "eer": counts["eer_n_err"] / counts["eer_n_ref"],
            "old_eer": dataset["eer"],
            "detail_path": dataset["detail_path"],
            "manifest": dataset["manifest"],
        }
        datasets.append(values)
        details.extend(rescored)
        print(f"Rescored {values['dataset']}: {values['parents']} recordings, EER {values['eer']:.4%}", flush=True)
    totals = {
        key: sum(row[key] for row in datasets)
        for key in ("parents", "segments", "old_eer_n_err", "old_eer_n_ref", "eer_n_err", "eer_n_ref")
    }
    if (totals["parents"], totals["segments"]) != (original["total"]["parents"], original["total"]["segments"]):
        raise ValueError("Incomplete suite")
    summary = {
        "source_summary": args.summary,
        "source_result_root": original["result_root"],
        "model_label": original["model_label"],
        "prompt_style": original["prompt_style"],
        "context_mode": original["context_mode"],
        "language_hint": original["language_hint"],
        "tools": {"bundle": measure.TOOLS_MANIFEST, "speechinsight_version": measure.SPEECHINSIGHT_VERSION},
        "metric": "EER only; nondisfluency, full parent DisplayTranscription",
        "decoded_again": False,
        "datasets": datasets,
        "total": {**totals, "eer": totals["eer_n_err"] / totals["eer_n_ref"]},
        "primary_eer": sum(row["eer"] for row in datasets) / len(datasets),
        "old_primary_eer": original["primary_eer"],
        "changed_recordings": sum(
            (row["eer_n_err"], row["eer_n_ref"]) != (row["old_eer_n_err"], row["old_eer_n_ref"]) for row in details
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "details.jsonl").write_text("".join(json.dumps(row) + "\n" for row in details))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare AISHELL human labels and deterministic absolute-error subsets.

No generated hypothesis is used as a training target. Blob publication is a
separate azcopy step. See docs/aishell_refine.md for commands and provenance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

SOURCE = "https://highperf01eus2.blob.core.windows.net/data"
PREFIX = "am_data/publicdataset_process/AIShell-1/train_data/feature_extraction/"
AUDIO_ROOT = "az://orngwus2cresco/data/speech/" + PREFIX
MODEL = "az://orngwus2cresco/data/speech/projects/phi-fastllm-2611/amlt-results/fast-llm-2611-qwen9b-flashenc-mtp-s2-data-v3-8-2-1-shadow/38000/qwen_hf/"
TEST_SHA = "81865255a717c8e759a9b96ebe93cb0a0ac273c2df2e29dc458a0609025c677e"
MANIFEST_SHA = "3afd464a34ec12be557a617cff7e6976b3562cd97851e759ce94de654e22c7c0"
FILENAMES_SHA = "943dd0851960c1f242ea61f0b90208af526f0bfbe803b12a36e1d64c2bf0813c"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace atomically so an interrupted selection cannot publish a prefix.
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def write_meta(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def utterance_id(value):
    match = re.search(r"BAC009S\d+W\d+", value)
    if not match:
        raise ValueError(f"Not an AISHELL utterance ID: {value!r}")
    return match.group()


def parse_chunk(data, kind, count):
    if data[:len(kind)].lower() != kind.encode():
        raise ValueError(f"Wrong chunk header: {kind}")
    offset = len(kind) + 4  # header followed by uint32 version
    result = []
    for index in range(count):
        actual, size = struct.unpack_from("<II", data, offset)
        offset += 8
        if actual != index or size <= 0 or offset + size > len(data):
            raise ValueError(f"Invalid {kind} record {index}")
        value = data[offset:offset + size].decode("utf-8-sig")
        result.append(json.loads(value) if kind == "info" else value)
        offset += size
    if offset != len(data):
        raise ValueError(f"Trailing bytes in {kind} chunk")
    return result


def prepare(args):
    import requests

    if sha(args.file_set) != MANIFEST_SHA or sha(args.filenames) != FILENAMES_SHA:
        raise ValueError("Source metadata hash differs from verified AISHELL source")
    chunks = json.loads(Path(args.file_set).read_text())["fileInfo"]
    expected = {}
    for line in Path(args.filenames).read_text(encoding="utf-8-sig").splitlines():
        filename, text = line.split("\t", 1)
        ident = utterance_id(filename)
        if ident in expected:
            raise ValueError(f"Duplicate filename {ident}")
        expected[ident] = text
    if len(chunks) != 301 or len(expected) != 120098:
        raise ValueError("Wrong full training source size")
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    sas = json.loads(Path(args.sas_cache).expanduser().read_text())[SOURCE]["sas"].lstrip("?")

    def fetch(chunk):
        name, count = chunk["name"], chunk["count"]
        parsed = {}
        for kind in ("info", "transcription"):
            dest = cache / f"{name}.{kind}"
            if not dest.exists():
                # Do not expose a credential-bearing URL in request exceptions.
                try:
                    response = requests.get(f"{SOURCE}/{PREFIX}{name}.{kind}?{sas}", timeout=90)
                except requests.RequestException:
                    raise RuntimeError(f"Source read failed: {name}.{kind}") from None
                if response.status_code != 200:
                    raise RuntimeError(f"Source read {name}.{kind}: HTTP {response.status_code}")
                dest.write_bytes(response.content)
            parsed[kind] = parse_chunk(dest.read_bytes(), kind, count)
        rows = []
        for index, (info, text) in enumerate(zip(parsed["info"], parsed["transcription"])):
            ident = utterance_id(info["filename"])
            if "".join(text.split()) != "".join(expected[ident].split()):
                raise ValueError(f"Human reference mismatch: {ident}")
            rows.append({"id": ident, "text": text, "language": "Chinese",
                         "audio_chunk": f"{AUDIO_ROOT}{name}.audio:{count}:{index}",
                         "source_filename": info["filename"], "source_chunk": name,
                         "source_chunk_index": index, "source_split": "train",
                         "source_manifest_sha256": MANIFEST_SHA})
        return rows

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        groups = list(pool.map(fetch, chunks))  # input order is deterministic
    rows = [row for group in groups for row in group]
    if len(rows) != len(expected) or {r["id"] for r in rows} != set(expected):
        raise ValueError("Full source identity/count mismatch")
    for index, row in enumerate(rows):
        row["source_row"] = index
    tests = read_jsonl(args.test_details)
    overlap = set(expected) & {utterance_id(r["id"]) for r in tests}
    if overlap:
        raise ValueError(f"Training/test leakage: {len(overlap)} IDs")
    output = Path(args.output)
    write_jsonl(output, rows)
    write_meta(str(output) + ".meta.json", {
        "rows": len(rows), "sha256": sha(output), "source": SOURCE + "/" + PREFIX,
        "source_manifest_sha256": MANIFEST_SHA, "source_filenames_sha256": FILENAMES_SHA,
        "chunks": len(chunks), "speakers": len({r["id"].split("W")[0] for r in rows}),
        "source_hours": sum(c["duration"] for c in chunks) / 3600,
        "human_references_verified": True, "train_test_id_overlap": len(overlap)})
    print(f"Prepared {len(rows)} full-training rows: {sha(output)}")


def validate_counts(row):
    for key in ("n_err", "n_ref", "n_ins", "n_del", "n_sub"):
        if type(row.get(key)) is not int or row[key] < 0:
            raise ValueError(f"Invalid {key} for {row.get('id')}")
    if row["n_ref"] <= 0 or row["n_err"] != sum(row[k] for k in ("n_ins", "n_del", "n_sub")):
        raise ValueError(f"Inconsistent edit counts for {row.get('id')}")


def select(args):
    inputs = [Path(p) for p in args.inputs]
    is_test = args.split == "validation"
    if is_test and (len(inputs) != 1 or sha(inputs[0]) != TEST_SHA):
        raise ValueError("Validation must use the pinned previous 2611r2 test details")
    if not is_test:
        indices = [int(re.fullmatch(r"part-(\d+)\.jsonl", p.name).group(1)) for p in inputs]
        if sorted(indices) != list(range(len(indices))):
            raise ValueError("Generation parts must be unique and contiguous from zero")
    rows = [row for path in inputs for row in read_jsonl(path)]
    expected_count, topk = (7176, 200) if is_test else (120098, 20000)
    if len(rows) != expected_count or len({r["id"] for r in rows}) != expected_count:
        raise ValueError(f"Incomplete/duplicate {args.split} results: {len(rows)}")
    # Keep the scored hypothesis in the audit, never in the training target.
    full = {} if is_test else {r["id"]: r for r in read_jsonl(args.full_manifest)}
    if not is_test and set(full) != {r["id"] for r in rows}:
        raise ValueError("Generation ID set does not equal full training manifest")
    for row in rows:
        validate_counts(row)
        if not is_test:
            original = full[row["id"]]
            duration = row.get("audio_duration_seconds")
            if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"Invalid measured audio duration: {row['id']}")
            if not isinstance(row.get("raw_response"), str):
                raise ValueError(f"Missing raw generation response: {row['id']}")
            for key in ("text", "audio_chunk", "source_row", "language"):
                if row.get(key) != original[key]:
                    raise ValueError(f"Generation metadata mismatch: {row['id']} {key}")
    ranked = sorted(rows, key=lambda r: (-r["n_err"], r["source_row"], r["id"]))
    selected, audit = [], []
    for rank, row in enumerate(ranked[:topk], 1):
        if is_test:
            record = {"id": row["id"], "text": row["gts"], "audio_path": row["audio_path"],
                      "language": "Chinese", "source_row": row["source_row"],
                      "source_split": "test", "diagnostic_subset": True,
                      "source_revision": row["source_revision"]}
        else:
            record = dict(full[row["id"]])
            if row.get("audio_duration_seconds", 0) <= 0:
                raise ValueError("Generation must include positive measured audio duration")
            record["duration"] = row["audio_duration_seconds"]
        record.update(selection_rank=rank, selection_n_err=row["n_err"],
                      selection_n_ref=row["n_ref"], selection_model=MODEL)
        selected.append(record)
        audit.append({**record, "scored_hypothesis": row["output"] if is_test else row["raw_response"],
                      **{k: row[k] for k in ("n_err", "n_ref", "n_ins", "n_del", "n_sub")}})
    if args.verify_scorer:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from recipe.phimm.reward.openasr_expansion import openasr_expansion_eval
        for row in audit:
            counts = openasr_expansion_eval(row["scored_hypothesis"], row["text"],
                                           version=2609, extra_info={"language": "Chinese"})
            for key in ("n_err", "n_ref", "n_ins", "n_del", "n_sub"):
                if counts[key] != row[key]:
                    raise ValueError(f"Canonical scorer mismatch: {row['id']} {key}")
    output = Path(args.output)
    write_jsonl(output, selected)
    write_jsonl(str(output) + ".audit.jsonl", audit)
    write_meta(str(output) + ".meta.json", {
        "rows": len(selected), "candidate_rows": len(rows), "sha256": sha(output),
        "sort": ["n_err descending", "source_row ascending", "id ascending"],
        "model": MODEL, "metric": "Mandarin CER; absolute edit count selection",
        "minimum_selected_n_err": selected[-1]["selection_n_err"],
        "maximum_selected_n_err": selected[0]["selection_n_err"],
        "zero_error_rows_selected": sum(r["selection_n_err"] == 0 for r in selected),
        "selected_n_err": sum(r["selection_n_err"] for r in selected),
        "selected_n_ref": sum(r["selection_n_ref"] for r in selected),
        "selected_max_duration": max((r.get("duration", 0) for r in selected), default=0) if not is_test else None,
        "selected_hours": sum(r["duration"] for r in selected) / 3600 if not is_test else None,
        "full_manifest_sha256": None if is_test else sha(args.full_manifest),
        "human_targets_retained": True, "diagnostic_subset": is_test,
        "selected_rows_rescored": args.verify_scorer,
        "input_sha256": {str(p): sha(p) for p in inputs}})
    print(f"Selected {len(selected)}/{len(rows)} {args.split} rows: {sha(output)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for key in ("file-set", "filenames", "test-details", "cache", "output"):
        prep.add_argument("--" + key, required=True)
    prep.add_argument("--sas-cache", default="~/.sas/azure_blob_sas_cache.json")
    prep.add_argument("--workers", type=int, default=12)
    prep.set_defaults(func=prepare)
    pick = sub.add_parser("select")
    pick.add_argument("--split", choices=["train", "validation"], required=True)
    pick.add_argument("--inputs", nargs="+", required=True)
    pick.add_argument("--full-manifest")
    pick.add_argument("--output", required=True)
    pick.add_argument("--verify-scorer", action="store_true", help="Re-score every selected row before writing")
    pick.set_defaults(func=select)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

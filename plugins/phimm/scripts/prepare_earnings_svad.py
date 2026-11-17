"""Reproducible, resumable preparation of the five Earnings long-form batches."""

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import yaml

NAMES = ["earn_fy26_batch1", "earn_fy26_batch2", "quartr_1st23_long", "quartr_2nd48_long", "quartr_3rd38_long"]
EXPECTED = dict(zip(NAMES, [80, 104, 98, 235, 222], strict=True))
OPTIONS = dict(max_len_sec=30.0, min_seg_sec=0.1, num_threads=1, empty_policy="fixed")


def digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def work(item):
    from recipe.phimm.tools.svad_segment import segment_row

    name, index, row, row_hash = item
    start = time.monotonic()
    for attempt in range(3):
        try:
            output = segment_row(row, **OPTIONS)
            return name, index, {"input_sha256": row_hash, "elapsed_seconds": time.monotonic() - start, "row": output}
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    from datasets import Dataset
    from recipe.phimm.data.dataset import process_ds

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--blob-root", help="Publication root for emitted datasource URIs")
    args = parser.parse_args()
    root = args.root.resolve()
    configs = yaml.safe_load((root / "provenance/source_configs.yaml").read_text())
    inputs, results, jobs = {}, {}, []
    for name in NAMES:
        raw_path = root / "provenance" / f"{name}.raw.jsonl"
        raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
        assert len(raw) == EXPECTED[name], (name, len(raw))
        normalized = process_ds(Dataset.from_list(raw), **configs[name].get("pre_process", {})).to_list()
        for row in normalized:
            row["data_source"] = name
        inputs[name] = normalized
        directory = root / "provenance/parents" / name
        directory.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(normalized):
            target = directory / f"{index:05d}.json"
            row_hash = digest(row)
            if target.exists():
                record = json.loads(target.read_text())
                assert record["input_sha256"] == row_hash, f"Stale checkpoint: {target}"
                results[name, index] = record
            else:
                jobs.append((name, index, row, row_hash))
    total = sum(EXPECTED.values())
    start = time.monotonic()

    def progress(status, failures=None):
        record = {
            "status": status,
            "parents_completed": len(results),
            "parents_total": total,
            "segments_completed": sum(len(v["row"]["audio_path"]) for v in results.values()),
            "elapsed_seconds_this_run": round(time.monotonic() - start, 1),
            "failures": failures or [],
        }
        atomic_json(root / "progress.json", record)
        print(json.dumps(record), flush=True)

    progress("running")
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        pending = {pool.submit(work, job): (job[0], job[1]) for job in jobs}
        for future in as_completed(pending):
            name, index = pending[future]
            try:
                _, _, record = future.result()
            except Exception as exc:
                failures.append({"source": name, "index": index, "error": str(exc)})
                progress("running", failures)
                continue
            atomic_json(root / "provenance/parents" / name / f"{index:05d}.json", record)
            results[name, index] = record
            if len(results) % 10 == 0 or len(results) == total:
                progress("running", failures)
    if failures:
        progress("incomplete", failures)
        raise RuntimeError(f"{len(failures)} parents failed; rerun this script to retry only missing parents")
    sources, combined, stats = [], [], []
    for name in NAMES:
        rows = [results[name, index]["row"] for index in range(EXPECTED[name])]
        for original, row in zip(inputs[name], rows, strict=True):
            assert row["text"] == original["text"] and row["id"] == original["id"]
            assert row["svad_source"] == original["audio_path"]
            assert len(row["svad_bounds"]) == len(row["audio_path"]) > 0
            previous = 0
            for spec, (left, right) in zip(row["audio_path"], row["svad_bounds"], strict=True):
                assert previous <= left < right <= row["svad_parent_duration"] + 1e-8
                assert right - left <= 30 + 1e-8
                assert spec == f"{row['svad_source']}#{left}:{right}"
                previous = right
        path = root / f"{name}.jsonl"
        text = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
        temp = path.with_suffix(".jsonl.tmp")
        temp.write_text(text)
        temp.replace(path)
        source = deepcopy(configs[name])
        source.pop("pre_process", None)
        source["jsonl_paths"] = f"{args.blob_root.rstrip('/')}/{path.name}" if args.blob_root else str(path)
        sources.append(source)
        combined.extend(rows)
        lengths = [end - start for row in rows for start, end in row["svad_bounds"]]
        stats.append(
            {
                "name": name,
                "parents": len(rows),
                "segments": len(lengths),
                "parent_hours": sum(row["svad_parent_duration"] for row in rows) / 3600,
                "retained_hours": sum(lengths) / 3600,
                "min_segment_seconds": min(lengths),
                "max_segment_seconds": max(lengths),
                "mean_segment_seconds": sum(lengths) / len(lengths),
                "fallback_parents": sum(row["svad_method"] == "fixed_fallback" for row in rows),
                "manifest": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    combined_path = root / "earnings_long_svad30.jsonl"
    combined_path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in combined))
    (root / "train_data.yaml").write_text(yaml.safe_dump(sources, sort_keys=False, allow_unicode=True))
    summary = {
        "options": OPTIONS,
        "parents": total,
        "segments": sum(s["segments"] for s in stats),
        "parent_hours": sum(s["parent_hours"] for s in stats),
        "retained_hours": sum(s["retained_hours"] for s in stats),
        "fallback_parents": sum(s["fallback_parents"] for s in stats),
        "batches": stats,
        "train_data_yaml": str(root / "train_data.yaml"),
        "combined_manifest": str(combined_path),
        "combined_sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
        "reference_policy": "one original full reference per parent; ordered audio_path list",
        "audio_output": "none; selectors reference original audio sources",
        "publication_root": args.blob_root,
    }
    atomic_json(root / "preparation.json", summary)
    progress("prepared")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

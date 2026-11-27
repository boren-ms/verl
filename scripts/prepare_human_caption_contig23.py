"""Materialize Human Caption contig23 from audited full-reference alignment tables.

Reuse the greedy <=30s cuts in the V3 validation SQLite tables. Check every
parent's reconstructed reference against its original reference hash, including
zero-duration words. Audio remains timed selectors of existing parent chunks.
"""

import argparse
import gzip
import hashlib
import json
import math
import sqlite3
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def jsha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def windows(size):
    """Match earnings contig23, with a single pair for two-segment parents."""
    if size < 2:
        return []
    if size == 2:
        return [(0, 2)]
    result = [(start, min(2 + start % 2, size - start)) for start in range(size - 1)]
    start = size - 3
    result.append((start, 5 - result[start][1]))
    return sorted(result)


def write(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare_shard(task):
    db_path, receipts_dir, output_dir = map(Path, task)
    name = db_path.stem
    receipt_path = receipts_dir / f"refilter_consecutive_max3_{name}_verified.json"
    publication_path = receipts_dir / f"refilter_consecutive_max3_{name}_publication_verified.json"
    receipt = json.loads(receipt_path.read_text())
    publication = json.loads(publication_path.read_text())
    assert receipt["status"] == "independent_runs_policy_all_version_hashes_coverage_pairs_exclusions_verified"
    assert publication["counts"] == receipt["counts"]
    identity = {
        "input_sqlite_sha256": sha(db_path),
        "receipt_sha256": sha(receipt_path),
        "publication_receipt_sha256": sha(publication_path),
        "script_sha256": sha(__file__),
    }
    out = output_dir / name
    complete = out / "COMPLETE.json"
    if complete.exists():
        saved = json.loads(complete.read_text())
        assert saved["identity"] == identity
        assert all(sha(out / file) == digest for file, digest in saved["output_sha256"].items())
        return saved
    out.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    assert db.execute("select count(*) from originals").fetchone()[0] == receipt["counts"]["filtered_originals"]
    assert db.execute("select count(*) from segments").fetchone()[0] == receipt["counts"]["filtered_originals"]
    query = (
        "select s.id,s.paths,s.labels,s.metadata,s.n,o.ref_sha,o.source,o.seconds,s.q_sha,o.q_sha "
        "from segments s join originals o on s.id=o.id"
    )
    counts = Counter()
    durations, sample_durations = [], []
    names = ["short_segments", "segmented_parents", "contig23", "contig23_exclusions"]
    streams = {key: gzip.open(out / f"{key}.jsonl.gz", "wt", encoding="utf-8", compresslevel=1) for key in names}
    try:
        for parent, raw_paths, raw_labels, raw_metadata, n, text_hash, source, seconds, sq, oq in db.execute(query):
            paths, labels, metadata = map(json.loads, [raw_paths, raw_labels, raw_metadata])
            assert len(paths) == len(labels) == len(metadata) == n and sq == oq
            text = " ".join(labels)
            assert jsha(text) == text_hash, (parent, "original text differs")
            lengths = []
            word_end, previous_end = 0, 0
            for i, (path, label, m) in enumerate(zip(paths, labels, metadata, strict=True)):
                start, end = m["start"], m["end"]
                assert math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= float(seconds)
                assert end - start <= 30 and start >= previous_end - 1e-6
                assert path == f"{source}#{start:.3f}:{end:.3f}"
                assert m["word_start"] == word_end and m["word_end"] - word_end == len(label.split())
                word_end, previous_end = m["word_end"], end
                length = end - start
                lengths.append(length)
                durations.append(length)
                write(
                    streams["short_segments"],
                    {
                        "id": f"{parent}:seg:{i}",
                        "parent_id": parent,
                        "parent_audio_path": source,
                        "audio_path": path,
                        "text": label,
                        "audio_length": length,
                        "language": "English",
                        "keywords": [],
                        "segment_index": i,
                        "segment_metadata": m,
                        "reference_policy": "all_human_words_including_zero_duration",
                    },
                )
            assert word_end == len(text.split())
            counts["parents"] += 1
            counts["original_words"] += word_end
            counts["short_segments"] += n
            parent_record = {
                "id": parent,
                "parent_id": parent,
                "parent_audio_path": source,
                "audio_path": paths,
                "text": text,
                "original_text": text,
                "segment_reference_texts": labels,
                "segment_metadata": metadata,
                "audio_length": sum(lengths),
                "original_audio_length": float(seconds),
                "language": "English",
                "keywords": [],
                "reference_policy": "all_human_words_including_zero_duration",
            }
            write(streams["segmented_parents"], parent_record)
            selected_windows = windows(n)
            assert len(selected_windows) == len(set(selected_windows))
            covered = set()
            for start, count in selected_windows:
                indices = list(range(start, start + count))
                covered.update(indices)
                sample_length = sum(lengths[start : start + count])
                sample_durations.append(sample_length)
                sample = {
                    "id": f"{parent}:contig23:{start}:{count}",
                    "parent_id": parent,
                    "parent_audio_path": source,
                    "audio_path": paths[start : start + count],
                    "text": " ".join(labels[start : start + count]),
                    "segment_reference_texts": labels[start : start + count],
                    "segment_metadata": metadata[start : start + count],
                    "segment_indices": indices,
                    "source_audio_lengths": lengths[start : start + count],
                    "audio_length": sample_length,
                    "language": "English",
                    "keywords": [],
                    "merge_count": count,
                    "merge_method": "contiguous23",
                    "reference_policy": "all_human_words_including_zero_duration",
                }
                assert count in (2, 3) and len(sample["audio_path"]) == count
                write(streams["contig23"], sample)
                counts[f"samples_{count}_segments"] += 1
                counts["sample_word_occurrences"] += len(sample["text"].split())
            if selected_windows:
                assert covered == set(range(n))
                counts["contig23_parents"] += 1
            else:
                counts["singleton_parents"] += 1
                write(
                    streams["contig23_exclusions"],
                    {
                        "id": parent,
                        "reason": "fewer_than_two_segments",
                        "segment_count": n,
                        "retained_in": "short_segments_and_segmented_parents",
                    },
                )
            counts["contig23_samples"] += len(selected_windows)
    finally:
        for stream in streams.values():
            stream.close()
        db.close()
    assert counts["parents"] == receipt["counts"]["filtered_originals"]
    assert counts["singleton_parents"] == receipt["counts"]["fewer_than_two_segment_parents"]
    import numpy as np

    np.save(out / "segment_durations.npy", durations)
    np.save(out / "sample_durations.npy", sample_durations)
    files = [f"{key}.jsonl.gz" for key in names] + ["segment_durations.npy", "sample_durations.npy"]
    saved = {
        "shard": name,
        "identity": identity,
        "counts": dict(counts),
        "output_sha256": {file: sha(out / file) for file in files},
    }
    complete.write_text(json.dumps(saved, indent=2) + "\n")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--receipts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    inputs = sorted(args.validation_dir.glob("shard-*.sqlite"))
    assert len(inputs) == 8
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(prepare_shard, (p, args.receipts_dir, args.output_dir)) for p in inputs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"completed": result["shard"], "counts": result["counts"]}), flush=True)
    counts = Counter()
    for result in results:
        counts.update(result["counts"])
    assert counts["parents"] == 417814 and counts["original_words"] == 398025213
    assert counts["short_segments"] == 5388363 and counts["singleton_parents"] == 14693
    import numpy as np

    stats = {}
    for name in ["segment", "sample"]:
        a = np.concatenate([np.load(args.output_dir / r["shard"] / f"{name}_durations.npy") for r in results])
        stats[name] = {
            "count": len(a),
            "mean_seconds": float(a.mean()),
            "percentiles_seconds": dict(
                zip(
                    ["P0", "P50", "P90", "P95", "P99", "P100"],
                    np.percentile(a, [0, 50, 90, 95, 99, 100]).tolist(),
                    strict=False,
                )
            ),
            "total_hours": float(a.sum() / 3600),
            "at_least_25s_percent": float((a >= 25).mean() * 100),
        }
    proof = {
        "status": "all_eight_shards_materialized_reference_hashes_and_bounds_verified",
        "counts": dict(counts),
        "duration_statistics": stats,
        "shards": sorted(results, key=lambda r: r["shard"]),
        "reference_policy": "all_human_words_including_zero_duration",
        "segmentation": "reuse_verified_greedy_maximal_aligned_word_prefix_le30s",
        "audio_storage": "existing_parent_timed_selectors_no_new_waveforms",
        "alignment_rerun": False,
        "uploaded": False,
        "training_started": False,
    }
    (args.output_dir / "COMPLETE.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({"counts": dict(counts), "duration_statistics": stats}, indent=2), flush=True)


if __name__ == "__main__":
    main()

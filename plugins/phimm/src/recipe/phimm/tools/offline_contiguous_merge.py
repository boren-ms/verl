"""Build ordered audio-list ASR data from utterances of each recording.

By default, each recording contributes as many samples as input utterances. Windows
alternate between two and three neighbors, and an additional unused window
preserves the sample count at recording boundaries. No rows are randomized,
no windows cross recordings, and audio remains an ordered list of paths.
Use --merge-mode full_recording for one sample containing all available
utterances per recording, each used exactly once. No audio files are created.
Use --merge-mode contiguous5 for every unique five-neighbor window. This mode
produces N - 4 samples per recording and requires at least five utterances.
"""

import argparse
import gzip
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import yaml

from recipe.phimm.data.dataset import jsonl_dataset, process_ds
from recipe.phimm.tools.random_merge_list import _merge_rows


def _start(row):
    values = [row.get("start_seconds"), row.get("part_start_seconds", 0)]
    if any(isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) for v in values):
        raise ValueError("Contiguous merge requires finite start_seconds and part_start_seconds")
    return sum(values)


def _duration(row):
    value = next(
        (row[k] for k in ("audio_length", "audio_duration_seconds", "duration") if row.get(k) is not None), None
    )
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise ValueError("Contiguous merge requires a positive finite duration on every utterance")
    return float(value)


def _segment_identity(row):
    match = re.fullmatch(r"(.+)_seg(\d+)(?:_part(\d+))?", str(row.get("id", "")))
    if match is None:
        raise ValueError("Segment-ID ordering requires IDs ending in _segN or _segN_partN")
    return match[1], (int(match[2]), int(match[3] or 0))


def contiguous_merge(
    rows,
    *,
    originals=None,
    group_key="original_audio_url",
    default_language=None,
    order_by="timestamp",
    merge_mode="contiguous23",
):
    """Return ordered windows or one full sample per recording.

    In full_recording mode, return one sample per recording, including singletons.
    The default contiguous23 mode returns N distinct windows from N input rows.
    In contiguous5 mode, return all N - 4 unique five-utterance windows.
    Default groups must contain at least three utterances: two utterances cannot form
    two distinct windows of length two or three. Recordings retain first-seen
    order; utterances are sorted by their absolute start including part offset,
    or by numeric segment/part IDs for sources without timestamp metadata.
    """
    if originals is None:
        originals = rows
    if len(originals) != len(rows):
        raise ValueError("Original and normalized rows must align")
    if not rows:
        raise ValueError("Contiguous merge requires nonempty input")
    if order_by not in ("timestamp", "segment_id"):
        raise ValueError("order_by must be timestamp or segment_id")
    if merge_mode not in ("contiguous23", "contiguous5", "full_recording"):
        raise ValueError("merge_mode must be contiguous23, contiguous5 or full_recording")
    groups = defaultdict(list)
    positions = {}
    for index, row in enumerate(rows):
        parent = row.get(group_key)
        path = row.get("audio_path") or row.get("audio_file") or row.get("audio_chunk")
        if order_by == "segment_id":
            name, position = _segment_identity(row)
            # Keep matching producer names in separate audio directories apart.
            parent = (path.rsplit("/", 1)[0] + "/" if isinstance(path, str) and "/" in path else "") + name
        else:
            position = _start(row)
        if not isinstance(parent, str) or not parent:
            raise ValueError(f"Contiguous merge requires nonempty {group_key}")
        if not isinstance(path, str) or not path or ";" in path:
            raise ValueError("Contiguous merge requires one scalar audio path per input utterance")
        _duration(row)
        positions[index] = position
        groups[parent].append(index)
    result = []
    for parent, indices in groups.items():
        indices.sort(key=lambda i: (positions[i], i))
        if len({positions[i] for i in indices}) != len(indices):
            raise ValueError(f"Recording {parent!r} has ambiguous duplicate utterance positions")
        size = len(indices)
        if merge_mode == "contiguous23" and size < 3:
            raise ValueError(f"Recording {parent!r} has {size} utterances; at least three are required")
        if merge_mode == "contiguous5" and size < 5:
            raise ValueError(f"Recording {parent!r} has {size} utterances; at least five are required")
        if merge_mode == "full_recording":
            windows = [(0, size)]
        elif merge_mode == "contiguous5":
            windows = [(start, 5) for start in range(size - 4)]
        else:
            windows = [(start, min(2 + start % 2, size - start)) for start in range(size - 1)]
            # Add the other length near the tail, instead of duplicating a window
            # or wrapping the last utterance back to the recording's beginning.
            start = size - 3
            chosen = windows[start][1]
            windows.append((start, 5 - chosen))
        for start, count in sorted(windows):
            members = indices[start : start + count]
            source_rows = [rows[i] for i in members]
            merged = _merge_rows(
                source_rows, members, [_duration(row) for row in source_rows], " ", default_language=default_language
            )
            merged["merge_count"] = merged.pop("random_merge_count")
            merged["source_indices"] = merged.pop("random_merge_source_indices")
            merged["source_ids"] = [str(rows[i].get("id", i)) for i in members]
            merged["source_metadata_json"] = [
                json.dumps(originals[i], ensure_ascii=False, sort_keys=True) for i in members
            ]
            merged["merge_parent_audio_path"] = parent
            if order_by == "timestamp":
                merged["source_start_seconds"] = [positions[i] for i in members]
            else:
                merged["source_segment_indices"] = [positions[i][0] for i in members]
                merged["source_part_indices"] = [positions[i][1] for i in members]
            merged["source_order_by"] = order_by
            merged["merge_method"] = merge_mode
            # Index-based identity remains unique even if producer IDs repeat.
            identity = hashlib.sha256(json.dumps([parent, members]).encode()).hexdigest()[:16]
            merged["id"] = f"{merge_mode}:{identity}"
            result.append(merged)
    return result


def prepare(
    source,
    output_dir,
    *,
    input_jsonl=None,
    blob_root=None,
    group_key="original_audio_url",
    order_by="timestamp",
    duration_cache=None,
    merge_mode="contiguous23",
):
    """Normalize raw rows, merge before task formatting, and write a datasource."""
    source = deepcopy(source)
    if source.get("dataset_name") != "jsonl":
        raise ValueError("Contiguous merge requires a JSONL datasource")
    post = source.get("post_process") or {}
    if post.get("input_egs_limit") is not None:
        raise ValueError("Apply input limits to the source before contiguous preparation")
    pre = source.get("pre_process") or {}
    if set(pre) - {"path_map", "rename_fields", "clean_tagged_text"}:
        raise ValueError("Only raw path/text normalization is supported before contiguous merging")
    if "clean_tagged_text" in pre:
        pre["clean_tagged_text"] = {**(pre["clean_tagged_text"] or {}), "preserve_existing_keywords": True}
    raw = jsonl_dataset(input_jsonl or source["jsonl_paths"])
    originals = raw.to_list()
    rows = process_ds(raw, **pre).to_list()
    if order_by == "auto":
        order_by = "timestamp" if any(group_key in row or "start_seconds" in row for row in rows) else "segment_id"
    if duration_cache is not None:
        for row in rows:
            if not any(row.get(key) is not None for key in ("audio_length", "audio_duration_seconds", "duration")):
                path = row.get("audio_path") or row.get("audio_file") or row.get("audio_chunk")
                row["audio_length"] = duration_cache.get(path) if isinstance(path, str) else None
    output = contiguous_merge(
        rows,
        originals=originals,
        group_key=group_key,
        default_language=(source.get("add_task_info") or {}).get("language"),
        order_by=order_by,
        merge_mode=merge_mode,
    )
    if merge_mode == "contiguous23":
        assert len(output) == len(rows)
    elif merge_mode == "contiguous5":
        assert len(output) == len(rows) - 4 * len({row["merge_parent_audio_path"] for row in output})
    else:
        assert len(output) == len({row["merge_parent_audio_path"] for row in output})
        assert sorted(i for row in output for i in row["source_indices"]) == list(range(len(rows)))
    assert {i for row in output for i in row["source_indices"]} == set(range(len(rows)))
    assert len({tuple(row["source_indices"]) for row in output}) == len(output)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = root / "data.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    assert jsonl_dataset(str(manifest)).to_list() == output
    compressed = root / "data.jsonl.gz"
    with manifest.open("rb") as source_file, compressed.open("wb") as target:
        with gzip.GzipFile(filename="", fileobj=target, mode="wb", mtime=0) as stream:
            shutil.copyfileobj(source_file, stream)
    block = {
        "dataset_name": "jsonl",
        "jsonl_paths": f"{blob_root.rstrip('/')}/data.jsonl.gz" if blob_root else str(manifest),
        "add_task_info": source.get("add_task_info") or {},
        "post_process": post,
    }
    (root / "train_data.yaml").write_text(yaml.safe_dump([block], sort_keys=False, allow_unicode=True))
    stats = {
        "method": merge_mode,
        "group_key": group_key if order_by == "timestamp" else "audio_directory + recording_name_from_id",
        "order_by": order_by,
        "source": source,
        "input_rows": len(rows),
        "output_rows": len(output),
        "recordings": len({row["merge_parent_audio_path"] for row in output}),
        "merge_count_distribution": dict(Counter(row["merge_count"] for row in output)),
        "input_audio_seconds": math.fsum(_duration(row) for row in rows),
        "output_audio_seconds": math.fsum(row["audio_length"] for row in output),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "compressed_sha256": hashlib.sha256(compressed.read_bytes()).hexdigest(),
        "normalized_source_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "audit": {
            "same_sample_count": len(output) == len(rows),
            "all_inputs_covered": True,
            "unique_windows": True,
            "jsonl_roundtrip": True,
            "each_input_once": merge_mode == "full_recording",
        },
    }
    (root / "preparation.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, nargs="+", required=True, help="YAML files of JSONL datasources")
    parser.add_argument("--input-jsonl", help="Optional local cache of the original source manifest")
    parser.add_argument("--input-map", type=Path, help="JSON mapping source manifest URIs to local cached files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blob-root", help="Remote manifest location; upload the generated files separately")
    parser.add_argument("--group-key", default="original_audio_url")
    parser.add_argument("--order-by", choices=("timestamp", "segment_id", "auto"), default="timestamp")
    parser.add_argument(
        "--merge-mode",
        choices=("contiguous23", "contiguous5", "full_recording"),
        default="contiguous23",
        help="Unique overlapping 2/3- or 5-utterance windows, or one audio-path list per complete recording",
    )
    parser.add_argument(
        "--duration-cache", type=Path, help="JSON mapping normalized audio paths to durations in seconds"
    )
    args = parser.parse_args()
    sources = []
    for path in args.source_config:
        blocks = yaml.safe_load(path.read_text())
        if not isinstance(blocks, list) or not blocks or any(not isinstance(block, dict) for block in blocks):
            parser.error("source-config must contain a nonempty list of datasource blocks")
        sources.extend(blocks)
    if args.input_jsonl and len(sources) != 1:
        parser.error("input-jsonl requires exactly one datasource; use input-map for multiple sources")
    cache = json.loads(args.duration_cache.read_text()) if args.duration_cache else None
    inputs = json.loads(args.input_map.read_text()) if args.input_map else {}
    if len(sources) > 1:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    records, prepared = [], []
    for index, source in enumerate(sources):
        child = f"source_{index:03d}"
        output = args.output_dir / child if len(sources) > 1 else args.output_dir
        blob = f"{args.blob_root.rstrip('/')}/{child}" if args.blob_root and len(sources) > 1 else args.blob_root
        stats = prepare(
            source,
            output,
            input_jsonl=args.input_jsonl or inputs.get(source["jsonl_paths"]),
            blob_root=blob,
            group_key=args.group_key,
            order_by=args.order_by,
            duration_cache=cache,
            merge_mode=args.merge_mode,
        )
        records.append(stats)
        prepared.extend(yaml.safe_load((output / "train_data.yaml").read_text()))
    if len(sources) > 1:
        stats = {
            "method": args.merge_mode,
            "input_rows": sum(record["input_rows"] for record in records),
            "output_rows": sum(record["output_rows"] for record in records),
            "recordings": sum(record["recordings"] for record in records),
            "merge_count_distribution": dict(
                sum((Counter(record["merge_count_distribution"]) for record in records), Counter())
            ),
            "sources": records,
        }
        (args.output_dir / "train_data.yaml").write_text(yaml.safe_dump(prepared, sort_keys=False, allow_unicode=True))
        (args.output_dir / "preparation.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps({key: stats[key] for key in ("input_rows", "output_rows", "recordings", "merge_count_distribution")})
    )


if __name__ == "__main__":
    main()

"""Split prepared full-recording audio lists into consecutive bounded samples."""

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path

import yaml
from datasets import Dataset
from recipe.phimm.data.dataset import process_ds
from recipe.phimm.tools.random_merge_list import _merge_rows


def recording_windows(parent, rows, *, max_audio_length=600.0, default_language="English"):
    """Use each consecutive short clip once; the cap is summed audio duration."""
    if not math.isfinite(max_audio_length) or max_audio_length <= 0:
        raise ValueError("max_audio_length must be finite and positive")
    paths = parent["audio_path"]
    durations = parent["source_audio_lengths"]
    if not rows or len(rows) != len(paths) or len(rows) != len(durations):
        raise ValueError("Parent paths, durations and normalized rows must align")
    if [row["audio_path"] for row in rows] != paths or " ".join(row["text"] for row in rows) != parent["text"]:
        raise ValueError("Normalized rows must preserve the prepared parent order and reference")
    windows, members = [], []
    for index, duration in enumerate(durations):
        if not math.isfinite(duration) or not 0 < duration <= max_audio_length:
            raise ValueError(f"Short segment {index} cannot fit the duration cap: {duration}")
        if members and math.fsum(durations[i] for i in [*members, index]) > max_audio_length:
            windows.append(members)
            members = []
        members.append(index)
    windows.append(members)
    output = []
    for window_index, members in enumerate(windows):
        indices = [parent["source_indices"][i] for i in members]
        result = _merge_rows(
            [rows[i] for i in members], indices, [durations[i] for i in members], " ", default_language=default_language
        )
        result["merge_count"] = result.pop("random_merge_count")
        result["source_indices"] = result.pop("random_merge_source_indices")
        for key in (
            "source_ids",
            "source_metadata_json",
            "source_start_seconds",
            "source_segment_indices",
            "source_part_indices",
        ):
            if key in parent:
                result[key] = [parent[key][i] for i in members]
        result.update(
            id="contiguous10m:" + hashlib.sha256(json.dumps([parent["id"], indices]).encode()).hexdigest()[:20],
            merge_parent_audio_path=parent["merge_parent_audio_path"],
            source_order_by=parent["source_order_by"],
            merge_window_index=window_index,
            merge_method="contiguous_duration_bounded",
            max_audio_length=max_audio_length,
        )
        output.append(result)
    assert [path for row in output for path in row["audio_path"]] == paths
    assert " ".join(row["text"] for row in output) == parent["text"]
    return output


def prepare(input_root, output_root, *, blob_root=None, max_audio_length=600.0):
    """Reuse audited full-recording manifests and their raw source provenance."""
    input_root, output_root = Path(input_root), Path(output_root)
    provenance = json.loads((input_root / "preparation.json").read_text())
    output_root.mkdir(parents=True, exist_ok=False)
    sources, stats = [], []
    for index, source_stats in enumerate(provenance["sources"]):
        source = deepcopy(source_stats["source"])
        original_path = input_root / f"source_{index:03d}/data.jsonl"
        assert hashlib.sha256(original_path.read_bytes()).hexdigest() == source_stats["manifest_sha256"]
        parents = [json.loads(line) for line in original_path.read_text().splitlines()]
        originals = [json.loads(value) for parent in parents for value in parent["source_metadata_json"]]
        normalized = process_ds(Dataset.from_list(originals), **source.get("pre_process", {})).to_list()
        output, offset = [], 0
        for parent in parents:
            size = len(parent["audio_path"])
            output.extend(
                recording_windows(
                    parent,
                    normalized[offset : offset + size],
                    max_audio_length=max_audio_length,
                    default_language=source.get("add_task_info", {}).get("language", "English"),
                )
            )
            offset += size
        assert offset == len(normalized) == source_stats["input_rows"]
        assert Counter(i for row in output for i in row["source_indices"]) == Counter(range(offset))
        directory = output_root / f"source_{index:03d}"
        directory.mkdir()
        path = directory / "data.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in output))
        compressed = directory / "data.jsonl.gz"
        compressed.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
        source.pop("pre_process", None)
        source["jsonl_paths"] = (
            f"{blob_root.rstrip('/')}/{directory.name}/data.jsonl.gz" if blob_root else str(path.resolve())
        )
        sources.append(source)
        stats.append(
            {
                "source": index,
                "name": source["post_process"]["add_field"]["fields"]["data_source"],
                "input_recordings": len(parents),
                "input_segments": offset,
                "output_samples": len(output),
                "audio_seconds": math.fsum(row["audio_length"] for row in output),
                "min_sample_seconds": min(row["audio_length"] for row in output),
                "max_sample_seconds": max(row["audio_length"] for row in output),
                "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "compressed_sha256": hashlib.sha256(compressed.read_bytes()).hexdigest(),
            }
        )
    (output_root / "train_data.yaml").write_text(yaml.safe_dump(sources, sort_keys=False))
    summary = {
        "method": "consecutive_short_segments_up_to_10_minutes",
        "duration_cap_seconds": max_audio_length,
        "duration_definition": "Sum of original short clip durations; no waveform concatenation.",
        "input_root": str(input_root),
        "input_segments": sum(s["input_segments"] for s in stats),
        "input_recordings": sum(s["input_recordings"] for s in stats),
        "output_samples": sum(s["output_samples"] for s in stats),
        "audio_seconds": math.fsum(s["audio_seconds"] for s in stats),
        "sources": stats,
        "audit": {
            "each_input_exactly_once": True,
            "order_preserved": True,
            "references_preserved": True,
            "no_audio_files_created": True,
        },
    }
    (output_root / "preparation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--blob-root")
    parser.add_argument("--max-audio-length", type=float, default=600)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))


if __name__ == "__main__":
    main()

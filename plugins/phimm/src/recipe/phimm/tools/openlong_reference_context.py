"""Prepare complete OpenLong parent groups with offline preceding-reference context.

Reuse audited segment alignments; do not infer text from audio or split parent
references heuristically. Audio selectors and full-parent scoring text are retained.
"""

import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path

import blobfile as bf
import yaml

from recipe.phimm.data.u2_full_reference import publish_identical, read_rows

ALIGNMENT_ROOT = (
    "az://orngwus2cresco/data/boren/data/Evaluation/OpenLong_testref_baseline_qwen_consensus_2609r2b_v1/full"
)
PARENT_ROOT = "az://orngwus2cresco/data/boren/data/openasr_long_jsonl"
RELEASE_ROOT = "az://orngwus2cresco/data/boren/data/Evaluation/OpenLong_reference_text_consensus_v2"
EXPECTED = {"earnings21": (44, 5552), "earnings22": (125, 17099), "coraal": (271, 22041)}
METADATA_KEYS = (
    "id",
    "audio_path",
    "text",
    "seg_index",
    "seg_start",
    "seg_end",
    "n_segments",
    "reference_word_start",
    "reference_word_end",
    "alignment_method",
    "alignment_status",
    "alignment_flags",
    "alignment_diagnostics",
)


def file_sha256(uri):
    """Hash the exact source manifest bytes without loading the file into memory."""
    digest = hashlib.sha256()
    with bf.BlobFile(uri, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parent_context_rows(aligned_rows, original_rows):
    """Validate complete aligned references and emit one ordered row per parent."""
    originals = {}
    for row in original_rows:
        path = row["audio_path"]
        if path in originals or not isinstance(row.get("text"), str):
            raise ValueError("Original parents must be unique and have string references")
        originals[path] = row
    groups = collections.defaultdict(list)
    for row in aligned_rows:
        groups[row["parent_audio_path"]].append(row)
    if not groups or set(groups) != set(originals):
        raise ValueError("Aligned segments must cover every original parent exactly")
    output, audit = [], []
    for path in sorted(groups):
        segments = groups[path]
        if any(type(row.get("seg_index")) is not int for row in segments):
            raise ValueError(f"Invalid segment indices for {path}")
        by_index = sorted(segments, key=lambda row: row["seg_index"])
        if [row["seg_index"] for row in by_index] != list(range(len(segments))) or any(
            type(row.get("n_segments")) is not int or row["n_segments"] != len(segments) for row in segments
        ):
            raise ValueError(f"Incomplete or duplicate segments for {path}")
        words = originals[path]["text"].split()
        cursor = 0
        for row in by_index:
            start, end = row.get("reference_word_start"), row.get("reference_word_end")
            # Audited empty segments have no word span, but still occupy a history turn.
            if start is None and end is None and isinstance(row.get("text"), str) and not row["text"].split():
                start = end = cursor
            if (
                type(start) is not int
                or type(end) is not int
                or start != cursor
                or end < start
                or end > len(words)
                or not isinstance(row.get("text"), str)
                or row["text"].split() != words[start:end]
            ):
                raise ValueError(f"Aligned text must partition the exact parent reference for {path}")
            cursor = end
            begin, finish = float(row["seg_start"]), float(row["seg_end"])
            if not math.isfinite(begin) or not math.isfinite(finish) or begin < 0 or finish <= begin:
                raise ValueError(f"Invalid segment times for {path}")
            audio = row["audio_path"]
            selector = audio.rsplit("#", 1)[-1].split(":") if isinstance(audio, str) else []
            if len(selector) != 2 or [float(value) for value in selector] != [begin, finish]:
                raise ValueError(f"Audio selector does not match aligned segment times for {path}")
            if not isinstance(row.get("alignment_method"), str) or not row["alignment_method"]:
                raise ValueError(f"Missing alignment provenance for {path}")
        if cursor != len(words):
            raise ValueError(f"Incomplete aligned parent reference for {path}")
        # Match group_validation_rows, including rare non-monotonic producer indices.
        ordered = sorted(segments, key=lambda row: (float(row["seg_start"]), row["seg_index"]))
        references = [row["text"] for row in ordered]
        new = {
            **originals[path],
            "parent_audio_path": path,
            "audio_path": [row["audio_path"] for row in ordered],
            "reference_scope": "parent",
            "n_segments": len(ordered),
            "seg_start": min(float(row["seg_start"]) for row in ordered),
            "seg_end": max(float(row["seg_end"]) for row in ordered),
            "segment_reference_texts": references,
            "source_metadata_json": [
                json.dumps({key: row[key] for key in METADATA_KEYS if key in row}, ensure_ascii=False)
                for row in ordered
            ],
        }
        output.append(new)
        audit.append(
            {
                "parent_audio_path": path,
                "segments": len(ordered),
                "reference_sha256": hashlib.sha256(new["text"].encode()).hexdigest(),
                "reference_words": len(words),
                "alignment_status_counts": dict(collections.Counter(row.get("alignment_status") for row in ordered)),
                "alignment_flag_counts": dict(
                    collections.Counter(flag for row in ordered for flag in row.get("alignment_flags", []))
                ),
                "reordered_by_timestamp": ordered != by_index,
                "segment_indices": [row["seg_index"] for row in ordered],
            }
        )
    return output, audit


def validation_source(source, uri):
    """Return the canonical verbatim English formatting for a prepared corpus."""
    return {
        "dataset_name": "jsonl",
        "jsonl_block_size": 64 * 1024 * 1024,
        "val_batch_size": -1,
        "jsonl_paths": uri,
        "add_task_info": {"style": "verb", "language": "English", "lang_hint": True},
        "post_process": {
            "add_field": {"fields": {"data_source": source}},
            "verl_format": {
                "prompt_key": "prompt",
                "extra_keys": [
                    "id",
                    "language",
                    "audio_path",
                    "parent_audio_path",
                    "n_segments",
                    "seg_start",
                    "seg_end",
                    "reference_scope",
                    "segment_reference_texts",
                    "source_metadata_json",
                ],
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-root", default=ALIGNMENT_ROOT)
    parser.add_argument("--parent-root", default=PARENT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publish-root", help="Publish verified immutable manifests to this Blob root")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "reference_scope": "parent",
        "context_source": "preceding_reference",
        "alignment_root": args.alignment_root,
        "parent_root": args.parent_root,
        "alignment_limit": "Reuses automatic consensus boundaries and retains all quality flags; not human verified.",
        "datasets": {},
    }
    sources = []
    for source, expected in EXPECTED.items():
        alignment_uri = f"{args.alignment_root.rstrip('/')}/{source}.jsonl"
        aligned = read_rows(alignment_uri)
        parent_files = sorted(
            {
                row["parent_audio_path"].split("/openasr_long_jsonl/", 1)[1].rsplit("/audio/", 1)[0] + "/data.jsonl"
                for row in aligned
            }
        )
        originals, inputs = [], []
        for relative in parent_files:
            uri = f"{args.parent_root.rstrip('/')}/{relative}"
            rows = read_rows(uri)
            originals.extend(rows)
            inputs.append({"uri": uri, "rows": len(rows), "sha256": file_sha256(uri)})
        prepared, audit = parent_context_rows(aligned, originals)
        if (len(prepared), len(aligned)) != expected:
            raise ValueError(f"Incomplete canonical {source}: expected {expected}, got {(len(prepared), len(aligned))}")
        filename = f"{source}.jsonl.gz"
        target = args.output_dir / filename
        # Stream parent rows to keep memory bounded; gzip metadata is deterministic.
        with target.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            for row in prepared:
                stream.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode())
        payload = target.read_bytes()
        uri = f"{args.publish_root.rstrip('/')}/{filename}" if args.publish_root else str(target.resolve())
        sources.append(validation_source(source, uri))
        summary["datasets"][source] = {
            "manifest": uri,
            "manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "compressed_bytes": len(payload),
            "parents": len(prepared),
            "segments": len(aligned),
            "alignment_manifest": alignment_uri,
            "alignment_manifest_sha256": file_sha256(alignment_uri),
            "original_inputs": inputs,
            "parent_audit": audit,
        }
        print(json.dumps({"source": source, "parents": len(prepared), "segments": len(aligned)}), flush=True)
    summary_bytes = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode()
    (args.output_dir / "preparation.json").write_bytes(summary_bytes)
    (args.output_dir / "val_data.yaml").write_text(yaml.safe_dump(sources, sort_keys=False, allow_unicode=True))
    if args.publish_root:
        # Publish only after all three complete corpora have been prepared and checked.
        for source in EXPECTED:
            filename = f"{source}.jsonl.gz"
            publish_identical(f"{args.publish_root.rstrip('/')}/{filename}", (args.output_dir / filename).read_bytes())
        publish_identical(f"{args.publish_root.rstrip('/')}/preparation.json", summary_bytes)
    print(f"Prepared validation sources: {args.output_dir.resolve() / 'val_data.yaml'}", flush=True)


if __name__ == "__main__":
    main()

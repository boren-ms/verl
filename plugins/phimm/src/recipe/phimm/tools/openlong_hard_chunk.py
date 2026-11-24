"""Prepare complete OpenLong recordings with fixed-duration, non-overlapping cuts.

Reuse ZIP member locations and exact parent references from the canonical parent
manifests. Their old segment boundaries and reference histories are not used.
One JSONL row per recording avoids repeating the full transcript per chunk; the
ASR validation loader expands its audio_path list and scores the joined outputs.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import blobfile as bf
import yaml

from recipe.phimm.data.u2_full_reference import publish_identical, read_rows
from verl.audio_zip import split_zip_audio_source

INPUT_ROOT = "az://orngwus2cresco/data/boren/data/Evaluation/OpenLong_reference_text_consensus_v2"
RELEASE_ROOT = "az://orngwus2cresco/data/boren/data/Evaluation/OpenLong_hard_chunk_2s_v1"
EXPECTED_PARENTS = {"earnings21": 44, "earnings22": 125, "coraal": 271}


def hard_chunk_parent(parent, chunk_seconds=2.0):
    """Cut the entire recording on the sample grid, retaining a shorter tail."""
    if isinstance(chunk_seconds, bool) or not math.isfinite(chunk_seconds) or chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be finite and positive")
    duration = parent.get("duration")
    rate = parent.get("sampling_rate")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, int | float)
        or not math.isfinite(duration)
        or duration <= 0
        or type(rate) is not int
        or rate <= 0
    ):
        raise ValueError("A positive original duration and integer sampling_rate are required")
    frames = round(duration * rate)
    step = round(chunk_seconds * rate)
    if frames <= 0 or not math.isclose(duration * rate, frames, abs_tol=1e-5, rel_tol=0):
        raise ValueError("Original duration must describe a whole number of audio samples")
    if step <= 0 or not math.isclose(chunk_seconds * rate, step, abs_tol=1e-5, rel_tol=0):
        raise ValueError("Chunk duration must describe a whole number of audio samples")
    paths = parent.get("audio_path")
    if not isinstance(paths, list) or not paths or any(not isinstance(path, str) for path in paths):
        raise ValueError("Expected canonical parent rows with a list of ZIP audio selectors")
    members = {path.rsplit("#", 1)[0] for path in paths}
    if len(members) != 1:
        raise ValueError("All old segments must reference the same complete ZIP member")
    member = members.pop()
    if split_zip_audio_source(member) is None:
        raise ValueError("Expected a ZIP member reference")
    if not isinstance(parent.get("text"), str) or not isinstance(parent.get("parent_audio_path"), str):
        raise ValueError("Exact full-parent text and parent_audio_path are required")
    # Copy only recording-level fields; never reuse old times, histories or alignments.
    new = {key: parent[key] for key in ("id", "dataset", "language") if key in parent}
    new.update(
        text=parent["text"],
        parent_audio_path=parent["parent_audio_path"],
        reference_scope="parent",
        sampling_rate=rate,
        duration=duration,
        audio_length_s=duration,
        seg_start=0.0,
        seg_end=duration,
        chunking_method="hard_nonoverlapping",
        chunk_length_seconds=chunk_seconds,
        audio_path=[f"{member}#{start / rate}:{min(start + step, frames) / rate}" for start in range(0, frames, step)],
    )
    new["n_segments"] = len(new["audio_path"])
    return new


def validation_source(source, uri):
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
                    "chunking_method",
                    "chunk_length_seconds",
                ],
            },
        },
    }


def prepare(input_root, output_dir, *, chunk_seconds=2.0, publish_root=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "chunking_method": "hard_nonoverlapping",
        "chunk_length_seconds": chunk_seconds,
        "tail_policy": "retain_shorter_final_chunk",
        "reference_scope": "parent",
        "input_root": input_root,
        "audio_storage": "Existing complete ZIP members with new time selectors",
        "datasets": {},
    }
    local_sources, sources = [], []
    for source, expected in EXPECTED_PARENTS.items():
        input_uri = f"{input_root.rstrip('/')}/{source}.jsonl.gz"
        parents = read_rows(input_uri)
        if len(parents) != expected or len({row["parent_audio_path"] for row in parents}) != expected:
            raise ValueError(f"Expected {expected} unique complete {source} recordings")
        filename = f"{source}.jsonl.gz"
        target = output_dir / filename
        audit = []
        digest = hashlib.sha256()
        with target.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            for parent in sorted(parents, key=lambda row: row["parent_audio_path"]):
                row = hard_chunk_parent(parent, chunk_seconds)
                data = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode()
                digest.update(data)
                stream.write(data)
                audit.append(
                    {
                        "parent_audio_path": row["parent_audio_path"],
                        "duration": row["duration"],
                        "sampling_rate": row["sampling_rate"],
                        "segments": row["n_segments"],
                        "reference_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                    }
                )
        uri = f"{publish_root.rstrip('/')}/{filename}" if publish_root else str(target.resolve())
        sources.append(validation_source(source, uri))
        local_sources.append(validation_source(source, str(target.resolve())))
        summary["datasets"][source] = {
            "manifest": uri,
            "input_manifest": input_uri,
            "input_rows_sha256": hashlib.sha256(
                json.dumps(parents, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
            "manifest_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "uncompressed_sha256": digest.hexdigest(),
            "compressed_bytes": target.stat().st_size,
            "parents": len(audit),
            "segments": sum(row["segments"] for row in audit),
            "audio_seconds": math.fsum(row["duration"] for row in audit),
            "parent_audit": audit,
        }
        print(
            json.dumps(
                {key: summary["datasets"][source][key] for key in ("parents", "segments", "audio_seconds")},
                sort_keys=True,
            )
            + f" {source}",
            flush=True,
        )
    summary_bytes = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode()
    (output_dir / "preparation.json").write_bytes(summary_bytes)
    header = "# @package data.val_data\n# Fixed hard cuts; full-parent references and shorter final tails retained.\n"
    (output_dir / "val_data.yaml").write_text(header + yaml.safe_dump(sources, sort_keys=False))
    (output_dir / "val_data_local.yaml").write_text(header + yaml.safe_dump(local_sources, sort_keys=False))
    if publish_root:
        # Prepare every corpus before publishing; refuse to replace different files.
        for source in EXPECTED_PARENTS:
            filename = f"{source}.jsonl.gz"
            publish_identical(f"{publish_root.rstrip('/')}/{filename}", (output_dir / filename).read_bytes())
        for filename in ("preparation.json", "val_data.yaml"):
            publish_identical(f"{publish_root.rstrip('/')}/{filename}", (output_dir / filename).read_bytes())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=INPUT_ROOT)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-seconds", default=2.0, type=float)
    parser.add_argument("--publish-root", help=f"For the standard 2-second release: {RELEASE_ROOT}")
    args = parser.parse_args()
    # Smaller blocks prevent write timeouts on slower links to Orange storage.
    bf.configure(azure_write_chunk_size=512 * 1024, read_timeout=180)
    prepare(args.input_root, args.output_dir, chunk_seconds=args.chunk_seconds, publish_root=args.publish_root)


if __name__ == "__main__":
    main()

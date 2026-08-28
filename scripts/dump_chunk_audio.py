#!/usr/bin/env python3
"""Dump audio files from a chunk dataset to a local folder and create a JSONL manifest.

Usage:
    python scripts/dump_chunk_audio.py \
        --manifest_file az://orngwus2cresco/data/speech/users/kskumar/aed_data/unlabeled_data_proc/temp_1/Partition0/AED/file_set.json \
        --chunk_path az://orngwus2cresco/data/speech/users/kskumar/aed_data/unlabeled_data_proc/temp_1/Partition0/AED/ChunkFiles/ \
        --output_dir ./dump_ghost_words \
        --max_egs 100
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))

from recipe.phimm.data.chunk import (
    load_chunk_info,
    load_data_from_chunk,
    get_chunk_type_path,
    parse_data,
)


def _is_remote(path):
    return path.startswith("az://") or path.startswith("s3://") or path.startswith("gs://")


def _write_bytes(filepath, data):
    """Write bytes to local or remote path."""
    import blobfile as bf
    with bf.BlobFile(filepath, "wb") as f:
        f.write(data)


def _write_text(filepath, text):
    """Write text to local or remote path."""
    import blobfile as bf
    with bf.BlobFile(filepath, "w") as f:
        f.write(text)


def dump_chunk_audio(manifest_file, chunk_path, output_dir, max_egs=None, chunk_types=None):
    """Dump audio and transcription from chunk dataset to files + JSONL."""
    import soundfile as sf
    import blobfile as bf

    chunk_types = chunk_types or ["audio", "info"]
    remote = _is_remote(output_dir)

    audio_dir = output_dir.rstrip("/") + "/audio"
    jsonl_path = output_dir.rstrip("/") + "/data.jsonl"

    if not remote:
        os.makedirs(audio_dir, exist_ok=True)

    # Load chunk manifest
    chunks = load_chunk_info(manifest_file=manifest_file, chunk_path=chunk_path)
    print(f"Found {len(chunks)} chunks")

    total_egs = sum(c["count"] for c in chunks)
    print(f"Total examples across all chunks: {total_egs}")
    if max_egs:
        print(f"Will dump up to {max_egs} examples")

    jsonl_lines = []
    global_idx = 0
    for chunk in tqdm(chunks, desc="Processing chunks"):
        if max_egs and global_idx >= max_egs:
            break

        count = chunk["count"]
        name = chunk["name"]

        # Load audio
        audio_chunk_file = chunk_path.rstrip("/") + f"/{name}.audio"
        audio_data = load_data_from_chunk(audio_chunk_file, "audio", count)

        # Try loading transcription/info
        info_data = [None] * count
        for ct in ["info", "transcription"]:
            ct_path = get_chunk_type_path(chunk, ct)
            if ct_path is None:
                continue
            ct_file = ct_path.rstrip("/") + f"/{name}.{ct}"
            try:
                if bf.exists(ct_file):
                    info_data = load_data_from_chunk(ct_file, ct, count)
                    break
            except Exception:
                continue

        for i in range(count):
            if max_egs and global_idx >= max_egs:
                break

            audio_array, sample_rate = audio_data[i]
            audio_filename = f"{name}_{i:04d}.wav"
            audio_filepath = audio_dir.rstrip("/") + "/" + audio_filename

            # Write audio to buffer then to dest
            buf = io.BytesIO()
            sf.write(buf, audio_array, sample_rate, format="WAV")
            _write_bytes(audio_filepath, buf.getvalue())

            # Build JSONL record
            record = {
                "id": global_idx,
                "audio_path": audio_filepath,
                "sample_rate": sample_rate,
                "duration": len(audio_array) / sample_rate,
                "chunk_name": name,
                "chunk_index": i,
            }
            if info_data[i] is not None:
                if isinstance(info_data[i], dict):
                    record.update(info_data[i])
                else:
                    record["transcription"] = str(info_data[i])

            jsonl_lines.append(json.dumps(record, ensure_ascii=False))
            global_idx += 1

    # Write JSONL
    _write_text(jsonl_path, "\n".join(jsonl_lines) + "\n")

    print(f"Dumped {global_idx} examples to {output_dir}")
    print(f"  Audio files: {audio_dir}/")
    print(f"  JSONL manifest: {jsonl_path}")


def main():
    parser = argparse.ArgumentParser(description="Dump chunk audio dataset to local files")
    parser.add_argument("--manifest_file", type=str, required=True, help="Path to file_set.json")
    parser.add_argument("--chunk_path", type=str, required=True, help="Path to ChunkFiles/ directory")
    parser.add_argument("--output_dir", type=str, default="./dump_output", help="Output directory")
    parser.add_argument("--max_egs", type=int, default=None, help="Max number of examples to dump")
    args = parser.parse_args()

    dump_chunk_audio(
        manifest_file=args.manifest_file,
        chunk_path=args.chunk_path,
        output_dir=args.output_dir,
        max_egs=args.max_egs,
    )


if __name__ == "__main__":
    main()

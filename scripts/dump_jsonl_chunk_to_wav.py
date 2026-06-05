#!/usr/bin/env python3
"""Dump audio chunks from a JSONL file to individual wav files.

For each record with an ``audio_chunk`` field (format ``chunk_file:count:index``),
extract the audio, write it as a wav to a ``wavs/`` folder alongside the input
JSONL, and produce a new JSONL (suffixed ``_wav``) with ``audio_path`` pointing
to the extracted wav.

Usage:
    python scripts/dump_jsonl_chunk_to_wav.py \
        az://orngwus2cresco/data/boren/data/verl/cached_qwen/qwen_oss_brackets/all.jsonl

    # Test with a small subset
    python scripts/dump_jsonl_chunk_to_wav.py \
        az://orngwus2cresco/data/boren/data/verl/cached_qwen/qwen_oss_brackets/all.jsonl \
        --max_egs 10
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))

import blobfile as bf
import soundfile as sf

from recipe.phimm.data.chunk import load_chunk_sample, resolve_path
from recipe.phimm.utils.audio import _is_chunk_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", help="Path to input JSONL (local or az://)")
    parser.add_argument("--max_egs", type=int, default=None, help="Max examples to process")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")
    args = parser.parse_args()

    input_path = args.input_jsonl

    # Derive output paths alongside the input jsonl
    if input_path.endswith(".jsonl"):
        stem = input_path[:-6]
    else:
        stem = input_path
    output_jsonl = stem + "_wav.jsonl"
    wavs_dir = stem.rsplit("/", 1)[0] + "/wavs"

    # Ensure wavs dir exists (local only; blobfile handles remote transparently)
    if not input_path.startswith("az://"):
        os.makedirs(wavs_dir, exist_ok=True)

    # Read input jsonl
    print(f"Reading {input_path} ...")
    with bf.BlobFile(input_path, "r") as f:
        lines = f.readlines()
    print(f"  {len(lines)} records found")

    if args.max_egs:
        lines = lines[: args.max_egs]
        print(f"  Processing first {len(lines)} records")

    out_lines = []
    skipped = 0

    for idx, line in enumerate(tqdm(lines, desc="Dumping wavs")):
        record = json.loads(line)
        chunk_spec = record.get("audio_chunk")

        if not chunk_spec or not _is_chunk_spec(chunk_spec):
            # No chunk to extract — keep record as-is
            out_lines.append(json.dumps(record, ensure_ascii=False))
            skipped += 1
            continue

        # Load audio from chunk
        result = load_chunk_sample(chunk_spec)
        if isinstance(result, list):
            result = result[0]
        data, sr = result

        # Write wav — reuse chunk spec as filename: "path:count:index" -> "path_count_index.wav"
        wav_filename = chunk_spec.rsplit("/", 1)[-1].replace(":", "_") + ".wav"
        wav_path = wavs_dir + "/" + wav_filename

        buf = io.BytesIO()
        sf.write(buf, data, sr, format="WAV")
        with bf.BlobFile(wav_path, "wb") as wf:
            wf.write(buf.getvalue())

        # Update record: replace audio_chunk with audio_path
        record["audio_path"] = wav_path
        record.pop("audio_chunk", None)
        out_lines.append(json.dumps(record, ensure_ascii=False))

    # Write output jsonl
    print(f"Writing {output_jsonl} ...")
    with bf.BlobFile(output_jsonl, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"Done: {len(out_lines)} records ({skipped} skipped, {len(out_lines) - skipped} wavs dumped)")
    print(f"  Wavs:  {wavs_dir}/")
    print(f"  JSONL: {output_jsonl}")


if __name__ == "__main__":
    main()

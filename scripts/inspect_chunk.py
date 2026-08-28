#!/usr/bin/env python3
"""Inspect a chunk-based ASR dataset from a JSON config (az:// or local).

Reuses recipe.phimm.data.chunk.create_chunk_datasets for loading.

Usage:
    python scripts/inspect_chunk_dataset.py <config_json> [--n 5] [--chunks 2]
    python scripts/inspect_chunk_dataset.py <config_json> --save_audio /tmp/audio/
"""
from __future__ import annotations

import os
import sys
import textwrap
from collections import Counter
from pathlib import Path

import fire
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parents[0] / ".."))
from recipe.phimm.data.chunk import create_chunk_datasets


def summarize(ds, n=5, save_audio=None):
    """Print dataset summary and first N samples."""
    if save_audio:
        os.makedirs(save_audio, exist_ok=True)

    print("=" * 70)
    print("  Dataset Summary")
    print("=" * 70)
    print(f"  Total samples : {len(ds)}")
    print(f"  Fields        : {ds.column_names}")

    if "language" in ds.column_names:
        print(f"  Languages     : {dict(Counter(ds['language']))}")

    text_col = next((c for c in ds.column_names if c not in ("audio_chunk", "language")), None)
    if text_col:
        texts = [str(t) for t in ds[text_col] if t]
        lens = [len(t.split()) for t in texts]
        if lens:
            print(f"  Transcript words ({text_col}): min={min(lens)}, max={max(lens)}, avg={sum(lens)/len(lens):.1f}")

    print("=" * 70)
    print(f"\n  First {n} samples:")
    print("-" * 70)

    for i in range(min(n, len(ds))):
        sample = ds[i]
        print(f"\n--- Sample {i} ---")
        for k, v in sample.items():
            if k == "audio_chunk" and save_audio and v:
                from recipe.phimm.data.chunk import load_chunk_sample
                audio = load_chunk_sample(v)
                if isinstance(audio, list):
                    audio_data, sr = audio[0]
                else:
                    audio_data, sr = audio
                out = os.path.join(save_audio, f"sample_{i}.wav")
                sf.write(out, audio_data, sr)
                print(f"  audio_chunk: {v}")
                print(f"  audio: saved {out} (sr={sr}, {len(audio_data)/sr:.2f}s)")
            else:
                val = textwrap.shorten(str(v), width=120, placeholder="...") if isinstance(v, str) else v
                print(f"  {k}: {val}")


def main(*config_json, n=5, chunks=None, chunk_types=None, save_audio=None):
    """Inspect a chunk-based ASR dataset.

    Args:
        config_json: Path(s) to dataset config JSON (local or az://).
        n: Number of samples to display.
        chunks: Max chunks to load.
        chunk_types: Chunk types to load (default: sft.0.messages.1.content).
        save_audio: Directory to save audio files for displayed samples.
    """
    config_json = list(config_json)
    chunk_types = chunk_types or ["sft.0.messages.1.content"]
    if isinstance(chunk_types, str):
        chunk_types = [chunk_types]
    if save_audio:
        chunk_types = ["audios"] + [ct for ct in chunk_types if ct != "audios"]

    print(f"Loading from: {config_json}")
    print(f"  max_chunks={chunks}, chunk_types={chunk_types}\n")

    ds = create_chunk_datasets(
        config_json,
        chunk_types=chunk_types,
        max_chunks=chunks,
        chunk_shuffle=False,
    )
    summarize(ds, n=n, save_audio=save_audio)


if __name__ == "__main__":
    fire.Fire(main)

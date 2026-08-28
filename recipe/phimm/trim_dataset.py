"""Prepare a trimmed-audio parquet dataset from TSV data configs.

Usage:
    python -m recipe.phimm.trim_dataset \
        --tsv_path az://orngwus2cresco/data/boren/data/LibriSpeech/debug.tsv \
        --output_dir az://orngwus2cresco/data/boren/data/verl/trimmed_output \
        --jobs 64
"""

import ast
from pathlib import Path

import blobfile as bf
import fire
import pandas as pd
from datasets import Dataset

from recipe.phimm.utils.trim_silence import SilenceTrimmer


def load_tsv(tsv_path: str) -> Dataset:
    """Load TSV (id, paths, msgs) into a Dataset with (id, audio_path, text)."""
    with bf.BlobFile(tsv_path, "r") as f:
        df = pd.read_csv(f, sep="\t", header=None, names=["id", "paths", "msgs"])
    tsv_dir = tsv_path.rsplit("/", 1)[0]
    records = []
    for _, r in df.iterrows():
        audio_path = ast.literal_eval(r["paths"])[0].replace("/root/data/LibriSpeech", tsv_dir)
        text = ast.literal_eval(r["msgs"])[0]["messages"][-1]["content"]
        records.append({"id": r["id"], "audio_path": audio_path, "text": text})
    return Dataset.from_list(records)


def _trim_batch(batch, output_dir, base_dir, rand_cut_ms):
    """Trim silence for a batch, write audio, return kept rows."""
    import random
    trimmer = SilenceTrimmer()
    out = {"id": [], "audio_file": [], "text": []}
    for i, audio_path in enumerate(batch["audio_path"]):
        try:
            audio, sr = trimmer.read_audio(audio_path)
            hc = random.randint(0, rand_cut_ms) if rand_cut_ms > 0 else 0
            tc = random.randint(0, rand_cut_ms) if rand_cut_ms > 0 else 0
            trimmed = trimmer.trim(audio, sr, head_cut_ms=hc, tail_cut_ms=tc)
            if trimmed is None:
                print(f"[skip] {batch['id'][i]}")
                continue
            rel = audio_path[len(base_dir):].lstrip("/") if base_dir and audio_path.startswith(base_dir) else Path(audio_path).name
            out_path = f"{output_dir.rstrip('/')}/{rel}"
            trimmer.write_audio(out_path, trimmed, sr)
            out["id"].append(batch["id"][i])
            out["audio_file"].append(out_path)
            out["text"].append(batch["text"][i])
            print(f"[done] {batch['id'][i]}  {len(audio)/sr:.1f}s->{len(trimmed)/sr:.1f}s")
        except Exception as e:
            print(f"[error] {batch['id'][i]}: {e}")
    return out


def show_examples(parquet_path: str, n: int = 3):
    """Print first n samples from a parquet file."""
    import json
    print(f"\n--- First {n} samples ---")
    with bf.BlobFile(parquet_path, "rb") as f:
        df = pd.read_parquet(f)
    for _, row in df.head(n).iterrows():
        print(json.dumps(dict(row), indent=2))


def main(tsv_path: str, output_dir: str, jobs: int = 64, n_examples: int = 0, rand_cut_ms: int = 300):
    """Load TSV, trim silence, save as parquet.

    Args:
        tsv_path: Path to TSV file (supports az://).
        output_dir: Where to save trimmed audio and parquet (supports az://).
        jobs: Number of parallel workers.
        n_examples: Number of samples to print after saving (0 to skip).
        rand_cut_ms: Max random head/tail cut in ms after trimming (default 300, 0 to disable).
    """
    SilenceTrimmer._ensure_model()
    print(f"Loading TSV: {tsv_path}")
    ds = load_tsv(tsv_path)
    print(f"  {len(ds)} utterances")

    base_dir = tsv_path.rsplit("/", 1)[0]
    audio_out = f"{output_dir.rstrip('/')}/audio"

    ds = ds.map(
        _trim_batch,
        batched=True,
        batch_size=16,
        num_proc=min(jobs, len(ds)),
        remove_columns=["audio_path"],
        fn_kwargs={"output_dir": audio_out, "base_dir": base_dir, "rand_cut_ms": rand_cut_ms},
    )
    print(f"\n{len(ds)} utterances after trimming")

    parquet_path = f"{output_dir.rstrip('/')}/dataset.parquet"
    with bf.BlobFile(parquet_path, "wb") as f:
        ds.to_parquet(f)
    print(f"Parquet saved: {parquet_path} ({len(ds)} rows)")

    if n_examples > 0:
        show_examples(parquet_path, n=n_examples)


if __name__ == "__main__":
    fire.Fire(main)

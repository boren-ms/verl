"""Align a JSONL dataset using Qwen3-ForcedAligner with datasets.map.

Loads audio from az:// blob paths, runs forced alignment in parallel across
GPUs (one aligner per worker process), and saves word timestamps to parquet.

Usage (on remote 8-GPU node):
    python -m recipe.phimm.align_dataset \
        --jsonl_path az://orngwus2cresco/data/boren/data/openasr_jsonl/librispeech/h100.jsonl \
        --output_path az://orngwus2cresco/data/boren/data/openasr_jsonl/librispeech/alignment.parquet \
        --audio_base az://orngwus2cresco/data/boren/data/openasr_jsonl/librispeech \
        --language English \
        --batch_size 8 \
        --n_gpus 8
"""

import json
import multiprocess
import subprocess
from pathlib import Path

import blobfile as bf
import fire
import torch
import yaml
from datasets import Dataset

multiprocess.set_start_method("spawn", force=True)

# Per-process aligner instance (populated lazily in forked workers)
_aligner = None


def load_ds(source: str) -> Dataset:
    """Load a dataset from JSONL path or YAML config path."""
    if source.endswith((".yaml", ".yml")):
        from recipe.phimm.data.dataset import create_datasets

        with bf.BlobFile(source, "r") as f:
            config = yaml.safe_load(f)
        # Strip training-specific augmentation keys; alignment only needs raw audio_path + text.
        _AUG_KEYS = {
            "add_task_info", "overlap_prefix", "context_prefix", "biasing",
            "simu_preference", "format_preference", "add_rare_keywords",
            "add_tag_keywords", "post_process", "cache_name",
        }
        for k in _AUG_KEYS:
            config.pop(k, None)
        print(f"Loading dataset from config: {source}")
        ds = create_datasets(config)
        if isinstance(ds, dict):
            from datasets import concatenate_datasets
            ds = concatenate_datasets(list(ds.values()))
        print(f"  {len(ds)} utterances, columns: {ds.column_names}")
        return ds
    # Default: JSONL
    print(f"Loading: {source}")
    with bf.BlobFile(source, "r") as f:
        items = [json.loads(line) for line in f if line.strip()]
    ds = Dataset.from_list(items)
    print(f"  {len(ds)} utterances")
    return ds


_KEEP_COLS = {"id", "text", "word_timestamps", "n_words"}
_KEEP_PREFIXES = ("id", "audio_")


def save_ds(ds: Dataset, output_path: str) -> Dataset:
    """Save only required columns to JSONL (default) or parquet."""
    keep = [c for c in ds.column_names
            if c in _KEEP_COLS or c.startswith(_KEEP_PREFIXES)]
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        print(f"Dropping columns: {drop}")
        ds = ds.remove_columns(drop)
    with bf.BlobFile(output_path, "wb") as f:
        if output_path.endswith(".jsonl"):
            ds.to_json(f, orient="records", lines=True)
        else:
            ds.to_parquet(f)
    print(f"Saved: {output_path} ({len(ds)} rows, columns: {ds.column_names})")
    return ds


def show_examples(ds: Dataset, n: int = 3):
    """Print first n samples from an aligned dataset."""
    if n <= 0:
        return
    print(f"\n--- First {n} samples ---")
    for i in range(min(n, len(ds))):
        row = ds[i]
        ts = json.loads(row["word_timestamps"])
        print(f"  {row['id']}: {row['n_words']} words, "
              f"first={ts[0] if ts else 'N/A'}")


_BLOB_MODEL = "az://orngwus2cresco/data/boren/data/verl/models/Qwen3-ForcedAligner-0.6B"


def _ensure_model() -> str:
    """Ensure model weights are available locally. Downloads from blob via bbb sync."""
    cache_dir = Path.home() / ".cache" / "qwen-aligner" / "Qwen3-ForcedAligner-0.6B"
    config_file = cache_dir / "config.json"
    if config_file.exists():
        print(f"Model ready: {cache_dir}")
        return str(cache_dir)
    print(f"Syncing model from {_BLOB_MODEL} ...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["bbb", "sync", _BLOB_MODEL, str(cache_dir)])
    print(f"Model synced: {cache_dir}")
    return str(cache_dir)


def _get_aligner(model_path: str, n_gpus: int, rank: int):
    global _aligner
    if _aligner is None:
        from qwen_asr import Qwen3ForcedAligner

        gpu_id = rank % n_gpus
        _aligner = Qwen3ForcedAligner.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=f"cuda:{gpu_id}",
        )
        print(f"[rank {rank}] Loaded aligner on cuda:{gpu_id}")
    return _aligner


def _align_batch(batch, rank, language, model_path, n_gpus):
    from recipe.phimm.utils.audio import load_audios

    aligner = _get_aligner(model_path, n_gpus, rank)
    audio_inputs = load_audios(batch)

    texts = batch["text"]
    languages = [language] * len(texts)

    try:
        results = aligner.align(audio=audio_inputs, text=texts, language=languages)
    except Exception as e:
        print(f"Batch error: {e}, falling back to one-by-one")
        results = []
        for audio_in, text, lang in zip(audio_inputs, texts, languages):
            try:
                r = aligner.align(audio=audio_in, text=text, language=lang)
                results.append(r[0])
            except Exception:
                results.append([])

    word_timestamps = []
    n_words = []
    for word_list in results:
        ts = [{"text": w.text, "start": w.start_time, "end": w.end_time} for w in word_list]
        word_timestamps.append(json.dumps(ts, ensure_ascii=False))
        n_words.append(len(ts))

    return {"word_timestamps": word_timestamps, "n_words": n_words}


def align_ds(ds: Dataset, language="English", batch_size=8, n_gpus=8, workers_per_gpu=2) -> Dataset:
    """Forced-align a dataset across multiple GPUs using datasets.map."""
    model_path = _ensure_model()
    ds = ds.map(
        _align_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=min(n_gpus * workers_per_gpu, len(ds)),
        with_rank=True,
        fn_kwargs={
            "language": language,
            "model_path": model_path,
            "n_gpus": n_gpus,
        },
    )
    print(f"\n{len(ds)} utterances aligned")
    return ds


_ALIGNED_ROOT = "az://orngwus2cresco/data/boren/data/verl/aligned"


def _default_output(source: str, ext="jsonl") -> str:
    """Derive default output path from source: <ALIGNED_ROOT>/<stem>.jsonl"""
    stem = source.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return f"{_ALIGNED_ROOT}/{stem}.{ext}"


def main(
    input_path: str,
    output_path: str = None,
    language: str = "English",
    batch_size: int = 8,
    n_gpus: int = 8,
    workers_per_gpu: int = 2,
    n_examples: int = 3,
):
    """Align a JSONL dataset and save word timestamps as parquet.

    Args:
        input_path: Path to input JSONL or YAML config (supports az://).
        output_path: Path to output parquet (supports az://).
        language: Language for alignment.
        batch_size: Batch size per GPU.
        n_gpus: Number of GPUs to use.
        workers_per_gpu: Number of worker processes per GPU (overlap I/O with compute).
        n_examples: Number of example rows to print after saving.
    """
    output_path = output_path or _default_output(input_path, ext="parquet")
    ds = load_ds(input_path)
    ds = align_ds(ds, language=language, batch_size=batch_size, n_gpus=n_gpus, workers_per_gpu=workers_per_gpu)
    ds = save_ds(ds, output_path)
    show_examples(ds, n_examples)


if __name__ == "__main__":
    fire.Fire(main)

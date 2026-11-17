"""Prepare ordered SVAD audio_path lists offline without writing audio files.

Each parent keeps its original full reference, using the existing PHIMM
``file#start:end`` selectors and multi-segment training contract.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import tempfile
from functools import lru_cache, partial
from pathlib import Path


@lru_cache(maxsize=1)
def _chunker(max_len_sec, onnx_model_path, num_threads):
    from recipe.phimm.utils.svad.svad import SVadChunker

    return SVadChunker(max_len_sec=max_len_sec, onnx_model_path=onnx_model_path, num_threads=num_threads)


def _validate(max_len_sec, min_seg_sec, num_threads, empty_policy):
    if not math.isfinite(max_len_sec) or max_len_sec < 0.2:
        raise ValueError("max_len_sec must be finite and at least 0.2 seconds")
    if not math.isfinite(min_seg_sec) or not 0 < min_seg_sec <= max_len_sec:
        raise ValueError("min_seg_sec must be positive and no greater than max_len_sec")
    if not isinstance(num_threads, int) or isinstance(num_threads, bool) or num_threads < 1:
        raise ValueError("num_threads must be a positive integer")
    if empty_policy not in ("fixed", "error"):
        raise ValueError("empty_policy must be fixed or error")


def segment_row(row, *, max_len_sec=30.0, min_seg_sec=0.1, num_threads=1, onnx_model_path=None, empty_policy="fixed"):
    """Return one parent with an ordered audio_path list and unchanged text."""
    import numpy as np
    from recipe.phimm.utils.audio import load_raw_audio, sf_read
    from scipy.signal import resample_poly

    _validate(max_len_sec, min_seg_sec, num_threads, empty_policy)
    source = row.get("audio_path") or row.get("audio_file") or row.get("audio_chunk")
    if not isinstance(source, str) or not source:
        raise ValueError("svad_segment requires one non-empty scalar audio source per parent")
    try:
        # Reuse the production reader, including its local/remote and selector
        # support. Explicit paths must take precedence over inline audio.
        audio, sr = load_raw_audio({"audio_path": source})
        audio = np.asarray(audio, dtype=np.float32)
        frames = len(audio)
        if not frames or not np.isfinite(audio).all():
            raise ValueError("Audio must be non-empty and contain only finite samples")
        duration = frames / sr
        if sr != 16000:
            divisor = math.gcd(sr, 16000)
            audio = resample_poly(audio, 16000 // divisor, sr // divisor).astype(np.float32)
        spans = _chunker(max_len_sec, onnx_model_path, num_threads).chunk(audio, 16000)
        maximum = math.floor(max_len_sec * sr)
        minimum = math.ceil(min_seg_sec * sr)
        bounds = []
        previous_end = 0
        for start, end in spans:
            # Remove overlapping context collars: each input sample should be
            # heard at most once when the parent is decoded sequentially.
            left = max(previous_end, min(frames, max(0, round(start * sr))))
            end = min(frames, max(left, round(end * sr)))
            while left < end:
                right = min(left + maximum, end)
                if right - left >= minimum:
                    bounds.append((left / sr, right / sr))
                    previous_end = right
                left = right
        fallback = not bounds
        if fallback:
            if empty_policy == "error":
                raise ValueError("SVAD found no retained speech spans")
            # Never emit an empty audio_path list: ASRDataset requires at least
            # one segment. Keep the parent and make this fallback explicit.
            bounds = [(left / sr, min(left + maximum, frames) / sr) for left in range(0, frames, maximum)]
        segment_source = f"{source}#0%:100%" if ";" in source else source
        result = {
            **row,
            "audio_path": [f"{segment_source}#{start}:{end}" for start, end in bounds],
            "svad_source": source,
            "svad_bounds": bounds,
            "svad_parent_duration": duration,
            "svad_method": "fixed_fallback" if fallback else "svad",
        }
        # The existing list reader must read selectors, not a parent waveform
        # embedded in the same row. Raw text and all other metadata are kept.
        result.pop("audio", None)
        return result
    except Exception as exc:
        raise RuntimeError(f"SVAD preparation failed for {source}: {exc}") from exc
    finally:
        # The production reader caches decoded waveforms for reuse during
        # decoding. Offline preparation visits each long parent once; retaining
        # up to 100 whole recordings per worker would exhaust host memory.
        sf_read.cache_clear()


def svad_segment(ds, *, enabled=True, num_proc=None, **options):
    """Apply SVAD once per parent during the existing offline preparation flow."""
    if not enabled:
        return ds
    _validate(
        options.get("max_len_sec", 30.0),
        options.get("min_seg_sec", 0.1),
        options.get("num_threads", 1),
        options.get("empty_policy", "fixed"),
    )
    return ds.map(
        partial(segment_row, **options),
        num_proc=num_proc,
        desc="Offline SVAD segmentation",
        remove_columns=["audio"] if "audio" in ds.column_names else None,
        load_from_cache_file=False,
    )


def _input_rows(src, audio_dir, audio_key):
    if src:
        manifest = Path(src).expanduser().resolve()
        with manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get(audio_key), str) or not row[audio_key]:
                    raise ValueError(f"{manifest}:{line_number}: expected a scalar {audio_key!r} path")
                source = row[audio_key]
                # The standalone CLI resolves plain relative paths against the
                # manifest. Complex production sources are used verbatim.
                if not any(marker in source for marker in (":", "#", ";", "!")):
                    source = str((manifest.parent / Path(source).expanduser()).resolve())
                yield {**row, "audio_path": source}
    else:
        root = Path(audio_dir).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Audio directory does not exist: {root}")
        extensions = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                yield {"audio_path": str(path)}


def prepare_manifest(*, dst, src=None, audio_dir=None, audio_key="audio_path", n_workers=1, **options):
    """Stream one output row per parent and publish a new JSONL atomically."""
    if bool(src) == bool(audio_dir):
        raise ValueError("Provide exactly one of src or audio_dir")
    if not isinstance(n_workers, int) or isinstance(n_workers, bool) or n_workers < 1:
        raise ValueError("n_workers must be a positive integer")
    _validate(
        options.get("max_len_sec", 30.0),
        options.get("min_seg_sec", 0.1),
        options.get("num_threads", 1),
        options.get("empty_policy", "fixed"),
    )
    destination = Path(dst).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    parents = segments = fallbacks = 0
    temporary = pool = None
    try:
        rows = _input_rows(src, audio_dir, audio_key)
        operation = partial(segment_row, **options)
        if n_workers == 1:
            results = map(operation, rows)
        else:
            pool = mp.get_context("spawn").Pool(n_workers)
            results = pool.imap(operation, rows, chunksize=1)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            for row in results:
                parents += 1
                segments += len(row["audio_path"])
                fallbacks += int(row["svad_method"] == "fixed_fallback")
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                if parents % 100 == 0:
                    print(f"Prepared {parents} parents / {segments} segments", flush=True)
        if not parents:
            raise ValueError("No input audio records found")
        os.link(temporary, destination)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    summary = {"output": str(destination), "parents": parents, "segments": segments, "fallback_parents": fallbacks}
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--src", help="JSONL manifest with one scalar audio path per recording.")
    source.add_argument("--audio-dir", help="Recursively read an audio directory.")
    parser.add_argument("--dst", required=True, help="New output JSONL; existing files are refused.")
    parser.add_argument("--audio-key", default="audio_path", help="Input audio field (output always uses audio_path).")
    parser.add_argument("--max-len-sec", type=float, default=30.0)
    parser.add_argument("--min-seg-sec", type=float, default=0.1)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=1, help="CPU inference threads per worker.")
    parser.add_argument("--onnx-model-path", help="Override the bundled local model.")
    parser.add_argument("--empty-policy", choices=("fixed", "error"), default="fixed")
    prepare_manifest(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()

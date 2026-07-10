# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Build a code-switch / language-mixed ASR dataset (JSONL) from chunk specs.

Given per-language chunk spec files (e.g. ``asr_chunk_cv15_en.json``,
``asr_chunk_cv15_zh.json`` ...), this script randomly draws two or more
utterances from *different* languages and concatenates their audio and
transcription into a single new sample. For every requested mix type
(e.g. ``en_zh``, ``zh_en``, ``fr_it``, ``en_zh_fr``) it produces
``--num-per-type`` samples. Every language inside a single mix type must be
distinct; use ``mix_size`` (2, 3, ...) to control how many languages are
concatenated when auto-generating mix types from a ``languages`` list.

The concatenated audio is written to ``<output-dir>/wavs_<name>/<id>.wav`` (16 kHz
mono) and one JSON object per mixed sample is written to ``<output-dir>/<name>.jsonl``.
The per-config ``wavs_<name>`` folder keeps different configs from overwriting each
other's audio when they share the same ``output_dir``.
Both local and blob (``az://`` / ``https://``) output dirs are supported.

Example
-------
    # Run with the default config
    python -m recipe.phimm.mix_lang_dataset

    # Pick a specific config
    python -m recipe.phimm.mix_lang_dataset --config-name mix_cv15_test

    # Override any field on the CLI (Hydra)
    python -m recipe.phimm.mix_lang_dataset \
        languages=en,zh,fr,it,es pair_mode=permutations \
        num_per_type=100 output_dir=az://.../mixed_lang

Each JSONL row looks like::

    {
      "id": "en_zh_0000",
      "audio_path": "<output-dir>/wavs_<name>/en_zh_0000.wav",
      "text": "<en transcription> <zh transcription>",
      "language": "en_zh",
      "mix_type": "en_zh",
      "duration": 6.42,
      "components": [
        {"language": "en", "text": "...", "audio_chunk": "az://...:400:5", "duration": 3.2},
        {"language": "zh", "text": "...", "audio_chunk": "az://...:400:9", "duration": 3.0}
      ]
    }
"""

from __future__ import annotations

import itertools
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import blobfile as bf
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from recipe.phimm.data.chunk import load_chunk_info, load_examples, load_specs, resolve_path
from recipe.phimm.utils.audio import (
    TARGET_SAMPLE_RATE,
    limit_audio,
    load_raw_audio,
    set_chunk_load_mode,
    sf_write,
)

DEFAULT_TEXT_FIELD = "sft.0.messages.1.content"


def parse_mix_types(mix_types: str) -> list[tuple[str, ...]]:
    combos = []
    for item in mix_types.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split("_")
        if len(parts) < 2 or not all(parts):
            raise ValueError(
                f"Invalid mix type {item!r}; expected format <langA>_<langB>[_<langC>...] (e.g. en_zh or en_zh_fr)."
            )
        if len(set(parts)) != len(parts):
            raise ValueError(f"Invalid mix type {item!r}; all languages in a mix type must be distinct.")
        combos.append(tuple(parts))
    if not combos:
        raise ValueError("No valid mix types provided.")
    return combos


def mix_types_from_languages(languages: str, pair_mode: str, mix_size: int) -> list[tuple[str, ...]]:
    """Auto-generate distinct-language mix combos from a comma-separated language list."""
    langs = [lang.strip() for lang in languages.split(",") if lang.strip()]
    # De-duplicate while preserving order.
    langs = list(dict.fromkeys(langs))
    if mix_size < 2:
        raise ValueError(f"mix_size must be >= 2, got {mix_size}.")
    if len(langs) < mix_size:
        raise ValueError(f"Need at least {mix_size} distinct languages to form mix_size={mix_size} combos, got {langs}.")
    combiner = itertools.permutations if pair_mode == "permutations" else itertools.combinations
    return [tuple(combo) for combo in combiner(langs, mix_size)]


def resolve_mix_types(
    mix_types: str | None, languages: str | None, pair_mode: str, mix_size: int
) -> list[tuple[str, ...]]:
    if languages and mix_types:
        raise ValueError("Set only one of 'languages' or 'mix_types', not both.")
    if languages:
        return mix_types_from_languages(languages, pair_mode, mix_size)
    if mix_types:
        return parse_mix_types(mix_types)
    raise ValueError("Set one of 'mix_types' or 'languages' in the config (both are null by default).")



def language_demand(combos: list[tuple[str, ...]], num_per_type: int) -> dict[str, int]:
    """Total number of samples each language must supply across all mix types."""
    demand: dict[str, int] = {}
    for combo in combos:
        for lang in combo:
            demand[lang] = demand.get(lang, 0) + num_per_type
    return demand


def build_language_pool(lang: str, spec_files, text_field: str, need: int, max_chunks: int | None) -> list[dict]:
    """Return a shuffled list of ``{audio_chunk, text, language}`` records for ``lang``.

    ``spec_files`` is a spec-file path (or list of paths) for this language.
    """
    if isinstance(spec_files, str):
        spec_files = [spec_files]
    specs = load_specs(list(spec_files))
    if not specs:
        raise ValueError(f"No data_sources found in specs for language {lang!r} ({spec_files}).")

    chunks: list[dict] = []
    for spec in specs:
        chunks.extend(load_chunk_info(**spec))
    random.shuffle(chunks)

    pool: list[dict] = []
    fields = ["audios", text_field]
    for n_loaded, chunk in enumerate(tqdm(chunks, desc=f"[{lang}] loading chunks", leave=False), start=1):
        examples = load_examples(chunk, fields)
        if examples:
            audio_chunks = examples.get("audio_chunk", [])
            texts = examples.get(text_field, [])
            for audio_chunk, text in zip(audio_chunks, texts, strict=False):
                if not audio_chunk or not (text and str(text).strip()):
                    continue
                pool.append({"audio_chunk": audio_chunk, "text": str(text).strip(), "language": lang})
        if max_chunks is not None and n_loaded >= max_chunks:
            break
        if len(pool) >= need:
            break

    random.shuffle(pool)
    return pool


def load_component_audio(record: dict, max_dur: float | None) -> tuple[np.ndarray, int]:
    """Load, downmix, resample to 16 kHz and length-limit a single component's audio."""
    data, sr = load_raw_audio(record)
    data, sr = limit_audio(np.asarray(data), sr, max_dur=max_dur)
    assert sr == TARGET_SAMPLE_RATE, f"expected {TARGET_SAMPLE_RATE} Hz after resample, got {sr}"
    return data, sr


def _resolve_output_dir(output_dir: str) -> str:
    """Expand user paths for local dirs; leave blob URIs (``az://``/``https://``) as-is."""
    if "://" in output_dir:
        return output_dir.rstrip("/")
    return os.path.abspath(os.path.expanduser(output_dir)).rstrip("/")


def run_mix(cfg: dict[str, Any]) -> None:
    seed = int(cfg.get("seed", 1234))
    random.seed(seed)
    np.random.seed(seed)
    set_chunk_load_mode(str(cfg.get("chunk_load_mode", "sample")))

    num_per_type = int(cfg.get("num_per_type", 100))
    specs = cfg.get("specs") or {}
    if not isinstance(specs, dict) or not specs:
        raise ValueError("Config must provide a non-empty 'specs' mapping of {language: spec_path(s)}.")
    # text_field may be a single global dotted field or a per-language mapping.
    text_field_cfg = cfg.get("text_field", DEFAULT_TEXT_FIELD)

    def text_field_for(lang: str) -> str:
        if isinstance(text_field_cfg, dict):
            if lang not in text_field_cfg:
                raise ValueError(f"No text_field configured for language {lang!r}; add it under 'text_field'.")
            return str(text_field_cfg[lang])
        return str(text_field_cfg)

    gap_sec_cfg = cfg.get("gap_sec", 0.3)
    if isinstance(gap_sec_cfg, (list, tuple)):
        if len(gap_sec_cfg) != 2:
            raise ValueError(f"gap_sec range must have exactly 2 values [min, max], got {gap_sec_cfg}.")
        gap_lo, gap_hi = float(gap_sec_cfg[0]), float(gap_sec_cfg[1])
    else:
        gap_lo = gap_hi = float(gap_sec_cfg)
    if gap_lo > gap_hi:
        gap_lo, gap_hi = gap_hi, gap_lo
    max_dur_cfg = cfg.get("max_dur")
    max_dur = float(max_dur_cfg) if max_dur_cfg is not None else None
    sep = str(cfg.get("sep", " "))
    allow_reuse = bool(cfg.get("allow_reuse", False))
    max_chunks_per_lang = cfg.get("max_chunks_per_lang")
    max_chunks_per_lang = int(max_chunks_per_lang) if max_chunks_per_lang is not None else None
    num_workers = int(cfg.get("num_workers", 1))
    if num_workers < 1:
        num_workers = os.cpu_count() or 1

    # Size blobfile's urllib3 connection pool to the worker count so parallel
    # reads/writes don't overflow the default (10) and spam "Connection pool is
    # full, discarding connection" warnings.
    bf.configure(connection_pool_max_size=max(num_workers, 10))

    pairs = resolve_mix_types(
        cfg.get("mix_types"),
        cfg.get("languages"),
        str(cfg.get("pair_mode", "permutations")),
        int(cfg.get("mix_size", 2)),
    )
    demand = language_demand(pairs, num_per_type)
    n_lang = len(demand)

    output_dir = _resolve_output_dir(str(cfg.get("output_dir", "~/data/mixed_lang")))
    name = str(cfg.get("name", "mixed")).format(n_lang=n_lang, num_per_type=num_per_type)
    # Use a per-config wav folder so different configs don't overwrite each
    # other's audio when they share the same output_dir.
    wav_dir = bf.join(output_dir, f"wavs_{name}")
    bf.makedirs(wav_dir)
    jsonl_path = bf.join(output_dir, f"{name}.jsonl")

    print(f"Mix types: {pairs}")
    print(f"Per-language sample demand: {demand}")
    print(f"Output dir: {output_dir}")

    # Build one pool per unique language.
    pools: dict[str, list[dict]] = {}
    cursors: dict[str, int] = {}
    for lang, need in demand.items():
        if lang not in specs:
            raise ValueError(f"No spec path configured for language {lang!r}; add it under 'specs'.")
        pool = build_language_pool(lang, specs[lang], text_field_for(lang), need, max_chunks_per_lang)
        if len(pool) < need and not allow_reuse:
            raise RuntimeError(
                f"Language {lang!r} pool has only {len(pool)} usable samples but {need} are required. "
                f"Increase max_chunks_per_lang or set allow_reuse=true."
            )
        print(f"  [{lang}] pool size = {len(pool)} (need {need})")
        pools[lang] = pool
        cursors[lang] = 0

    def draw(lang: str) -> dict:
        pool = pools[lang]
        idx = cursors[lang]
        if idx >= len(pool):
            if not allow_reuse:
                raise RuntimeError(f"Exhausted pool for language {lang!r}.")
            random.shuffle(pool)
            cursors[lang] = 0
            idx = 0
        cursors[lang] = idx + 1
        return pool[idx]

    # Phase 1: draw all sample specs sequentially so RNG (record draws + gap) is
    # deterministic and pool cursors stay consistent regardless of worker count.
    tasks: list[dict] = []
    for combo in pairs:
        mix_type = "_".join(combo)
        for i in range(num_per_type):
            recs = [draw(lang) for lang in combo]
            # One gap between every consecutive pair of components.
            gaps = [random.uniform(gap_lo, gap_hi) for _ in range(len(combo) - 1)]
            tasks.append(
                {
                    "sample_id": f"{mix_type}_{i:04d}",
                    "mix_type": mix_type,
                    "langs": list(combo),
                    "recs": recs,
                    "gaps": gaps,
                }
            )

    def process_task(task: dict) -> dict | None:
        """Load + mix + write one sample's audio; return its JSONL row (or None)."""
        mix_type = task["mix_type"]
        sample_id = task["sample_id"]
        langs = task["langs"]
        recs = task["recs"]
        try:
            audios = [load_component_audio(rec, max_dur)[0].astype(np.float32) for rec in recs]
        except Exception as exc:  # noqa: BLE001 - skip unreadable audio, keep going
            print(f"[WARN] skip {sample_id}: {exc}")
            return None

        gaps = task["gaps"]
        segments: list[np.ndarray] = []
        for idx, audio in enumerate(audios):
            segments.append(audio)
            if idx < len(gaps):
                segments.append(np.zeros(int(gaps[idx] * TARGET_SAMPLE_RATE), dtype=np.float32))
        mixed = np.concatenate(segments)
        wav_path = bf.join(wav_dir, f"{sample_id}.wav")
        sf_write(wav_path, mixed, TARGET_SAMPLE_RATE)

        text = sep.join(rec["text"] for rec in recs)
        return {
            "id": sample_id,
            "audio_path": wav_path,
            "text": text,
            "language": mix_type,
            "mix_type": mix_type,
            "duration": round(len(mixed) / TARGET_SAMPLE_RATE, 3),
            "gaps": [round(g, 3) for g in gaps],
            "components": [
                {
                    "language": lang,
                    "text": rec["text"],
                    "audio_chunk": resolve_path(rec["audio_chunk"]),
                    "duration": round(len(audio) / TARGET_SAMPLE_RATE, 3),
                }
                for lang, rec, audio in zip(langs, recs, audios, strict=True)
            ],
        }

    # Phase 2: process (load/mix/write) in parallel, then persist rows.
    n_written = 0
    print(f"Processing {len(tasks)} samples with {num_workers} worker(s)...")
    with bf.BlobFile(jsonl_path, "w") as out_f:
        if num_workers == 1:
            for task in tqdm(tasks, desc="mixing"):
                row = process_task(task)
                if row is not None:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_written += 1
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_task, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc="mixing"):
                    row = future.result()
                    if row is not None:
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n_written += 1

    print(f"Wrote {n_written} mixed samples to {jsonl_path}")
    print(f"Mixed wavs under {wav_dir}")


@hydra.main(config_path="config/data/mix", config_name="mix_cv15", version_base=None)
def main(config: DictConfig) -> None:
    cfg = OmegaConf.to_container(config, resolve=True)
    run_mix(cfg)


if __name__ == "__main__":
    main()


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Build a code-switch / language-mixed ASR dataset (JSONL) from chunk specs.

Given per-language chunk spec files (e.g. ``asr_chunk_cv15_en.json``,
``asr_chunk_cv15_zh.json`` ...), this script randomly draws two or more
utterances from different adjacent languages and concatenates their audio and
transcription into a single new sample. For every requested mix type
(e.g. ``en_zh``, ``zh_en``, ``fr_it``, ``en_zh_fr``) it produces
``--num-per-type`` samples. A language may recur after an intervening language
(e.g. ``en_zh_en``), but adjacent components must differ. Use ``mix_size``
(2, 3, ...) to control how many languages are concatenated when
auto-generating mix types from a ``languages`` list.

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
        languages=en,zh,fr,it,es \
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
        if any(parts[index] == parts[index + 1] for index in range(len(parts) - 1)):
            raise ValueError(f"Invalid mix type {item!r}; adjacent languages in a mix type must differ.")
        combos.append(tuple(parts))
    if not combos:
        raise ValueError("No valid mix types provided.")
    return combos


def mix_types_from_languages(languages: str, mix_size: int) -> list[tuple[str, ...]]:
    """Auto-generate mixes with distinct adjacent languages from a language list."""
    langs = [lang.strip() for lang in languages.split(",") if lang.strip()]
    # De-duplicate while preserving order.
    langs = list(dict.fromkeys(langs))
    if mix_size < 2:
        raise ValueError(f"mix_size must be >= 2, got {mix_size}.")
    if len(langs) < 2:
        raise ValueError(f"Need at least 2 languages to form ordered mixes, got {langs}.")
    return [
        tuple(combo)
        for combo in itertools.product(langs, repeat=mix_size)
        if all(combo[index] != combo[index + 1] for index in range(len(combo) - 1))
    ]


def resolve_mix_types(
    mix_types: str | None, languages: str | None, mix_size: int
) -> list[tuple[str, ...]]:
    if languages and mix_types:
        raise ValueError("Set only one of 'languages' or 'mix_types', not both.")
    if languages:
        return mix_types_from_languages(languages, mix_size)
    if mix_types:
        return parse_mix_types(mix_types)
    raise ValueError("Set one of 'mix_types' or 'languages' in the config (both are null by default).")



def language_demand(combos: list[tuple[str, ...]], num_per_type: int, draws_per_slot: int = 1) -> dict[str, int]:
    """Total number of samples each language must supply across all mix types.

    When *draws_per_slot* > 1 (target_duration mode), each language slot draws
    multiple records so the demand is scaled accordingly.
    """
    demand: dict[str, int] = {}
    for combo in combos:
        for lang in combo:
            demand[lang] = demand.get(lang, 0) + num_per_type * draws_per_slot
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
    sep = str(cfg.get("sep", " "))
    allow_reuse = bool(cfg.get("allow_reuse", False))

    # --- target_duration: randomize total mixed audio length ---
    target_dur_cfg = cfg.get("target_duration")
    if target_dur_cfg is not None:
        if isinstance(target_dur_cfg, (list, tuple)):
            if len(target_dur_cfg) != 2:
                raise ValueError(f"target_duration range must have 2 values [min, max], got {target_dur_cfg}.")
            target_dur_lo, target_dur_hi = float(target_dur_cfg[0]), float(target_dur_cfg[1])
            if target_dur_lo > target_dur_hi:
                target_dur_lo, target_dur_hi = target_dur_hi, target_dur_lo
        else:
            target_dur_lo = target_dur_hi = float(target_dur_cfg)
    else:
        target_dur_lo = target_dur_hi = 0.0  # sentinel: disabled
    use_target_duration = target_dur_cfg is not None

    max_component_utts = int(cfg.get("max_component_utts", 10))
    max_chunks_per_lang = cfg.get("max_chunks_per_lang")
    max_chunks_per_lang = int(max_chunks_per_lang) if max_chunks_per_lang is not None else None
    num_workers = int(cfg.get("num_workers", 1))
    if num_workers < 1:
        num_workers = os.cpu_count() or 1
    manifest_checkpoint_rows = int(cfg.get("manifest_checkpoint_rows", 100))
    if manifest_checkpoint_rows < 1:
        raise ValueError(
            f"manifest_checkpoint_rows must be >= 1, got {manifest_checkpoint_rows}."
        )

    # Size blobfile's urllib3 connection pool to the worker count so parallel
    # reads/writes don't overflow the default (10) and spam "Connection pool is
    # full, discarding connection" warnings.
    bf.configure(connection_pool_max_size=max(num_workers, 10))

    pairs = resolve_mix_types(
        cfg.get("mix_types"),
        cfg.get("languages"),
        int(cfg.get("mix_size", 2)),
    )
    draws_per_slot = max_component_utts if use_target_duration else 1
    demand = language_demand(pairs, num_per_type, draws_per_slot)
    n_lang = len(demand)

    output_dir = _resolve_output_dir(str(cfg.get("output_dir", "~/data/mixed_lang")))
    name = str(cfg.get("name", "mixed")).format(n_lang=n_lang, num_per_type=num_per_type)
    # Use a per-config wav folder so different configs don't overwrite each
    # other's audio when they share the same output_dir.
    wav_dir = bf.join(output_dir, f"wavs_{name}")
    bf.makedirs(wav_dir)
    jsonl_path = bf.join(output_dir, f"{name}.jsonl")

    completed_ids: set[str] = set()
    if bf.exists(jsonl_path):
        with bf.BlobFile(jsonl_path, "r") as in_f:
            for line_number, line in enumerate(in_f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    sample_id = row["id"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"Cannot resume from malformed row {line_number} in {jsonl_path}: {exc}"
                    ) from exc
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError(f"Cannot resume from row {line_number} in {jsonl_path}: invalid sample id.")
                completed_ids.add(sample_id)
    print(f"Mix types: {pairs}")
    print(f"Per-language sample demand: {demand}")
    print(f"Output dir: {output_dir}")
    if completed_ids:
        print(f"Resuming from {len(completed_ids)} samples already recorded in {jsonl_path}")

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
            # One gap between every consecutive pair of components.
            gaps = [random.uniform(gap_lo, gap_hi) for _ in range(len(combo) - 1)]

            if use_target_duration:
                # Draw a target duration for this sample and distribute randomly.
                sample_target = random.uniform(target_dur_lo, target_dur_hi)
                # Random partition: draw N weights from Dirichlet-like uniform splits.
                weights = [random.random() for _ in combo]
                weight_sum = sum(weights)
                per_lang_targets = [sample_target * w / weight_sum for w in weights]
                # Draw multiple candidate records per language slot.
                recs = [[draw(lang) for _ in range(max_component_utts)] for lang in combo]
            else:
                # Single utterance per language, no duration constraint.
                per_lang_targets = [float("inf")] * len(combo)
                recs = [[draw(lang)] for lang in combo]

            tasks.append(
                {
                    "sample_id": f"{mix_type}_{i:04d}",
                    "mix_type": mix_type,
                    "langs": list(combo),
                    "recs": recs,
                    "gaps": gaps,
                    "per_lang_targets": per_lang_targets,
                }
            )

    def _build_component_audio(
        rec_candidates: list[dict], per_lang_target: float
    ) -> tuple[np.ndarray, list[dict]] | None:
        """Concatenate multiple utterances for one language slot to reach *per_lang_target* seconds.

        Returns (component_audio, used_records_with_durations) or None on failure.
        Each used record gets a 'duration' key added (actual seconds used).
        Gaps between intra-language utterances use the same gap_sec range as inter-language gaps.
        When *per_lang_target* is inf, all candidates are used without a duration constraint.
        """
        parts: list[np.ndarray] = []
        used: list[dict] = []
        accumulated = 0.0

        for rec in rec_candidates:
            try:
                audio, _ = load_component_audio(rec, max_dur=None)
                audio = audio.astype(np.float32)
            except Exception:  # noqa: BLE001
                continue  # skip unreadable, try next candidate

            remaining = per_lang_target - accumulated
            if remaining <= 0:
                break

            # Add intra-language gap before this utterance (not before the first),
            # drawn from the same gap_sec range used for inter-language gaps.
            if parts and gap_hi > 0:
                gap_dur = random.uniform(gap_lo, gap_hi)
                intra_gap_samples = int(gap_dur * TARGET_SAMPLE_RATE)
                if accumulated + gap_dur >= per_lang_target:
                    break
                parts.append(np.zeros(intra_gap_samples, dtype=np.float32))
                accumulated += gap_dur

            audio_dur = len(audio) / TARGET_SAMPLE_RATE
            parts.append(audio)
            accumulated += audio_dur
            used.append({**rec, "duration": round(audio_dur, 3)})

            if accumulated >= per_lang_target:
                break

        if not parts:
            return None
        return np.concatenate(parts), used

    def process_task(task: dict) -> dict | None:
        """Load + mix + write one sample's audio; return its JSONL row (or None)."""
        mix_type = task["mix_type"]
        sample_id = task["sample_id"]
        langs = task["langs"]
        recs = task["recs"]
        per_lang_targets = task["per_lang_targets"]

        component_audios: list[np.ndarray] = []
        component_meta: list[dict] = []
        try:
            for lang, rec_candidates, plt in zip(langs, recs, per_lang_targets, strict=True):
                result = _build_component_audio(rec_candidates, plt)
                if result is None:
                    print(f"[WARN] skip {sample_id}: no usable audio for {lang}")
                    return None
                comp_audio, used_recs = result
                component_audios.append(comp_audio)
                comp_text = sep.join(r["text"] for r in used_recs)
                component_meta.append({
                    "language": lang,
                    "text": comp_text,
                    "duration": round(len(comp_audio) / TARGET_SAMPLE_RATE, 3),
                    "utterances": [
                        {
                            "audio_chunk": resolve_path(r["audio_chunk"]),
                            "text": r["text"],
                            "duration": r["duration"],
                        }
                        for r in used_recs
                    ],
                })
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] skip {sample_id}: {exc}")
            return None

        # Assemble final mixed audio with inter-language gaps.
        gaps = task["gaps"]
        segments: list[np.ndarray] = []
        for idx, audio in enumerate(component_audios):
            segments.append(audio)
            if idx < len(gaps):
                segments.append(np.zeros(int(gaps[idx] * TARGET_SAMPLE_RATE), dtype=np.float32))
        mixed = np.concatenate(segments)
        wav_path = bf.join(wav_dir, f"{sample_id}.wav")
        sf_write(wav_path, mixed, TARGET_SAMPLE_RATE)

        text = sep.join(m["text"] for m in component_meta)
        row: dict[str, Any] = {
            "id": sample_id,
            "audio_path": wav_path,
            "text": text,
            "language": mix_type,
            "mix_type": mix_type,
            "duration": round(len(mixed) / TARGET_SAMPLE_RATE, 3),
            "gaps": [round(g, 3) for g in gaps],
            "components": component_meta,
        }
        if use_target_duration:
            row["target_duration"] = round(sum(per_lang_targets), 3)
        return row

    # Phase 2: process only missing tasks and append their rows to the
    # existing manifest. Phase 1 still draws all specs to preserve the seeded
    # pool cursor assignment for the tasks that remain.
    pending_tasks = [task for task in tasks if task["sample_id"] not in completed_ids]
    n_written = 0
    checkpoint_rows: list[dict] = []

    def write_checkpoint() -> None:
        """Publish accumulated rows by closing the BlobFile append handle."""
        nonlocal n_written
        if not checkpoint_rows:
            return
        with bf.BlobFile(jsonl_path, "a") as out_f:
            for row in checkpoint_rows:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_written += len(checkpoint_rows)
        checkpoint_rows.clear()

    print(f"Processing {len(pending_tasks)} of {len(tasks)} samples with {num_workers} worker(s)...")
    if num_workers == 1:
        for task in tqdm(pending_tasks, desc="mixing"):
            row = process_task(task)
            if row is not None:
                checkpoint_rows.append(row)
                if len(checkpoint_rows) >= manifest_checkpoint_rows:
                    write_checkpoint()
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_task, task) for task in pending_tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="mixing"):
                row = future.result()
                if row is not None:
                    checkpoint_rows.append(row)
                    if len(checkpoint_rows) >= manifest_checkpoint_rows:
                        write_checkpoint()
    write_checkpoint()

    print(f"Wrote {n_written} mixed samples to {jsonl_path} ({len(completed_ids) + n_written} total)")
    print(f"Mixed wavs under {wav_dir}")


@hydra.main(config_path="config/data/mix", config_name="mix_cv15", version_base=None)
def main(config: DictConfig) -> None:
    cfg = OmegaConf.to_container(config, resolve=True)
    run_mix(cfg)


if __name__ == "__main__":
    main()


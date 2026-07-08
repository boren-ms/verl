# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Build a code-switch / language-mixed ASR dataset (JSONL) from chunk specs.

Given per-language chunk spec files (e.g. ``asr_chunk_cv15_en.json``,
``asr_chunk_cv15_zh.json`` ...), this script randomly draws two utterances from
two *different* languages and concatenates their audio and transcription into a
single new sample. For every requested mix type (e.g. ``en_zh``, ``zh_en``,
``fr_it``, ``es_zh``) it produces ``--num-per-type`` samples.

The concatenated audio is written to ``<output-dir>/wavs/<id>.wav`` (16 kHz mono)
and one JSON object per mixed sample is appended to ``<output-dir>/<name>.jsonl``.

Example
-------
    python -m recipe.phimm.mix_lang_dataset \
        --mix-types en_zh,zh_en,fr_it,es_zh \
        --num-per-type 100 \
        --output-dir ~/data/mixed_lang

Each JSONL row looks like::

    {
      "id": "en_zh_0000",
      "audio_path": "/abs/wavs/en_zh_0000.wav",
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

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from recipe.phimm.data.chunk import load_chunk_info, load_examples, load_specs, resolve_path
from recipe.phimm.utils.audio import (
    TARGET_SAMPLE_RATE,
    limit_audio,
    load_raw_audio,
    set_chunk_load_mode,
    sf_write,
)

DEFAULT_SPEC_TEMPLATE = (
    "az://orngwus2cresco/data/speech/users/ruchaofan/DataSpecs/"
    "mlang_asr_data_2605/oss/asr_chunk_cv15_{lang}.json"
)
DEFAULT_TEXT_FIELD = "sft.0.messages.1.content"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mix-types",
        type=str,
        default="en_zh,zh_en,fr_it,es_zh",
        help="Comma separated <langA>_<langB> mix types (order = audio/text order).",
    )
    parser.add_argument("--num-per-type", type=int, default=100, help="Number of mixed samples per mix type.")
    parser.add_argument(
        "--spec-template",
        type=str,
        default=DEFAULT_SPEC_TEMPLATE,
        help="Spec file path with a {lang} placeholder.",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default=DEFAULT_TEXT_FIELD,
        help="Dotted chunk field holding the transcription text.",
    )
    parser.add_argument("--output-dir", type=str, default="~/data/mixed_lang", help="Local output directory.")
    parser.add_argument("--name", type=str, default="mixed", help="Base name for the output jsonl file.")
    parser.add_argument("--gap-sec", type=float, default=0.3, help="Silence (seconds) inserted between the two audios.")
    parser.add_argument("--max-dur", type=float, default=20.0, help="Max duration (seconds) per component audio.")
    parser.add_argument("--sep", type=str, default=" ", help="Separator string between the two transcriptions.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument(
        "--max-chunks-per-lang",
        type=int,
        default=None,
        help="Cap on chunks loaded per language (default: just enough to cover demand).",
    )
    parser.add_argument(
        "--allow-reuse",
        action="store_true",
        help="Allow sampling with replacement when a language pool is too small.",
    )
    parser.add_argument(
        "--chunk-load-mode",
        type=str,
        default="sample",
        choices=("sample", "cached"),
        help="'sample' seeks to each audio (fast for scattered random access); "
        "'cached' loads the whole 400-audio chunk into memory.",
    )
    return parser.parse_args()


def parse_mix_types(mix_types: str) -> list[tuple[str, str]]:
    pairs = []
    for item in mix_types.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split("_")
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"Invalid mix type {item!r}; expected format <langA>_<langB> (e.g. en_zh).")
        pairs.append((parts[0], parts[1]))
    if not pairs:
        raise ValueError("No valid mix types provided.")
    return pairs


def language_demand(pairs: list[tuple[str, str]], num_per_type: int) -> dict[str, int]:
    """Total number of samples each language must supply across all mix types."""
    demand: dict[str, int] = {}
    for lang_a, lang_b in pairs:
        demand[lang_a] = demand.get(lang_a, 0) + num_per_type
        demand[lang_b] = demand.get(lang_b, 0) + num_per_type
    return demand


def build_language_pool(lang: str, spec_template: str, text_field: str, need: int, max_chunks: int | None) -> list[dict]:
    """Return a shuffled list of ``{audio_chunk, text, language}`` records for ``lang``."""
    spec_file = spec_template.format(lang=lang)
    specs = load_specs([spec_file])
    if not specs:
        raise ValueError(f"No data_sources found in spec for language {lang!r} ({spec_file}).")

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
            for audio_chunk, text in zip(audio_chunks, texts):
                if not audio_chunk or not (text and str(text).strip()):
                    continue
                pool.append({"audio_chunk": audio_chunk, "text": str(text).strip(), "language": lang})
        if max_chunks is not None and n_loaded >= max_chunks:
            break
        if len(pool) >= need:
            break

    random.shuffle(pool)
    return pool


def load_component_audio(record: dict, max_dur: float) -> tuple[np.ndarray, int]:
    """Load, downmix, resample and length-limit a single component's audio."""
    data, sr = load_raw_audio(record)
    data, sr = limit_audio(np.asarray(data), sr, max_dur=max_dur)
    return data, sr


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_chunk_load_mode(args.chunk_load_mode)

    pairs = parse_mix_types(args.mix_types)
    demand = language_demand(pairs, args.num_per_type)

    output_dir = Path(args.output_dir).expanduser()
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{args.name}.jsonl"

    print(f"Mix types: {pairs}")
    print(f"Per-language sample demand: {demand}")

    # Build one pool per unique language.
    pools: dict[str, list[dict]] = {}
    cursors: dict[str, int] = {}
    for lang, need in demand.items():
        max_chunks = args.max_chunks_per_lang
        pool = build_language_pool(lang, args.spec_template, args.text_field, need, max_chunks)
        if len(pool) < need and not args.allow_reuse:
            raise RuntimeError(
                f"Language {lang!r} pool has only {len(pool)} usable samples but {need} are required. "
                f"Increase --max-chunks-per-lang or pass --allow-reuse."
            )
        print(f"  [{lang}] pool size = {len(pool)} (need {need})")
        pools[lang] = pool
        cursors[lang] = 0

    def draw(lang: str) -> dict:
        pool = pools[lang]
        idx = cursors[lang]
        if idx >= len(pool):
            if not args.allow_reuse:
                raise RuntimeError(f"Exhausted pool for language {lang!r}.")
            random.shuffle(pool)
            cursors[lang] = 0
            idx = 0
        cursors[lang] = idx + 1
        return pool[idx]

    gap = np.zeros(int(args.gap_sec * TARGET_SAMPLE_RATE), dtype=np.float32)
    n_written = 0
    with open(jsonl_path, "w", encoding="utf-8") as out_f:
        for lang_a, lang_b in pairs:
            mix_type = f"{lang_a}_{lang_b}"
            for i in tqdm(range(args.num_per_type), desc=f"mixing {mix_type}"):
                rec_a = draw(lang_a)
                rec_b = draw(lang_b)
                try:
                    audio_a, _ = load_component_audio(rec_a, args.max_dur)
                    audio_b, _ = load_component_audio(rec_b, args.max_dur)
                except Exception as exc:  # noqa: BLE001 - skip unreadable audio, keep going
                    print(f"[WARN] skip {mix_type} #{i}: {exc}")
                    continue

                mixed = np.concatenate([audio_a.astype(np.float32), gap, audio_b.astype(np.float32)])
                sample_id = f"{mix_type}_{i:04d}"
                wav_path = wav_dir / f"{sample_id}.wav"
                sf_write(str(wav_path), mixed, TARGET_SAMPLE_RATE)

                text = f"{rec_a['text']}{args.sep}{rec_b['text']}"
                row = {
                    "id": sample_id,
                    "audio_path": str(wav_path),
                    "text": text,
                    "language": mix_type,
                    "mix_type": mix_type,
                    "duration": round(len(mixed) / TARGET_SAMPLE_RATE, 3),
                    "components": [
                        {
                            "language": lang_a,
                            "text": rec_a["text"],
                            "audio_chunk": resolve_path(rec_a["audio_chunk"]),
                            "duration": round(len(audio_a) / TARGET_SAMPLE_RATE, 3),
                        },
                        {
                            "language": lang_b,
                            "text": rec_b["text"],
                            "audio_chunk": resolve_path(rec_b["audio_chunk"]),
                            "duration": round(len(audio_b) / TARGET_SAMPLE_RATE, 3),
                        },
                    ],
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"Wrote {n_written} mixed samples to {jsonl_path}")
    print(f"Mixed wavs under {wav_dir}")


if __name__ == "__main__":
    main()

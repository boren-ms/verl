#!/usr/bin/env python3
"""Randomly augment speech audio with noise injection and speed perturbation.

Each run writes augmented WAVs under OUTPUT_DIR/audio/ and example metadata to
OUTPUT_DIR/data.jsonl. The JSONL uses audio_path to point at each generated WAV.

Examples:
    # Augment local WAVs with noise sampled from another local folder.
    python scripts/augment_audio.py \
        --input_dir /path/to/librispeech_wavs \
        --noise_dir /path/to/noise_wavs \
        --output_dir /tmp/augmented_librispeech \
        --max_examples 20

    # Pull example utterances from OpenASR LibriSpeech and augment them.
    python scripts/augment_audio.py \
        --openasr_librispeech \
        --openasr_split test.clean \
        --noise_dir /path/to/noise_wavs \
        --output_dir /tmp/augmented_openasr_librispeech \
        --max_examples 20

    # Augment paths listed in a JSONL manifest with an audio_path/audio_file/audio field.
    python scripts/augment_audio.py \
        --input_manifest az://orngwus2cresco/data/boren/data/openasr_jsonl/librispeech/h100.jsonl \
        --noise_dir /path/to/noise_wavs \
        --output_dir /tmp/augmented_manifest \
        --max_examples 20

    # Speed-only augmentation, no noise folder needed.
    python scripts/augment_audio.py \
        --openasr_librispeech \
        --output_dir /tmp/speed_only_librispeech \
        --noise_prob 0 \
        --speed_min 1 \
        --speed_max 2 \
        --speed_prob 1 \
        --max_examples 20
"""

import argparse
import io
import json
import math
import os
import random
import sys
from pathlib import Path

import blobfile as bf
import numpy as np
import soundfile as sf


sys.path.insert(0, str(Path(__file__).parents[1]))


AUDIO_EXTENSIONS = (".wav", ".flac")


def is_blob_path(path: str) -> bool:
    return "://" in path


def make_dirs(path: str) -> None:
    if is_blob_path(path):
        if not bf.exists(path):
            bf.makedirs(path)
    else:
        Path(path).mkdir(parents=True, exist_ok=True)


def read_audio(path: str) -> tuple[np.ndarray, int]:
    with bf.BlobFile(path, "rb") as audio_file:
        audio, sample_rate = sf.read(audio_file, always_2d=False)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def write_audio(path: str, audio: np.ndarray, sample_rate: int) -> None:
    parent = path.rsplit("/", 1)[0]
    make_dirs(parent)
    with bf.BlobFile(path, "wb") as audio_file:
        sf.write(audio_file, audio, sample_rate, format="WAV")


def collect_audio_paths(audio_dir: str, recursive: bool = True) -> list[str]:
    if is_blob_path(audio_dir):
        patterns = [f"{audio_dir.rstrip('/')}/**/*{ext}" if recursive else f"{audio_dir.rstrip('/')}/*{ext}" for ext in AUDIO_EXTENSIONS]
        paths = []
        for pattern in patterns:
            paths.extend(bf.glob(pattern))
        return sorted(paths)

    root = Path(audio_dir).expanduser()
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(str(path) for path in iterator if path.suffix.lower() in AUDIO_EXTENSIONS)


def read_manifest_audio_paths(manifest_path: str, audio_base: str | None = None) -> list[dict]:
    records = []
    with bf.BlobFile(manifest_path, "r") as manifest_file:
        for line_idx, line in enumerate(manifest_file):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            audio_path = row.get("audio_path") or row.get("audio_file") or row.get("audio")
            if not audio_path:
                continue
            if audio_base and not is_blob_path(audio_path) and not os.path.isabs(audio_path):
                audio_path = f"{audio_base.rstrip('/')}/{audio_path}"
            records.append({"id": row.get("id", line_idx), "audio_path": audio_path, "text": row.get("text")})
    return records


def linear_resample(audio: np.ndarray, output_len: int) -> np.ndarray:
    if output_len <= 0:
        raise ValueError(f"output_len must be positive, got {output_len}")
    if len(audio) == output_len:
        return audio.copy()
    if len(audio) == 0:
        return audio.copy()

    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=output_len, endpoint=True)
    if audio.ndim == 1:
        return np.interp(x_new, x_old, audio).astype(np.float32)

    channels = [np.interp(x_new, x_old, audio[:, channel]) for channel in range(audio.shape[1])]
    return np.stack(channels, axis=1).astype(np.float32)


def speed_perturb(audio: np.ndarray, speed_factor: float) -> np.ndarray:
    """Change playback speed by resampling the waveform length.

    speed_factor > 1.0 makes the utterance shorter/faster. speed_factor < 1.0
    makes it longer/slower. This intentionally avoids optional DSP dependencies.
    """
    if speed_factor <= 0:
        raise ValueError(f"speed_factor must be positive, got {speed_factor}")
    output_len = max(1, int(round(len(audio) / speed_factor)))
    return linear_resample(audio, output_len)


def resample_to_sample_rate(audio: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if source_sample_rate == target_sample_rate:
        return audio
    output_len = max(1, int(round(len(audio) * target_sample_rate / source_sample_rate)))
    return linear_resample(audio, output_len)


def match_channels(noise: np.ndarray, target_ndim: int, target_channels: int | None = None) -> np.ndarray:
    if target_ndim == 1:
        return noise.mean(axis=1) if noise.ndim == 2 else noise
    if noise.ndim == 1:
        return np.repeat(noise[:, None], target_channels or 1, axis=1)
    if target_channels is not None and noise.shape[1] != target_channels:
        if noise.shape[1] > target_channels:
            return noise[:, :target_channels]
        repeats = math.ceil(target_channels / noise.shape[1])
        return np.tile(noise, (1, repeats))[:, :target_channels]
    return noise


def crop_or_tile(noise: np.ndarray, target_len: int, rng: random.Random) -> np.ndarray:
    if len(noise) == 0:
        raise ValueError("noise signal is empty")
    if len(noise) >= target_len:
        start = rng.randint(0, len(noise) - target_len)
        return noise[start : start + target_len]
    repeats = math.ceil(target_len / len(noise))
    if noise.ndim == 1:
        tiled = np.tile(noise, repeats)
    else:
        tiled = np.tile(noise, (repeats, 1))
    return tiled[:target_len]


def mix_noise_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float, rng: random.Random) -> np.ndarray:
    """Inject noise into clean audio at the requested SNR in dB."""
    noise = match_channels(noise, clean.ndim, clean.shape[1] if clean.ndim == 2 else None)
    noise = crop_or_tile(noise, len(clean), rng)

    clean_power = float(np.mean(np.square(clean)))
    noise_power = float(np.mean(np.square(noise)))
    if clean_power <= 0.0 or noise_power <= 0.0:
        return clean.copy()

    noise_scale = math.sqrt(clean_power / (noise_power * (10.0 ** (snr_db / 10.0))))
    return clean + noise * noise_scale


def peak_normalize(audio: np.ndarray, peak: float = 0.99) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if max_abs > peak:
        audio = audio * (peak / max_abs)
    return audio.astype(np.float32)


def augment_audio(
    audio: np.ndarray,
    sample_rate: int,
    noise_paths: list[str],
    rng: random.Random,
    snr_range: tuple[float, float] = (10.0, 12.0),
    speed_range: tuple[float, float] = (0.9, 1.1),
    noise_prob: float = 1.0,
    speed_prob: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Apply random speed perturbation and optional random noise injection."""
    metadata = {
        "snr_db": None,
        "noise_path": None,
        "speed_factor": 1.0,
    }

    augmented = np.asarray(audio, dtype=np.float32)
    if rng.random() < speed_prob:
        speed_factor = rng.uniform(*speed_range)
        augmented = speed_perturb(augmented, speed_factor)
        metadata["speed_factor"] = speed_factor

    if noise_paths and rng.random() < noise_prob:
        noise_path = rng.choice(noise_paths)
        noise, noise_sample_rate = read_audio(noise_path)
        noise = resample_to_sample_rate(noise, noise_sample_rate, sample_rate)
        snr_db = rng.uniform(*snr_range)
        augmented = mix_noise_at_snr(augmented, noise, snr_db, rng)
        metadata.update({"snr_db": snr_db, "noise_path": noise_path})

    return peak_normalize(augmented), metadata


def openasr_librispeech_records(split: str, max_examples: int | None, seed: int) -> list[dict]:
    from datasets import Audio
    from datasets import load_dataset

    dataset = load_dataset("hf-audio/esb-datasets-test-only-sorted", "librispeech", split=split)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    if max_examples is not None and max_examples < len(dataset):
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
        dataset = dataset.select(indices[:max_examples])

    records = []
    for idx, row in enumerate(dataset):
        audio = row["audio"]
        if audio.get("bytes") is not None:
            audio_array, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        else:
            audio_array, sample_rate = read_audio(audio["path"])
        records.append(
            {
                "id": row.get("id", f"openasr_librispeech_{idx}"),
                "audio": np.asarray(audio_array, dtype=np.float32),
                "sample_rate": int(sample_rate),
                "text": row.get("text"),
            }
        )
    return records


def build_input_records(args: argparse.Namespace) -> list[dict]:
    if args.openasr_librispeech:
        return openasr_librispeech_records(args.openasr_split, args.max_examples, args.seed)

    if args.input_manifest:
        records = read_manifest_audio_paths(args.input_manifest, args.manifest_audio_base)
    elif args.input_dir:
        records = [{"id": Path(path).stem, "audio_path": path, "text": None} for path in collect_audio_paths(args.input_dir)]
    else:
        raise ValueError("Provide one of --input_dir, --input_manifest, or --openasr_librispeech")

    rng = random.Random(args.seed)
    rng.shuffle(records)
    if args.max_examples is not None:
        records = records[: args.max_examples]
    return records


def output_name(record_id: str | int, index: int) -> str:
    stem = str(record_id).replace("/", "_").replace(" ", "_") or f"sample_{index}"
    return f"{index:06d}_{stem}.wav"


def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    make_dirs(args.output_dir)
    audio_output_dir = f"{args.output_dir.rstrip('/')}/audio"
    make_dirs(audio_output_dir)

    noise_paths = collect_audio_paths(args.noise_dir) if args.noise_dir else []
    if args.noise_prob > 0.0 and not noise_paths:
        raise ValueError("--noise_dir must contain at least one .wav/.flac file when --noise_prob > 0")

    records = build_input_records(args)
    if not records:
        raise ValueError("No input audio records found")

    data_jsonl_path = f"{args.output_dir.rstrip('/')}/data.jsonl"
    with bf.BlobFile(data_jsonl_path, "w") as manifest_file:
        for index, record in enumerate(records):
            if "audio" in record:
                audio = np.asarray(record["audio"], dtype=np.float32)
                sample_rate = int(record["sample_rate"])
                source_audio_path = None
            else:
                source_audio_path = record["audio_path"]
                audio, sample_rate = read_audio(source_audio_path)

            augmented, metadata = augment_audio(
                audio=audio,
                sample_rate=sample_rate,
                noise_paths=noise_paths,
                rng=rng,
                snr_range=(args.snr_min, args.snr_max),
                speed_range=(args.speed_min, args.speed_max),
                noise_prob=args.noise_prob,
                speed_prob=args.speed_prob,
            )
            out_path = f"{audio_output_dir.rstrip('/')}/{output_name(record['id'], index)}"
            write_audio(out_path, augmented, sample_rate)

            out_record = {
                "id": record["id"],
                "source_audio_path": source_audio_path,
                "audio_path": out_path,
                "sample_rate": sample_rate,
                "duration": len(augmented) / sample_rate,
                "text": record.get("text"),
                **metadata,
            }
            manifest_file.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            print(
                f"[{index + 1}/{len(records)}] wrote {out_path} "
                f"speed={metadata['speed_factor']:.3f} snr={metadata['snr_db']}"
            )

    print(f"Data JSONL: {data_jsonl_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment speech audio with random noise and speed perturbation.")
    source = parser.add_argument_group("input source")
    source.add_argument("--input_dir", type=str, help="Directory containing input .wav/.flac files")
    source.add_argument("--input_manifest", type=str, help="JSONL manifest with audio_path/audio_file/audio field")
    source.add_argument("--manifest_audio_base", type=str, help="Base directory for relative manifest audio paths")
    source.add_argument("--openasr_librispeech", action="store_true", help="Use example audio from OpenASR LibriSpeech")
    source.add_argument("--openasr_split", type=str, default="test.clean", help="OpenASR LibriSpeech split")

    parser.add_argument("--noise_dir", type=str, help="Directory containing noise .wav/.flac files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for augmented WAVs and manifest")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of utterances to augment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    parser.add_argument("--snr_min", type=float, default=10.0, help="Minimum SNR in dB")
    parser.add_argument("--snr_max", type=float, default=12.0, help="Maximum SNR in dB")
    parser.add_argument("--noise_prob", type=float, default=1.0, help="Probability of adding noise to an utterance")
    parser.add_argument("--speed_min", type=float, default=0.9, help="Minimum speed factor")
    parser.add_argument("--speed_max", type=float, default=1.1, help="Maximum speed factor")
    parser.add_argument("--speed_prob", type=float, default=1.0, help="Probability of speed perturbing an utterance")
    args = parser.parse_args()

    if args.snr_min > args.snr_max:
        parser.error("--snr_min must be <= --snr_max")
    if args.speed_min <= 0 or args.speed_max <= 0 or args.speed_min > args.speed_max:
        parser.error("speed range must be positive and --speed_min must be <= --speed_max")
    if not 0.0 <= args.noise_prob <= 1.0:
        parser.error("--noise_prob must be between 0 and 1")
    if args.noise_prob > 0.0 and not args.noise_dir:
        parser.error("--noise_dir is required when --noise_prob > 0")
    if not 0.0 <= args.speed_prob <= 1.0:
        parser.error("--speed_prob must be between 0 and 1")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

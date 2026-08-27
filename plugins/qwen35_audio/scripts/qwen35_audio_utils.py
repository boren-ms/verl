"""Shared input staging and audio loading helpers for Qwen3.5-Audio scripts."""

import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

TARGET_SAMPLE_RATE = 16_000
SUPPORTED_AUDIO_SUFFIXES = {".flac", ".mp3", ".ogg", ".wav"}


def _env_default(names: tuple[str, ...], fallback: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return fallback


def add_input_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_model_path: str,
    default_audio_path: str,
    default_cache_root: str,
    model_env_names: tuple[str, ...] = ("QWEN35_AUDIO_MODEL",),
    audio_env_names: tuple[str, ...] = ("QWEN35_AUDIO_SAMPLE",),
) -> None:
    parser.set_defaults(default_audio_path=_env_default(audio_env_names, default_audio_path))
    parser.add_argument(
        "--model",
        default=_env_default(model_env_names, default_model_path),
        help="Path to the converted Qwen3.5-Audio HuggingFace checkpoint.",
    )
    parser.add_argument(
        "--audio",
        action="append",
        default=None,
        help="Path to an audio file readable by soundfile. May be passed multiple times.",
    )
    parser.add_argument(
        "--audio-folder",
        action="append",
        default=None,
        help="Local folder to scan recursively for supported audio files. May be passed multiple times.",
    )
    parser.add_argument(
        "--local-cache-root",
        default=os.getenv("QWEN35_AUDIO_CACHE_ROOT", default_cache_root),
        help="Local directory used to stage az:// model and audio inputs.",
    )
    parser.add_argument(
        "--skip-stage",
        action="store_true",
        help="Do not stage az:// inputs; pass paths through directly.",
    )


def is_az_path(path: str) -> bool:
    return path.startswith("az://")


def run_bbb(*args: str) -> None:
    if shutil.which("bbb") is None:
        raise RuntimeError("bbb is required to stage az:// paths on a fresh node")
    print("bbb " + " ".join(args))
    subprocess.run(["bbb", *args], check=True)


def stage_input(source: str, destination: Path, *, is_dir: bool) -> str:
    if not is_az_path(source):
        return source

    if destination.exists():
        print(f"using_staged_path={destination}")
        return str(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if is_dir:
        destination.mkdir(parents=True, exist_ok=True)
        run_bbb("sync", source, str(destination))
    else:
        run_bbb("cp", source, str(destination))
    return str(destination)


def cache_dir_name(source: str) -> str:
    source_key = source.rstrip("/")
    source_name = Path(source_key).name or "model"
    digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:10]
    return f"{source_name}-{digest}"


def cache_file_name(source: str) -> str:
    source_path = Path(source.rstrip("/"))
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    return f"{source_path.stem}-{digest}{source_path.suffix}"


def resolve_audio_sources(
    args: argparse.Namespace,
    default_audio_path: str | None = None,
) -> list[str]:
    sources = list(args.audio or [])
    audio_folders = args.audio_folder or []
    for folder_value in audio_folders:
        if is_az_path(folder_value):
            raise ValueError("--audio-folder requires a local path; stage az:// folders first")
        folder = Path(folder_value).expanduser()
        if not folder.is_dir():
            raise ValueError(f"audio folder does not exist or is not a directory: {folder}")
        sources.extend(
            str(path)
            for path in sorted(folder.rglob("*"))
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
        )

    if not sources and audio_folders:
        raise ValueError("no supported audio files found in the requested folders")
    if not sources:
        sources.append(default_audio_path or args.default_audio_path)

    return list(dict.fromkeys(sources))


def stage_inputs(
    args: argparse.Namespace,
    default_audio_path: str | None = None,
) -> tuple[str, list[tuple[str, str]]]:
    audio_sources = resolve_audio_sources(args, default_audio_path)
    if args.skip_stage:
        return args.model, [(source, source) for source in audio_sources]

    cache_root = Path(args.local_cache_root)
    model_path = stage_input(
        args.model,
        cache_root / "models" / cache_dir_name(args.model),
        is_dir=True,
    )
    audio_paths = [
        (
            source,
            stage_input(
                source,
                cache_root / "audio" / cache_file_name(source),
                is_dir=False,
            ),
        )
        for source in audio_sources
    ]
    return model_path, audio_paths


def load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(audio_path)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        import torchaudio.functional as F

        waveform = F.resample(
            torch.from_numpy(waveform),
            sample_rate,
            TARGET_SAMPLE_RATE,
        ).numpy()
        sample_rate = TARGET_SAMPLE_RATE
    return waveform.astype(np.float32), sample_rate
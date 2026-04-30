"""Audio augmentation helpers for phimm datasets."""

import json
import math
import random
import re

import blobfile as bf
import numpy as np

from recipe.phimm.utils.audio import sf_read


def load_audio_paths_from_jsonl(jsonl_path, audio_key="audio_path"):
    audio_paths = []
    with bf.BlobFile(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            audio_path = row.get(audio_key) or row.get("audio_file") or row.get("audio")
            if not audio_path:
                continue
            audio_paths.append(audio_path)
    if not audio_paths:
        raise ValueError(f"No noise audio paths found in {jsonl_path} with key {audio_key!r}.")
    return audio_paths


def load_audio_paths_from_dir(noise_dir):
    audio_paths = sorted(bf.glob(f"{str(noise_dir).rstrip('/')}/*.wav"))
    if not audio_paths:
        raise FileNotFoundError(f"No .wav noise files found in {noise_dir}")
    return audio_paths


def load_noise_paths(noise_path):
    if noise_path is None:
        return []
    noise_path = str(noise_path)
    if noise_path.endswith(".jsonl"):
        return load_audio_paths_from_jsonl(noise_path)
    if bf.isdir(noise_path):
        return load_audio_paths_from_dir(noise_path)
    if noise_path.endswith(".wav") and bf.exists(noise_path):
        return [noise_path]
    if bf.exists(noise_path):
        return load_audio_paths_from_jsonl(noise_path)
    raise FileNotFoundError(f"noise_path must be a JSONL file or directory of WAV files: {noise_path}")


def to_float32_audio(audio):
    return np.asarray(audio, dtype=np.float32)


def as_range(value, default):
    if value is None:
        value = default
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value, value]
    if len(values) == 1:
        values = [values[0], values[0]]
    return [float(values[0]), float(values[-1])]


def linear_resample(audio, output_len):
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


def speed_perturb_audio(audio, speed_factor):
    if speed_factor <= 0:
        raise ValueError(f"speed_factor must be positive, got {speed_factor}")
    output_len = max(1, int(round(len(audio) / speed_factor)))
    return linear_resample(audio, output_len)


def resample_audio_to_sr(audio, source_sr, target_sr):
    if source_sr == target_sr:
        return audio
    output_len = max(1, int(round(len(audio) * target_sr / source_sr)))
    return linear_resample(audio, output_len)


def match_audio_channels(noise, audio):
    if audio.ndim == 1:
        return noise.mean(axis=1) if noise.ndim == 2 else noise
    target_channels = audio.shape[1]
    if noise.ndim == 1:
        return np.repeat(noise[:, None], target_channels, axis=1)
    if noise.shape[1] > target_channels:
        return noise[:, :target_channels]
    if noise.shape[1] < target_channels:
        repeats = math.ceil(target_channels / noise.shape[1])
        return np.tile(noise, (1, repeats))[:, :target_channels]
    return noise


def crop_or_tile_audio(noise, target_len, random_state):
    if len(noise) == 0:
        raise ValueError("noise signal is empty")
    if len(noise) >= target_len:
        start = random_state.randint(0, len(noise) - target_len)
        return noise[start : start + target_len]
    repeats = math.ceil(target_len / len(noise))
    tiled = np.tile(noise, repeats) if noise.ndim == 1 else np.tile(noise, (repeats, 1))
    return tiled[:target_len]


def mix_noise_at_snr(audio, noise, snr_db, random_state):
    noise = match_audio_channels(noise, audio)
    noise = crop_or_tile_audio(noise, len(audio), random_state)
    audio_power = float(np.mean(np.square(audio)))
    noise_power = float(np.mean(np.square(noise)))
    if audio_power <= 0.0 or noise_power <= 0.0:
        return audio.copy()
    noise_scale = math.sqrt(audio_power / (noise_power * (10.0 ** (snr_db / 10.0))))
    return audio + noise * noise_scale


def peak_normalize_audio(audio, peak=0.99):
    max_abs = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if max_abs > peak:
        audio = audio * (peak / max_abs)
    return audio.astype(np.float32)


def safe_audio_stem(value, fallback):
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or fallback)).strip("_")
    return stem or str(fallback)


class AudioAugmenter:
    def __init__(
        self,
        speed_prob=1.0,
        speed_range=None,
        noise_prob=1.0,
        snr_range=None,
        noise_path=None,
        peak=0.99,
        seed=0,
        audio_reader=sf_read,
    ):
        self.speed_prob = float(speed_prob)
        self.speed_range = as_range(speed_range, [0.9, 1.1])
        self.noise_prob = float(noise_prob)
        self.snr_range = as_range(snr_range, [10.0, 12.0])
        self.peak = float(peak)
        self.seed = int(seed)
        self.audio_reader = audio_reader
        self.noise_paths = load_noise_paths(noise_path)

        if noise_path is not None:
            print(f"Loaded {len(self.noise_paths)} noise audio paths from {noise_path}")

        if self.noise_prob > 0 and not self.noise_paths:
            raise ValueError("noise_path must be set when noise_prob > 0")

    def _random_state(self, example_index=None):
        if example_index is None:
            return random.Random(self.seed)
        return random.Random(self.seed + int(example_index))

    def _speed_perturb_audio(self, audio, random_state):
        if random_state.random() >= self.speed_prob:
            return audio, 1.0
        speed_factor = random_state.uniform(self.speed_range[0], self.speed_range[-1])
        return speed_perturb_audio(audio, speed_factor), speed_factor

    def speed_perturb_audio(self, audio, example_index=None):
        return self._speed_perturb_audio(to_float32_audio(audio), self._random_state(example_index))

    def _random_add_noise(self, audio, sr, random_state):
        if not self.noise_paths or random_state.random() >= self.noise_prob:
            return audio, None, None
        noise_path = random_state.choice(self.noise_paths)
        noise, noise_sr = self.audio_reader(noise_path)
        noise = to_float32_audio(noise)
        noise = resample_audio_to_sr(noise, noise_sr, sr)
        snr_db = random_state.uniform(self.snr_range[0], self.snr_range[-1])
        return mix_noise_at_snr(audio, noise, snr_db, random_state), snr_db, noise_path

    def random_add_noise(self, audio, sr, example_index=None):
        return self._random_add_noise(to_float32_audio(audio), sr, self._random_state(example_index))

    def augment(self, audio, sr, example_index=None):
        random_state = self._random_state(example_index)
        augmented = to_float32_audio(audio)
        augmented, speed_factor = self._speed_perturb_audio(augmented, random_state)
        augmented, snr_db, noise_path = self._random_add_noise(augmented, sr, random_state)
        return peak_normalize_audio(augmented, peak=self.peak), {
            "speed_factor": speed_factor,
            "snr_db": snr_db,
            "noise_path": noise_path,
        }

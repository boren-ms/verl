#!/usr/bin/env python3
"""Save a SpecAugment-style STFT augmentation as a WAV (see specaugment_wav.md)."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.interpolate import interp1d


@dataclass(frozen=True)
class SpecAugmentConfig:
    window_ms: float = 25.0
    hop_ms: float = 10.0
    time_warp_param: int = 80
    freq_mask_param: int = 27
    time_mask_param: int = 100
    num_freq_masks: int = 2
    num_time_masks: int = 2
    time_mask_max_ratio: float = 0.2

    def __post_init__(self):
        if not (np.isfinite(self.window_ms) and np.isfinite(self.hop_ms)):
            raise ValueError("Window and hop must be finite.")
        if not 0 < self.hop_ms < self.window_ms:
            raise ValueError("Require 0 < hop_ms < window_ms.")
        for name in ("time_warp_param", "freq_mask_param", "time_mask_param", "num_freq_masks", "num_time_masks"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if not 0 <= self.time_mask_max_ratio <= 1:
            raise ValueError("time_mask_max_ratio must be between 0 and 1.")


def augment_magnitude(magnitude, config, rng, *, return_metadata=False):
    """Warp/mask (..., frequency, time), sharing one policy across channels.

    Widths include zero, as in SpecAugment. Masked linear magnitudes are zero.
    Time warping uses a monotonic, piecewise-linear map with fixed endpoints.
    """
    result = np.array(magnitude, copy=True)
    n_freq, n_time = result.shape[-2:]
    metadata = {"frequency_bins": n_freq, "time_frames": n_time, "warp": None, "masks": []}
    warp = config.time_warp_param
    if warp and n_time > 2 * warp + 1:
        anchor = int(rng.integers(warp, n_time - warp))
        destination = anchor + int(rng.integers(-warp, warp + 1))
        destination = np.clip(destination, 1, n_time - 2)
        coordinates = np.interp(np.arange(n_time), [0, destination, n_time - 1], [0, anchor, n_time - 1])
        result = interp1d(np.arange(n_time), result, axis=-1, assume_sorted=True)(coordinates)
        metadata["warp"] = {"anchor_frame": anchor, "destination_frame": int(destination)}

    limits = (
        (-2, min(config.freq_mask_param, n_freq), config.num_freq_masks),
        (-1, min(config.time_mask_param, int(config.time_mask_max_ratio * n_time)), config.num_time_masks),
    )
    for axis, max_width, count in limits:
        if not max_width:
            continue
        for _ in range(count):
            width = int(rng.integers(0, max_width + 1))
            start = int(rng.integers(0, result.shape[axis] - width + 1))
            selection = [slice(None)] * result.ndim
            selection[axis] = slice(start, start + width)
            result[tuple(selection)] = 0
            metadata["masks"].append({"axis": "frequency" if axis == -2 else "time", "start": start, "width": width})
    return (result, metadata) if return_metadata else result


def augment_waveform(audio, sample_rate, config=None, *, seed=None, return_metadata=False):
    """Augment (samples,) or (samples, channels), preserving shape and sample rate.

    This WAV adaptation operates on linear STFT magnitudes, retaining original
    phase for inverse STFT. It is not the paper's log-mel training transform.
    """
    config = config or SpecAugmentConfig()
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim not in (1, 2) or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Audio must be a nonempty, finite mono or multichannel waveform.")
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer.")
    window = round(sample_rate * config.window_ms / 1000)
    hop = round(sample_rate * config.hop_ms / 1000)
    if not 0 < hop < window:
        raise ValueError("Window/hop round to invalid sample counts at this sample rate.")
    waveform = audio[None, :] if audio.ndim == 1 else audio.T
    waveform = np.pad(waveform, ((0, 0), (0, max(0, window - waveform.shape[-1]))))
    _, _, spectrum = signal.stft(waveform, fs=sample_rate, nperseg=window, noverlap=window - hop)
    magnitude, metadata = augment_magnitude(np.abs(spectrum), config, np.random.default_rng(seed), return_metadata=True)
    metadata.update(window_samples=window, hop_samples=hop, sample_rate=int(sample_rate))
    # Use original phase; masked magnitudes become exactly zero complex bins.
    augmented = magnitude * np.exp(1j * np.angle(spectrum))
    _, reconstructed = signal.istft(augmented, fs=sample_rate, nperseg=window, noverlap=window - hop)
    reconstructed = reconstructed[:, : audio.shape[0]]
    output = reconstructed[0] if audio.ndim == 1 else reconstructed.T
    return (output, metadata) if return_metadata else output


def augment_wav(input_path, output_path, config=None, *, seed=None, overwrite=False):
    """Read a local WAV and save float WAV samples without clipping or resampling."""
    source, destination = Path(input_path), Path(output_path)
    if source.resolve() == destination.resolve() or (destination.exists() and source.samefile(destination)):
        raise ValueError("Input and output must be different files.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    if destination.suffix.lower() != ".wav":
        raise ValueError("Output must have a .wav extension.")
    with sf.SoundFile(source) as stream:
        if stream.format not in ("WAV", "WAVEX", "RF64"):
            raise ValueError("Input must be a WAV file.")
        sample_rate = stream.samplerate
        audio = stream.read(dtype="float64")
    augmented = augment_waveform(audio, sample_rate, config, seed=seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, augmented, sample_rate, format="WAV", subtype="FLOAT")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible augmentation")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output")
    defaults = SpecAugmentConfig()
    for name in defaults.__dataclass_fields__:
        default = getattr(defaults, name)
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    args = parser.parse_args()
    try:
        config = SpecAugmentConfig(**{name: getattr(args, name) for name in defaults.__dataclass_fields__})
        output = augment_wav(args.input, args.output, config, seed=args.seed, overwrite=args.overwrite)
    except (ValueError, OSError, sf.LibsndfileError) as error:
        parser.exit(2, f"error: {error}\n")
    print(output)


if __name__ == "__main__":
    main()

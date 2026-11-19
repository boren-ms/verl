"""CPU coverage for offline SpecAugment-style WAV generation."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.specaugment_wav import SpecAugmentConfig, augment_magnitude, augment_wav, augment_waveform


def tone(samples=16000, rate=16000):
    t = np.arange(samples) / rate
    return 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 3000 * t)


@pytest.mark.parametrize("samples,rate,channels", [(1, 16000, 1), (73, 8000, 2), (22051, 22050, 1), (48001, 48000, 2)])
def test_disabled_policy_roundtrips_waveform(samples, rate, channels):
    audio = tone(samples, rate)
    if channels == 2:
        audio = np.column_stack((audio, -audio))
    config = SpecAugmentConfig(time_warp_param=0, num_freq_masks=0, num_time_masks=0)
    output = augment_waveform(audio, rate, config, seed=3)
    assert output.shape == audio.shape
    np.testing.assert_allclose(output, audio, atol=1e-12)


class FixedRandom:
    """Select known widths/starts to assert mask orientation and warp mapping."""

    def __init__(self, values):
        self.values = iter(values)

    def integers(self, low, high):
        value = next(self.values)
        assert low <= value < high
        return value


def test_masks_zero_only_selected_frequency_and_time_bands():
    original = np.ones((2, 8, 10))
    config = SpecAugmentConfig(
        time_warp_param=0,
        freq_mask_param=3,
        time_mask_param=4,
        num_freq_masks=1,
        num_time_masks=1,
        time_mask_max_ratio=0.2,
    )
    output = augment_magnitude(original, config, FixedRandom([3, 2, 2, 6]))
    expected = original.copy()
    expected[:, 2:5, :] = 0
    expected[:, :, 6:8] = 0
    np.testing.assert_array_equal(output, expected)
    assert original.all()


def test_time_warp_moves_anchor_and_preserves_endpoints():
    magnitude = np.broadcast_to(np.arange(20, dtype=float), (2, 8, 20)).copy()
    config = SpecAugmentConfig(time_warp_param=3, num_freq_masks=0, num_time_masks=0)
    output = augment_magnitude(magnitude, config, FixedRandom([8, 2]))
    assert output[0, 0, 10] == 8
    np.testing.assert_array_equal(output[..., 0], magnitude[..., 0])
    np.testing.assert_array_equal(output[..., -1], magnitude[..., -1])
    assert np.all(np.diff(output, axis=-1) > 0)


@pytest.mark.parametrize("samples", [1, 79, 16000, 32001])
def test_augmentation_is_finite_reproducible_and_shares_channel_policy(samples):
    audio = np.column_stack((tone(samples), tone(samples)))
    output = augment_waveform(audio, 16000, seed=42)
    assert output.shape == audio.shape
    assert np.isfinite(output).all()
    np.testing.assert_array_equal(output, augment_waveform(audio, 16000, seed=42))
    np.testing.assert_array_equal(output[:, 0], output[:, 1])
    if samples >= 16000:
        assert not np.allclose(output, audio)
        assert not np.allclose(output, augment_waveform(audio, 16000, seed=43))


def test_silence_stays_silent():
    np.testing.assert_array_equal(augment_waveform(np.zeros(32000), 16000, seed=5), np.zeros(32000))


def test_cli_saves_wav_and_refuses_overwrites(tmp_path):
    source, destination = tmp_path / "source.wav", tmp_path / "nested" / "augmented.wav"
    sf.write(source, np.column_stack((tone(), -tone())), 16000, subtype="PCM_16")
    original_bytes = source.read_bytes()
    script = Path(__file__).resolve().parents[2] / "scripts" / "specaugment_wav.py"
    command = [sys.executable, str(script), str(source), str(destination), "--seed", "42"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    output, rate = sf.read(destination)
    assert rate == 16000
    assert output.shape == (16000, 2)
    assert np.isfinite(output).all()
    assert sf.info(destination).subtype == "FLOAT"
    assert source.read_bytes() == original_bytes
    assert subprocess.run(command, capture_output=True).returncode == 2
    with pytest.raises(ValueError, match="different files"):
        augment_wav(source, source, overwrite=True)
    with pytest.raises(ValueError, match="different files"):
        alias = tmp_path / "alias.wav"
        alias.hardlink_to(source)
        augment_wav(source, alias, overwrite=True)
    subprocess.run([*command, "--overwrite"], check=True, capture_output=True)


@pytest.mark.parametrize("audio", [[], [np.nan], [np.inf], np.zeros((2, 1, 3))])
def test_invalid_audio_is_rejected(audio):
    with pytest.raises(ValueError):
        augment_waveform(audio, 16000)


@pytest.mark.parametrize("kwargs", [{"time_warp_param": -1}, {"time_mask_max_ratio": 1.1}, {"hop_ms": 25}])
def test_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SpecAugmentConfig(**kwargs)


def test_per_audio_trace_matches_random_ranges_and_reconstruction():
    audio = tone()
    config = SpecAugmentConfig(
        time_warp_param=20,
        freq_mask_param=8,
        num_freq_masks=1,
        time_mask_param=10,
        num_time_masks=1,
        time_mask_max_ratio=0.02,
    )
    traces = []
    for seed in range(12):
        output, sampled = augment_waveform(audio, 16000, config, seed=seed, return_metadata=True)
        np.testing.assert_array_equal(output, augment_waveform(audio, 16000, config, seed=seed))
        assert abs(sampled["warp"]["anchor_frame"] - sampled["warp"]["destination_frame"]) <= 20
        for mask in sampled["masks"]:
            axis_size = sampled["frequency_bins"] if mask["axis"] == "frequency" else sampled["time_frames"]
            limit = 8 if mask["axis"] == "frequency" else min(10, int(0.02 * axis_size))
            assert 0 <= mask["width"] <= limit
            assert 0 <= mask["start"] <= axis_size - mask["width"]
        traces.append(str(sampled))
    assert len(set(traces)) == 12

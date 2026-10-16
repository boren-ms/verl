import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.augment_audio import (
    augment_audio,
    mix_noise_at_snr,
    parse_args,
    read_manifest_audio_paths,
    resample_to_sample_rate,
    reverberate_audio,
)


@pytest.mark.parametrize("snr_db", [-5, 0, 20])
@pytest.mark.parametrize("channels", [1, 2])
def test_measured_snr(snr_db, channels):
    rng = np.random.default_rng(7)
    shape = (16000,) if channels == 1 else (16000, channels)
    speech = rng.normal(0, 0.1, shape).astype(np.float32)
    noise = rng.normal(0, 0.1, 500).astype(np.float32)
    mixed = mix_noise_at_snr(speech, noise, snr_db, random.Random(4))
    measured = 10 * np.log10(np.mean(speech**2) / np.mean((mixed - speech) ** 2))
    assert measured == pytest.approx(snr_db, abs=1e-4)


@pytest.mark.parametrize("channels", [1, 2])
def test_reverberation_matches_convolution(channels):
    speech = np.array([1, 0, 0, 0, 0], dtype=np.float32)
    if channels == 2:
        speech = np.column_stack([speech, speech * 0.5])
    rir = np.array([0, 1, 0.5, 0.25], dtype=np.float32)
    actual, metadata = reverberate_audio(speech, rir, 16000, 16000, random.Random(0))
    expected = np.array([1, 0.5, 0.25, 0, 0], dtype=np.float32)
    if channels == 2:
        expected = np.column_stack([expected, expected * 0.5])
    expected *= np.sqrt(np.mean(speech**2) / np.mean(expected**2))
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert metadata["rir_peak_delay_samples"] == 1
    assert actual.shape == speech.shape


def test_multichannel_rir_selects_one_channel_without_cancellation():
    speech = np.linspace(-0.1, 0.1, 200, dtype=np.float32)
    rir = np.array([[0, 0], [1, -1], [0.3, -0.3]], dtype=np.float32)
    actual, metadata = reverberate_audio(speech, rir, 16000, 16000, random.Random(2))
    assert metadata["rir_channel"] in (0, 1)
    assert np.mean(actual**2) == pytest.approx(np.mean(speech**2), rel=1e-5)


def test_polyphase_resampling_shape_and_antialiasing():
    time = np.arange(48000) / 48000
    audio = np.column_stack([np.sin(2 * np.pi * 1000 * time), np.sin(2 * np.pi * 12000 * time)])
    resampled = resample_to_sample_rate(audio, 48000, 16000)
    assert resampled.shape == (16000, 2)
    assert np.mean(resampled[100:-100, 1] ** 2) < 1e-5
    assert np.mean(resampled[100:-100, 0] ** 2) == pytest.approx(0.5, abs=0.005)


@pytest.mark.parametrize("bad", [np.zeros(10), np.array([]), np.array([np.nan]), np.array([np.inf])])
def test_invalid_signals_are_errors(bad):
    speech = np.ones(10)
    with pytest.raises(ValueError):
        reverberate_audio(speech, bad, 16000, 16000, random.Random(0))
    with pytest.raises(ValueError):
        mix_noise_at_snr(speech, bad, 10, random.Random(0))
    with pytest.raises(ValueError):
        mix_noise_at_snr(bad, speech, 10, random.Random(0))


def test_silent_noise_crop_is_error():
    with pytest.raises(ValueError, match="Selected noise segment"):
        mix_noise_at_snr(np.ones(5), np.r_[np.zeros(20), 1.0], 10, random.Random(1))


def test_reverb_then_noise_snr_reproducibility_and_peak(tmp_path):
    noise_path = tmp_path / "noise.wav"
    rir_path = tmp_path / "rir.wav"
    speech = np.random.default_rng(8).normal(0, 0.1, 2000).astype(np.float32)
    sf.write(noise_path, np.random.default_rng(5).normal(0, 0.1, 1000), 8000, subtype="FLOAT")
    sf.write(rir_path, np.array([0, 1, 0.5]), 8000, subtype="FLOAT")
    kwargs = {
        "audio": speech,
        "sample_rate": 16000,
        "noise_paths": [str(noise_path)],
        "rir_paths": [str(rir_path)],
        "snr_range": (-5, -5),
        "speed_prob": 0,
    }
    first, metadata = augment_audio(**kwargs, rng=random.Random(4))
    second, repeated_metadata = augment_audio(**kwargs, rng=random.Random(4))
    np.testing.assert_array_equal(first, second)
    assert metadata == repeated_metadata
    assert metadata["snr_db"] == -5
    assert metadata["rir_path"] == str(rir_path)
    assert len(first) == len(speech)
    assert np.max(np.abs(first)) <= 0.990001


@pytest.mark.parametrize("snr_db", [-5, 0, 20])
def test_snr_is_relative_to_reverberant_speech(tmp_path, snr_db):
    noise_path = tmp_path / "noise.wav"
    rir_path = tmp_path / "rir.wav"
    sf.write(noise_path, np.random.default_rng(5).normal(0, 0.1, 3000), 16000, subtype="FLOAT")
    sf.write(rir_path, [0, 1, 0.8, 0.4], 16000, subtype="FLOAT")
    kwargs = {
        "audio": np.random.default_rng(8).normal(0, 0.001, 2000).astype(np.float32),
        "sample_rate": 16000,
        "noise_paths": [str(noise_path)],
        "rir_paths": [str(rir_path)],
        "snr_range": (snr_db, snr_db),
        "speed_prob": 0,
    }
    reverberant, _ = augment_audio(**kwargs, rng=random.Random(4), noise_prob=0)
    mixture, _ = augment_audio(**kwargs, rng=random.Random(4))
    measured = 10 * np.log10(np.mean(reverberant**2) / np.mean((mixture - reverberant) ** 2))
    assert measured == pytest.approx(snr_db, abs=1e-4)


def test_no_rir_no_noise_no_speed_preserves_audio():
    speech = np.linspace(-0.1, 0.1, 300, dtype=np.float32)
    actual, metadata = augment_audio(speech, 16000, [], random.Random(1), noise_prob=0, speed_prob=0)
    np.testing.assert_array_equal(actual, speech)
    assert metadata["rir_path"] is None


@pytest.mark.parametrize("audio_key", ["audio_path", "audio_file", "audio"])
def test_cli_end_to_end_preserves_manifest_fields(tmp_path, audio_key):
    noise_dir = tmp_path / "noise"
    rir_dir = tmp_path / "rir"
    noise_dir.mkdir()
    rir_dir.mkdir()
    speech = np.random.default_rng(0).normal(0, 0.1, 4000)
    sf.write(tmp_path / "speech.wav", speech, 16000)
    sf.write(noise_dir / "noise.wav", np.random.default_rng(1).normal(0, 0.1, 800), 8000)
    sf.write(rir_dir / "rir.wav", [0, 1, 0.5, 0.1], 16000, subtype="FLOAT")
    manifest = tmp_path / "source.jsonl"
    manifest.write_text(json.dumps({"id": "utterance", audio_key: "speech.wav", "gts": ["hello"], "tag": 3}) + "\n")
    output = tmp_path / "output"
    script = Path(__file__).resolve().parents[2] / "scripts" / "augment_audio.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input_manifest",
            str(manifest),
            "--manifest_audio_base",
            str(tmp_path),
            "--noise_dir",
            str(noise_dir),
            "--rir_dir",
            str(rir_dir),
            "--output_dir",
            str(output),
            "--snr_min",
            "0",
            "--snr_max",
            "0",
            "--speed_prob",
            "0",
            "--seed",
            "7",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    row = json.loads((output / "data.jsonl").read_text())
    assert row["gts"] == ["hello"]
    assert row["tag"] == 3
    assert row["snr_db"] == 0
    assert row["augmentation_seed"] == 7
    assert row["rir_path"] == str(rir_dir / "rir.wav")
    audio, sr = sf.read(row["audio_path"])
    assert sr == 16000
    assert len(audio) == len(speech)
    assert np.max(np.abs(audio)) < 1
    source, _ = sf.read(tmp_path / "speech.wav", dtype="float32")
    rir, _ = sf.read(rir_dir / "rir.wav", dtype="float32")
    reference, _ = reverberate_audio(source, rir, 16000, 16000, random.Random(0))
    measured = 10 * np.log10(np.mean(reference**2) / np.mean((audio - reference) ** 2))
    assert measured == pytest.approx(0, abs=0.01)


@pytest.mark.parametrize(
    "options",
    [
        ["--snr_min", "nan"],
        ["--snr_min", "20", "--snr_max", "-5"],
        ["--speed_max", "inf"],
        ["--rir_prob", "-1"],
        ["--rir_prob", "nan"],
        ["--max_examples", "0"],
        ["--input_manifest", "also.jsonl"],
    ],
)
def test_invalid_cli_options(monkeypatch, options):
    monkeypatch.setattr(
        sys,
        "argv",
        ["augment_audio.py", "--input_dir", "speech", "--output_dir", "out", "--noise_dir", "noise", *options],
    )
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 2


def test_manifest_missing_audio_is_error(tmp_path):
    manifest = tmp_path / "invalid.jsonl"
    manifest.write_text('{"text": "missing audio"}\n')
    with pytest.raises(ValueError, match=":1: expected"):
        read_manifest_audio_paths(str(manifest))


@pytest.mark.parametrize("snr_db", [float("nan"), float("inf"), -100000, 100000])
def test_invalid_snr_is_error(snr_db):
    with pytest.raises(ValueError, match="SNR"):
        mix_noise_at_snr(np.ones(10), np.ones(10), snr_db, random.Random(0))

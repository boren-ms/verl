import io
import random

import numpy as np
import pytest
import soundfile as sf

from scripts.build_cv11_augmented_dataset import file_sha256, load_metadata, make_example


@pytest.mark.parametrize("snr", [0.0, 12.0])
@pytest.mark.parametrize("level", [0.1, 0.000001])
def test_speed_rir_noise_preserves_text_and_verifies_pcm16_snr(tmp_path, snr, level):
    sr = 16000
    speech = np.sin(np.arange(sr * 2) * 0.1).astype(np.float32) * level
    buf = io.BytesIO()
    sf.write(buf, speech, sr, format="WAV", subtype="FLOAT")
    rir = tmp_path / "rir.wav"
    noise = tmp_path / "noise.wav"
    sf.write(rir, np.array([0, 1, 0.5, 0.3, 0.1]), sr, subtype="FLOAT")
    sf.write(noise, np.random.default_rng(42).normal(0, 0.1, sr * 3), sr)
    records = [{"audio_path": str(p), "sha256": file_sha256(p), "license": "CC-BY-4.0"} for p in [rir, noise]]
    cfg = {"seed": 42, "rir_seed": 43, "speed_range": [1.5, 1.5], "snr_range": [snr, snr],
           "sample_rate": sr, "output_dir": "az://test/data", "data_source": "noisy_cv11_en_train",
           "source_upstream": "mozilla-foundation/common_voice_11_0", "source_release": "cv11",
           "source_repo": "test/mirror", "source_revision": "fixed", "source_split": "train", "source_license": "CC0-1.0"}
    row = make_example(0, "common_voice_en_1.mp3", 'A "quoted" sentence, with digits 12.',
                       buf.getvalue(), "train.tar", "en_train_0", cfg, [records[0]], [records[1]],
                       tmp_path, tmp_path, tmp_path, "fingerprint")
    local = tmp_path / "common_voice_en_1.wav"
    output, output_sr = sf.read(local)
    assert output_sr == sr and len(output) == round(len(speech) / 1.5)
    assert row["text"] == 'A "quoted" sentence, with digits 12.'
    assert row["source_split"] == "train"
    assert row["augmentation_order"] == ["speed", "rir", "noise"]
    assert abs(row["measured_snr_db_after_pcm16"] - snr) < 0.02
    assert row["audio_sha256"] == file_sha256(local)
    # Replay creates byte-identical WAVs and metadata independently of global RNG.
    random.seed(999)
    replay = make_example(0, "common_voice_en_1.mp3", row["text"], buf.getvalue(), "train.tar",
                          "en_train_0", cfg, [records[0]], [records[1]], tmp_path, tmp_path, tmp_path, "fingerprint")
    assert row == replay


def test_tsv_membership_and_exact_transcripts(tmp_path):
    source = tmp_path / "train.tsv"
    source.write_text('path\tsentence\tlocale\na.mp3\tA "quoted" sentence.\ten\n')
    assert load_metadata(source, 1) == {"a.mp3": (0, 'A "quoted" sentence.')}
    with pytest.raises(ValueError, match="count/uniqueness"):
        load_metadata(source, 2)
    source.write_text("path\tsentence\tlocale\na.mp3\t\ten\n")
    with pytest.raises(ValueError, match="empty text"):
        load_metadata(source, 1)

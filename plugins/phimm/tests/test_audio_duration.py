"""Check segment duration limits using in-memory waveforms only."""

from types import SimpleNamespace

import numpy as np
import pytest

from recipe.phimm.sft.dataset import ASRSFTDataset
from recipe.phimm.utils.audio import load_audio


@pytest.mark.parametrize("seconds", [2, 4, 8, 30])
def test_segment_at_duration_limit_preserves_every_sample(seconds):
    waveform = np.arange(seconds * 16000, dtype=np.float32)
    audio, sr = load_audio({"audio": waveform, "sr": 16000}, max_dur=seconds)
    assert sr == 16000
    np.testing.assert_array_equal(audio, waveform)
    with pytest.raises(ValueError, match="exceeds max_audio_dur"):
        load_audio({"audio": np.append(waveform, 0), "sr": 16000}, max_dur=seconds)


def test_no_duration_limit_keeps_complete_audio():
    waveform = np.arange(640001, dtype=np.float32)
    np.testing.assert_array_equal(load_audio({"audio": waveform, "sr": 16000})[0], waveform)


@pytest.mark.parametrize("is_train", [False, True])
def test_sft_rejects_overlong_audio_before_processing(is_train):
    dataset = ASRSFTDataset.__new__(ASRSFTDataset)
    dataset.ds = [
        {"audio": np.zeros(16001, dtype=np.float32), "sr": 16000, "reward_model": {"gt_output": "full target"}}
    ]
    dataset.config = SimpleNamespace(max_audio_dur=1 if is_train else None, val_max_audio_dur=None if is_train else 1)
    dataset.is_train = is_train
    with pytest.raises(ValueError, match="exceeds max_audio_dur=1"):
        dataset[0]

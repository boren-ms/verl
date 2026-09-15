import numpy as np
import pytest
import soundfile as sf

from recipe.phimm.utils.audio import _is_time_chunk_spec, load_raw_audio


def test_load_raw_audio_reads_percentage_range(tmp_path):
    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, np.arange(1000, dtype=np.float32) / 1000, 16000)

    audio, sample_rate = load_raw_audio({"audio_path": f"{audio_path}#0%:10%"})

    assert sample_rate == 16000
    assert len(audio) == 100


@pytest.mark.parametrize(
    ("segment", "expected_length"),
    [("0.1:-0.1", 800), ("0.01:-0.02", 970), ("0.01:", 990), (":-0.02", 980)],
)
def test_load_raw_audio_reads_edge_relative_ranges(tmp_path, segment, expected_length):
    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, np.arange(1000, dtype=np.float32) / 1000, 1000)

    audio, sample_rate = load_raw_audio({"audio_path": f"{audio_path}#{segment}"})

    assert sample_rate == 1000
    assert len(audio) == expected_length


def test_time_chunk_spec_accepts_seconds_and_percentages():
    assert _is_time_chunk_spec("sample.wav#0:1.5")
    assert _is_time_chunk_spec("sample.wav#0%:10%")
    assert _is_time_chunk_spec("sample.wav#0.1:-0.1")
    assert _is_time_chunk_spec("sample.wav#0.1:")
    assert _is_time_chunk_spec("sample.wav#:-0.1")
    assert not _is_time_chunk_spec("sample.wav#start:end")
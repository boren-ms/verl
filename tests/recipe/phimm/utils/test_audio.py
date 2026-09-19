import numpy as np
import pytest
import soundfile as sf

from recipe.phimm.utils import audio as audio_module
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


def test_load_raw_audio_skips_overlapping_edge_relative_cuts(tmp_path, caplog):
    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, np.arange(1000, dtype=np.float32) / 1000, 1000)

    audio, sample_rate = load_raw_audio({"audio_path": f"{audio_path}#0.6:-0.5"})

    assert sample_rate == 1000
    assert len(audio) == 1000
    assert "Skipping overlapping edge cuts" in caplog.text


def test_load_raw_audio_skips_edge_cuts_leaving_too_little_audio(tmp_path, caplog):
    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, np.arange(1000, dtype=np.float32) / 1000, 1000)

    audio, sample_rate = load_raw_audio({"audio_path": f"{audio_path}#0.47:-0.43"})

    assert sample_rate == 1000
    assert len(audio) == 1000
    assert "Skipping edge cuts" in caplog.text
    assert "leave only 0.100s" in caplog.text


def test_load_raw_audio_slices_nested_chunk_spec(monkeypatch):
    source = np.arange(1000, dtype=np.float32)
    monkeypatch.setattr(audio_module, "_chunk_load_mode", "sample")
    monkeypatch.setattr(audio_module, "load_chunk_sample", lambda spec: (source, 1000))

    audio, sample_rate = load_raw_audio(
        {"audio_path": "az://container/chunk.audio:2000:464#0.1:-0.2"}
    )

    assert sample_rate == 1000
    np.testing.assert_array_equal(audio, source[100:800])


def test_load_raw_audio_skips_edge_cuts_longer_than_nested_chunk(monkeypatch, caplog):
    source = np.arange(500, dtype=np.float32)
    monkeypatch.setattr(audio_module, "_chunk_load_mode", "sample")
    monkeypatch.setattr(audio_module, "load_chunk_sample", lambda spec: (source, 1000))

    audio, sample_rate = load_raw_audio(
        {"audio_path": "az://container/chunk.audio:2000:464#0.81393:-0.160458"}
    )

    assert sample_rate == 1000
    np.testing.assert_array_equal(audio, source)
    assert "Skipping overlapping edge cuts" in caplog.text


def test_time_chunk_spec_accepts_seconds_and_percentages():
    assert _is_time_chunk_spec("sample.wav#0:1.5")
    assert _is_time_chunk_spec("sample.wav#0%:10%")
    assert _is_time_chunk_spec("sample.wav#0.1:-0.1")
    assert _is_time_chunk_spec("sample.wav#0.1:")
    assert _is_time_chunk_spec("sample.wav#:-0.1")
    assert not _is_time_chunk_spec("sample.wav#start:end")
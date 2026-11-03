import io

import numpy as np
import pytest
import soundfile as sf

from recipe.phimm.utils import audio as audio_module
from recipe.phimm.utils.audio import _is_time_chunk_spec, load_audio, load_raw_audio, load_raw_audios


def test_load_raw_audio_reads_huggingface_embedded_bytes():
    buffer = io.BytesIO()
    expected = np.arange(1000, dtype=np.float32) / 1000
    sf.write(buffer, expected, 16000, format="WAV", subtype="FLOAT")

    audio, sample_rate = load_raw_audio(
        {"audio": {"bytes": buffer.getvalue(), "path": "sample.wav"}}
    )

    assert sample_rate == 16000
    np.testing.assert_array_equal(audio, expected)


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


def test_concatenated_audio_paths_preserve_order_and_apply_duration_to_combination(tmp_path):
    first, second = tmp_path / "first.wav", tmp_path / "second.wav"
    sf.write(first, np.full(3200, 0.25), 16000, subtype="FLOAT")
    sf.write(second, np.tile([0.5, 1.0], (3200, 1)), 16000, subtype="FLOAT")
    source = f"{first}; {second}"

    audio, sr = load_raw_audio({"audio_path": source})
    assert sr == 16000
    np.testing.assert_array_equal(audio, np.r_[np.full(3200, 0.25), np.full(3200, 0.75)])
    limited, sr = load_audio({"audio_path": source}, max_dur=0.3)
    assert sr == 16000
    np.testing.assert_array_equal(limited, audio[:4800])
    batched = list(load_raw_audios({"audio_path": [source, str(first)]}))
    np.testing.assert_array_equal(batched[0][0], audio)
    np.testing.assert_array_equal(batched[1][0], audio[:3200])


def test_concatenated_audio_resamples_different_sample_rates(tmp_path):
    first, second = tmp_path / "first.wav", tmp_path / "second.wav"
    sf.write(first, np.full(1600, 0.25), 8000, subtype="FLOAT")
    sf.write(second, np.full(3200, 0.75), 16000, subtype="FLOAT")

    audio, sr = load_raw_audio({"audio_path": f"{first};{second}"})

    assert sr == 16000
    assert len(audio) == 6400
    np.testing.assert_allclose(audio[3200:], 0.75)
    np.testing.assert_allclose(audio[100:3100], 0.25, atol=1e-3)


def test_concatenated_sources_support_time_ranges_and_chunk_specs(tmp_path, monkeypatch):
    path = tmp_path / "audio.wav"
    sf.write(path, np.arange(1000, dtype=np.float32) / 1000, 1000, subtype="FLOAT")
    monkeypatch.setattr(audio_module, "_chunk_load_mode", "sample")
    monkeypatch.setattr(audio_module, "load_chunk_sample", lambda spec: (np.ones(50), 1000))

    audio, sr = load_raw_audio({"audio_path": f"{path}#0.1:0.2;chunk.audio:10:2"})

    assert sr == 1000
    np.testing.assert_allclose(audio[:100], np.arange(100, 200) / 1000)
    np.testing.assert_array_equal(audio[100:], np.ones(50))


def test_concatenated_remote_audio_loads_through_individual_cache_files(tmp_path, monkeypatch):
    from verl import audio_cache

    monkeypatch.setattr(audio_cache, "LOCAL_DATA_ROOT", tmp_path / "cache")
    monkeypatch.setattr(audio_cache, "LOCK_ROOT", tmp_path / "locks")
    copied = []

    def transfer(source, destination):
        copied.append(source)
        value = 0.25 if source.endswith("a.wav") else 0.75
        sf.write(destination, np.full(3200, value), 16000, format="WAV", subtype="FLOAT")

    monkeypatch.setattr(audio_cache, "_run_bbb_transfer", transfer)
    prefix = "az://orngwus2cresco/data/test/"
    first, second = prefix + "a.wav", prefix + "b.wav"
    audio, rate = load_audio({"audio_path": first + ";" + second})
    assert rate == 16000
    np.testing.assert_array_equal(audio, np.r_[np.full(3200, 0.25), np.full(3200, 0.75)])
    assert copied == [first, second]
    load_audio({"audio_path": second + ";" + first})
    assert copied == [first, second]


@pytest.mark.parametrize("source", [";first.wav", "first.wav;", "first.wav; ;second.wav"])
def test_concatenated_audio_rejects_empty_path(source):
    with pytest.raises(ValueError, match="Empty audio path"):
        load_raw_audio({"audio_path": source})

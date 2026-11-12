import io

import numpy as np
import pytest
import soundfile as sf

from recipe.phimm.utils import audio as audio_module
from recipe.phimm.data.chunk import load_chunk_sample
from recipe.phimm.data import indexed_chunk
from recipe.phimm.data.indexed_chunk import _offsets, load_chunk_sample_indexed


def _audio(index):
    stream = io.BytesIO()
    sf.write(stream, np.linspace(-0.2, 0.2, 160 + index), 16000, format="WAV")
    return stream.getvalue()


def _write(path, kind, count):
    with path.open("wb") as stream:
        stream.write(kind.encode() + (1).to_bytes(4, "little"))
        for index in range(count):
            stream.write(index.to_bytes(4, "little"))
            if kind == "audios":
                stream.write((2).to_bytes(4, "little"))
            for subindex in range(2 if kind == "audios" else 1):
                payload = _audio(index + subindex)
                stream.write(len(payload).to_bytes(4, "little") + payload)


@pytest.mark.parametrize("kind", ["audio", "audios"])
def test_indexed_reads_match_original_in_random_order(tmp_path, kind):
    path = tmp_path / f"chunk.{kind}"
    _write(path, kind, 12)
    _offsets.cache_clear()
    for index in [11, 0, 5, 1, 11, 7]:
        spec = f"{path}:12:{index}"
        old, new = load_chunk_sample(spec), load_chunk_sample_indexed(spec)
        for expected, actual in zip(
            old if kind == "audios" else [old], new if kind == "audios" else [new], strict=True
        ):
            assert expected[1] == actual[1]
            np.testing.assert_array_equal(expected[0], actual[0])
    assert _offsets.cache_info().misses == 1


def test_replaced_chunk_invalidates_offset_cache(tmp_path):
    path = tmp_path / "chunk.audio"
    _write(path, "audio", 2)
    load_chunk_sample_indexed(f"{path}:2:1")
    replacement = tmp_path / "replacement.audio"
    _write(replacement, "audio", 3)
    replacement.replace(path)
    old = load_chunk_sample(f"{path}:3:2")
    new = load_chunk_sample_indexed(f"{path}:3:2")
    np.testing.assert_array_equal(old[0], new[0])


def test_truncated_chunk_rejected(tmp_path):
    path = tmp_path / "chunk.audio"
    _write(path, "audio", 2)
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(ValueError, match="Truncated"):
        load_chunk_sample_indexed(f"{path}:2:0")


def test_default_audio_route_reads_indexed_chunk(tmp_path, monkeypatch):
    path = tmp_path / "chunk.audio"
    _write(path, "audio", 2)

    def unexpected_reader(spec):
        pytest.fail("Default chunk reading must use the offset index")

    monkeypatch.setattr(audio_module, "load_chunk_sample", unexpected_reader)
    monkeypatch.setattr(audio_module, "load_chunk_example", unexpected_reader)
    assert audio_module._chunk_load_mode == "indexed"
    actual, rate = audio_module.load_raw_audio({"audio_chunk": f"{path}:2:1"})
    expected, expected_rate = sf.read(io.BytesIO(_audio(1)))
    assert rate == expected_rate
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mode, reader", [("sample", "load_chunk_sample"), ("cached", "load_chunk_example")])
def test_explicit_legacy_modes_still_select_their_reader(monkeypatch, mode, reader):
    expected = np.arange(160)
    monkeypatch.setattr(audio_module, "_chunk_load_mode", mode)
    monkeypatch.setattr(audio_module, reader, lambda spec: (expected, 16000))
    actual, rate = audio_module._load_chunk("chunk.audio:2:1")
    assert rate == 16000
    np.testing.assert_array_equal(actual, expected)


def test_nonlocalized_remote_chunk_uses_blob_metadata(tmp_path, monkeypatch):
    path = tmp_path / "chunk.audio"
    _write(path, "audio", 2)
    remote = "az://external-container/chunk.audio"
    real_open = indexed_chunk.bf.BlobFile
    metadata = indexed_chunk.bf.stat(str(path))
    monkeypatch.setattr(indexed_chunk, "localize_audio_source", lambda source: source)
    monkeypatch.setattr(indexed_chunk.bf, "stat", lambda source: metadata)
    monkeypatch.setattr(
        indexed_chunk.bf, "BlobFile", lambda source, mode: real_open(str(path) if source == remote else source, mode)
    )
    actual, rate = load_chunk_sample_indexed(f"{remote}:2:1")
    expected, expected_rate = sf.read(io.BytesIO(_audio(1)))
    assert rate == expected_rate
    np.testing.assert_array_equal(actual, expected)


def test_index_cache_is_bounded(tmp_path):
    _offsets.cache_clear()
    for index in range(34):
        path = tmp_path / f"chunk_{index}.audio"
        _write(path, "audio", 1)
        load_chunk_sample_indexed(f"{path}:1:0")
    assert _offsets.cache_info().currsize == 32

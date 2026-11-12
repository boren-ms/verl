import io
import json
import ast
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from recipe.phimm.utils import audio as audio_module
from recipe.phimm.data.chunk import load_chunk_sample
from recipe.phimm.data import indexed_chunk
from recipe.phimm.data.indexed_chunk import _offsets, load_chunk_sample_indexed
from recipe.phimm.data.chunk import create_chunk_datasets, load_examples_from_chunks
from verl.audio_chunk import read_chunk_audio, split_chunk_audio_source
from verl import audio_cache


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


@pytest.mark.parametrize("kind", ["audio", "audios"])
def test_manifest_expansion_materializes_byte_paths_and_preserves_labels(tmp_path, monkeypatch, kind):
    audio = tmp_path / f"chunk.{kind}"
    _write(audio, kind, 3)
    text = tmp_path / "chunk.transcription"
    with text.open("wb") as stream:
        stream.write(b"transcription" + (1).to_bytes(4, "little"))
        for index, value in enumerate(["first", "second", "third"]):
            data = value.encode()
            stream.write(index.to_bytes(4, "little") + len(data).to_bytes(4, "little") + data)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"fileInfo": [{"name": "chunk", "count": 3}]}))
    ds = create_chunk_datasets(
        specs=[{"manifest_file": str(manifest), "chunk_path": str(tmp_path), "language": "en"}],
        chunk_types=[kind, "transcription"],
        chunk_shuffle=False,
        num_proc=1,
        index_audio=True,
    )
    assert list(ds["audio_chunk"]) == [f"{audio}:3:{i}" for i in range(3)]
    assert list(ds["transcription"]) == ["first", "second", "third"]
    assert all(split_chunk_audio_source(row["audio_path"]) is not None for row in ds)
    from recipe.phimm.data.dataset import verl_format_ds

    formatted = verl_format_ds(ds.rename_column("transcription", "text").add_column("prompt", ["Transcribe"] * 3))
    output = tmp_path / "indexed.jsonl"
    formatted.to_json(str(output), force_ascii=False)
    saved = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["reward_model"]["ground_truth"] for row in saved] == ["first", "second", "third"]
    assert [row["audio_chunk"] for row in saved] == list(ds["audio_chunk"])
    assert [row["audio_path"] for row in saved] == list(ds["audio_path"])

    def unexpected_scan(*args, **kwargs):
        pytest.fail("Byte-indexed rows must not build or scan an ordinal index during audio reads")

    monkeypatch.setattr(indexed_chunk, "_offsets", unexpected_scan)
    for index, row in enumerate(saved):
        actual, rate = audio_module.load_raw_audio(row)
        expected, expected_rate = sf.read(io.BytesIO(_audio(index)))
        assert rate == expected_rate
        np.testing.assert_array_equal(actual, expected)
    # Time slicing still applies to the decoded payload, not the chunk file.
    row = ds[1]
    row["audio_path"] += "#0%:50%"
    actual, _ = audio_module.load_raw_audio(row)
    assert len(actual) == 161 // 2


def test_indexed_manifest_missing_chunk_preserves_empty_schema(tmp_path):
    result = load_examples_from_chunks(
        {"name": ["missing"], "count": [2], "chunk_path": [str(tmp_path)]},
        ["audio", "transcription"],
        index_audio=True,
    )
    assert result == {"audio_chunk": [], "audio_path": [], "transcription": [], "language": []}


def test_manifest_expansion_defaults_to_metadata_only(tmp_path):
    # Invalid audio bytes prove default expansion does not open/index audio.
    audio = tmp_path / "chunk.audio"
    audio.write_bytes(b"not an audio chunk")
    payload = b"original reference"
    (tmp_path / "chunk.transcription").write_bytes(
        b"transcription"
        + (1).to_bytes(4, "little")
        + (0).to_bytes(4, "little")
        + len(payload).to_bytes(4, "little")
        + payload
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"fileInfo": [{"name": "chunk", "count": 1}]}))
    ds = create_chunk_datasets(
        specs=[{"manifest_file": str(manifest), "chunk_path": str(tmp_path)}],
        chunk_shuffle=False,
        num_proc=1,
    )
    assert len(ds) == 1
    assert ds[0]["audio_chunk"] == f"{audio}:1:0"
    assert ds[0]["transcription"] == "original reference"
    assert "audio_path" not in ds.column_names


def test_byte_reference_cache_maps_the_physical_file(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_cache, "LOCAL_DATA_ROOT", tmp_path)
    source = "az://orngwus2cresco/data/test/chunk.audios!27:123#0.1:0.2"
    assert audio_cache._split_audio_source(source) == (
        "az://orngwus2cresco/data/test/chunk.audios",
        "!27:123#0.1:0.2",
    )
    assert audio_cache.local_audio_source(source) == str(tmp_path / "test/chunk.audios") + "!27:123#0.1:0.2"


@pytest.mark.parametrize("source", ["chunk.audio!-1:20", "chunk.audio!0:0", "chunk.audios!x:20"])
def test_invalid_byte_reference_rejected(source):
    with pytest.raises(ValueError, match="Invalid chunk byte reference"):
        split_chunk_audio_source(source)


def test_byte_payload_truncation_rejected(tmp_path):
    path = tmp_path / "chunk.audio"
    path.write_bytes(b"abc")
    with pytest.raises(EOFError, match="Truncated chunk payload"):
        read_chunk_audio(f"{path}!1:10")


def test_generic_audio_reader_supports_materialized_byte_reference(tmp_path):
    # Isolate the actual audio functions from unrelated torchvision imports.
    from cachetools import FIFOCache, cached
    import blobfile as bf

    path = Path(__file__).parents[4] / "verl/utils/dataset/vision_utils.py"
    functions = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in ("sf_read", "process_audio")
    ]
    namespace = {
        "cached": cached,
        "FIFOCache": FIFOCache,
        "BytesIO": io.BytesIO,
        "sf": sf,
        "bf": bf,
        "split_chunk_audio_source": split_chunk_audio_source,
        "read_chunk_audio": read_chunk_audio,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    audio = tmp_path / "chunk.audio"
    _write(audio, "audio", 2)
    reference = indexed_chunk.chunk_audio_byte_references(str(audio), 2)[1]
    actual, rate = namespace["process_audio"](reference)
    expected, expected_rate = sf.read(io.BytesIO(_audio(1)))
    assert rate == expected_rate
    np.testing.assert_array_equal(actual, expected)

import json
import zipfile

import blobfile as bf
import numpy as np
import pytest
import soundfile as sf

from recipe.phimm.cache_eval_audio import collect_audio_files
from recipe.phimm.utils.audio import load_raw_audio
from scripts.pack_audio_zip import main, pack_audio_zip, pack_parquet_audio_zip
from verl import audio_cache
from verl.audio_zip import read_zip_audio, split_zip_audio_source


def make_dataset(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    expected = np.arange(1000, dtype=np.float32) / 1000
    for name in ("a.wav", "b.wav"):
        sf.write(audio_dir / name, expected, 1000, subtype="FLOAT")
    rows = [
        {"id": "first", "audio_path": "audio/a.wav", "text": "hello", "keywords": ["hello"]},
        {"id": "segment", "audio_path": "audio/a.wav#0.1:0.5", "text": "world"},
        {"id": "last", "audio_path": "audio/b.wav", "text": "last"},
    ]
    manifest = tmp_path / "data.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest, rows, expected


def test_pack_and_load_preserves_bytes_rows_metadata_and_segments(tmp_path):
    manifest, original, expected = make_dataset(tmp_path)
    archive = str(tmp_path / "audio.zip")
    output = str(tmp_path / "data_zip.jsonl")
    summary = pack_audio_zip(str(manifest), archive, output, workers=2)
    assert summary["rows"] == 3
    assert summary["unique_files"] == 2
    rows = [json.loads(line) for line in bf.BlobFile(output)]
    with zipfile.ZipFile(archive) as packed:
        assert packed.testzip() is None
        assert len(packed.namelist()) == 2
        for row, source_row in zip(rows, original, strict=True):
            assert row["original_audio_path"] == source_row["audio_path"]
            assert {key: row[key] for key in source_row if key != "audio_path"} == {
                key: value for key, value in source_row.items() if key != "audio_path"
            }
            source = source_row["audio_path"].split("#")[0]
            reference = row["audio_path"].split("#")[0]
            assert read_zip_audio(reference) == (tmp_path / source).read_bytes()
            assert packed.read(row["audio_zip_member"]) == read_zip_audio(reference)
            assert packed.getinfo(row["audio_zip_member"]).compress_type == zipfile.ZIP_STORED
            assert packed.getinfo(row["audio_zip_member"]).extract_version == 45
            audio, rate = load_raw_audio(row)
            assert rate == 1000
            np.testing.assert_array_equal(audio, expected[100:500] if row["id"] == "segment" else expected)
    assert rows[1]["audio_path"] == rows[0]["audio_path"] + "#0.1:0.5"
    assert manifest.read_text() == "".join(json.dumps(row) + "\n" for row in original)
    config = tmp_path / "dataset.yaml"
    config.write_text(f"dataset_name: jsonl\njsonl_paths: {output}\n")
    assert collect_audio_files(str(config)) == [archive]


def test_remote_zip_reads_only_member_range(tmp_path, monkeypatch):
    manifest, _, _ = make_dataset(tmp_path)
    archive = str(tmp_path / "audio.zip")
    output = tmp_path / "packed.jsonl"
    pack_audio_zip(str(manifest), archive, str(output))
    row = json.loads(output.read_text().splitlines()[0])
    _, offset, size = split_zip_audio_source(row["audio_path"])
    remote = "az://other/data/audio.zip"
    calls = []
    original_open = bf.BlobFile

    def open_remote(path, mode, **kwargs):
        calls.append((path, mode, kwargs))
        return original_open(archive, mode, **kwargs)

    monkeypatch.setattr(bf, "BlobFile", open_remote)
    assert read_zip_audio(f"{remote}!{offset}:{size}") == (tmp_path / "audio/a.wav").read_bytes()
    assert calls == [(remote, "rb", {"streaming": True, "buffer_size": size})]


def test_default_rl_audio_loader_supports_zip(tmp_path):
    from verl.utils.dataset.vision_utils import process_audio

    manifest, _, expected = make_dataset(tmp_path)
    output = tmp_path / "packed.jsonl"
    pack_audio_zip(str(manifest), str(tmp_path / "audio.zip"), str(output))
    row = json.loads(output.read_text().splitlines()[0])
    audio, rate = process_audio(row["audio_path"])
    assert rate == 1000
    np.testing.assert_array_equal(audio, expected)


def test_cache_maps_and_copies_archive_once(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_cache, "LOCAL_DATA_ROOT", tmp_path / "cache")
    monkeypatch.setattr(audio_cache, "LOCK_ROOT", tmp_path / "locks")
    archive = "az://orngwus2cresco/data/test/audio.zip"
    copied = []

    def transfer(source, destination):
        copied.append(source)
        destination.write_bytes(b"zip")

    monkeypatch.setattr(audio_cache, "_run_bbb_transfer", transfer)
    for selector in ("!100:20", "!200:30#0%:50%"):
        source = archive + selector
        assert audio_cache._split_audio_source(source) == (archive, selector)
        expected = str(tmp_path / "cache/test/audio.zip") + selector
        assert audio_cache.cache_audio_source(source) == expected
        assert audio_cache.resolve_audio_source(source) == expected
    assert copied == [archive]


def test_loader_prefers_cached_archive(tmp_path, monkeypatch):
    manifest, _, expected = make_dataset(tmp_path)
    output = tmp_path / "packed.jsonl"
    local_zip = tmp_path / "cache/test/audio.zip"
    pack_audio_zip(str(manifest), str(local_zip), str(output))
    row = json.loads(output.read_text().splitlines()[0])
    _, offset, size = split_zip_audio_source(row["audio_path"])
    monkeypatch.setattr(audio_cache, "LOCAL_DATA_ROOT", tmp_path / "cache")
    audio, rate = load_raw_audio(
        {"audio_path": f"az://orngwus2cresco/data/test/audio.zip!{offset}:{size}"}
    )
    assert rate == 1000
    np.testing.assert_array_equal(audio, expected)


def test_local_zip_reference_is_absolute(tmp_path, monkeypatch):
    manifest, _, _ = make_dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    pack_audio_zip(str(manifest), "audio.zip", "packed.jsonl")
    row = json.loads((tmp_path / "packed.jsonl").read_text().splitlines()[0])
    assert split_zip_audio_source(row["audio_path"])[0] == str(tmp_path / "audio.zip")


@pytest.mark.parametrize("reference", ["a.zip!foo:20", "a.zip!0:0", "a.zip!-1:20"])
def test_invalid_zip_reference_fails(reference):
    with pytest.raises(ValueError, match="Invalid ZIP audio reference"):
        split_zip_audio_source(reference)


def test_truncated_member_fails(tmp_path):
    archive = tmp_path / "short.zip"
    archive.write_bytes(b"abc")
    with pytest.raises(EOFError, match="Truncated ZIP audio"):
        read_zip_audio(f"{archive}!1:10")


def test_pack_refuses_overwrite(tmp_path):
    manifest, _, _ = make_dataset(tmp_path)
    archive = tmp_path / "audio.zip"
    archive.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        pack_audio_zip(str(manifest), str(archive), str(tmp_path / "packed.jsonl"))
    assert archive.read_bytes() == b"existing"


@pytest.mark.parametrize("content", ["", "{}\n", "not json\n", '{"audio_path":"file.audio:20:1"}\n'])
def test_invalid_manifest_does_not_publish(tmp_path, content):
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(content)
    with pytest.raises(ValueError):
        pack_audio_zip(str(manifest), str(tmp_path / "audio.zip"), str(tmp_path / "packed.jsonl"))
    assert not (tmp_path / "audio.zip").exists()
    assert not (tmp_path / "packed.jsonl").exists()


def test_missing_audio_does_not_publish(tmp_path):
    manifest = tmp_path / "data.jsonl"
    manifest.write_text('{"audio_path":"missing.wav"}\n')
    with pytest.raises(FileNotFoundError):
        pack_audio_zip(str(manifest), str(tmp_path / "audio.zip"), str(tmp_path / "packed.jsonl"))
    assert not (tmp_path / "audio.zip").exists()
    assert not (tmp_path / "packed.jsonl").exists()


def test_cli(tmp_path, monkeypatch, capsys):
    manifest, _, _ = make_dataset(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pack_audio_zip", str(manifest), "--zip-path", str(tmp_path / "audio.zip"),
            "--output-jsonl", str(tmp_path / "packed.jsonl"), "--workers", "2",
        ],
    )
    main()
    assert json.loads(capsys.readouterr().out)["unique_files"] == 2


def test_pack_parquet_preserves_embedded_audio_and_metadata(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    manifest, _, expected = make_dataset(tmp_path)
    audio_bytes = (tmp_path / "audio/a.wav").read_bytes()
    for index in range(2):
        pq.write_table(
            pa.Table.from_pylist([{
                "id": f"row-{index}", "text": "hello", "speaker_id": index,
                "lattice": [["hello", "world"]],
                "audio": {"bytes": audio_bytes, "path": "sample.wav"},
            }]),
            tmp_path / f"part-{index}.parquet",
        )
    output = tmp_path / "parquet_zip.jsonl"
    summary = pack_parquet_audio_zip(
        str(tmp_path / "*.parquet"), str(tmp_path / "parquet.zip"), str(output)
    )
    assert summary["rows"] == 2
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    for index, row in enumerate(rows):
        assert row["id"] == f"row-{index}"
        assert row["text"] == "hello"
        assert row["speaker_id"] == index
        assert row["lattice"] == [["hello", "world"]]
        assert row["original_audio_path"] == "sample.wav"
        assert "audio" not in row
        assert read_zip_audio(row["audio_path"]) == audio_bytes
        audio, rate = load_raw_audio(row)
        assert rate == 1000
        np.testing.assert_array_equal(audio, expected)
    assert manifest.exists()


def test_pack_custom_audio_field_preserves_parent_grouping(tmp_path):
    manifest, _, expected = make_dataset(tmp_path)
    manifest.write_text(json.dumps({
        "WavPath": "audio/a.wav#0.1:0.5",
        "DisplayTranscription": "hello", "parent_audio_path": "original-parent.wav",
        "seg_index": 2, "seg_start": 0.1,
    }) + "\n")
    output = tmp_path / "custom_zip.jsonl"
    pack_audio_zip(str(manifest), str(tmp_path / "custom.zip"), str(output), audio_field="WavPath")
    row = json.loads(output.read_text())
    assert row["parent_audio_path"] == "original-parent.wav"
    assert row["seg_index"] == 2
    assert row["seg_start"] == 0.1
    assert row["DisplayTranscription"] == "hello"
    audio, rate = load_raw_audio({"audio_path": row["WavPath"]})
    assert rate == 1000
    np.testing.assert_array_equal(audio, expected[100:500])

import gzip
import json

import pytest
import yaml

from recipe.phimm import cache_eval_audio as prefetch
from verl import audio_cache

REMOTE = "az://orngwus2cresco/data/Evaluation/"


def write_config(tmp_path, rows, repeated=False):
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    entry = {"dataset_name": "jsonl", "jsonl_paths": str(manifest)}
    config = tmp_path / "dataset.yaml"
    config.write_text(yaml.safe_dump([entry, entry] if repeated else entry))
    return config


def test_deduplicates_manifests_and_physical_files(tmp_path, monkeypatch):
    config = write_config(
        tmp_path,
        [
            {"audio_path": REMOTE + "a.wav#0:1", "parent_audio_path": REMOTE + "parent.wav"},
            {"audio_path": REMOTE + "a.wav#1:2"},
            {"audio_chunk": REMOTE + "chunk.audio:20:1#0%:50%"},
            {"url": REMOTE + "chunk.audio:20:2"},
        ],
        repeated=True,
    )
    opened = []
    original = prefetch.bf.BlobFile

    def tracked_open(path, *args, **kwargs):
        opened.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(prefetch.bf, "BlobFile", tracked_open)
    assert prefetch.collect_audio_files(str(config)) == [REMOTE + "a.wav", REMOTE + "chunk.audio"]
    assert opened.count(str(tmp_path / "samples.jsonl")) == 1


@pytest.mark.parametrize("path_kind", ["directory", "glob", "list"])
def test_supports_gzip_and_manifest_expansion(tmp_path, path_kind):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "a.jsonl").write_text(json.dumps({"audio_path": REMOTE + "a.wav"}) + "\n")
    with gzip.open(manifests / "b.jsonl.gz", "wt") as stream:
        stream.write(json.dumps({"audio_file": REMOTE + "b.wav"}) + "\n")
    paths = {
        "directory": str(manifests),
        "glob": str(manifests / "*.jsonl*"),
        "list": [str(manifests / "a.jsonl"), str(manifests / "b.jsonl.gz")],
    }
    config = tmp_path / "dataset.yaml"
    config.write_text(yaml.safe_dump({"dataset_name": "jsonl", "jsonl_paths": paths[path_kind]}))
    assert prefetch.collect_audio_files(str(config)) == [REMOTE + "a.wav", REMOTE + "b.wav"]


def test_prefetch_uses_loader_cache_and_resumes(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(audio_cache, "LOCAL_DATA_ROOT", cache_root)
    monkeypatch.setattr(prefetch, "LOCAL_DATA_ROOT", cache_root)
    monkeypatch.setattr(audio_cache, "LOCK_ROOT", tmp_path / "locks")
    local = tmp_path / "local.wav"
    local.write_bytes(b"local audio")
    rows = [{"audio_path": REMOTE + "a.wav#0:1"}, {"audio_path": REMOTE + "a.wav#1:2"}, {"audio_path": str(local)}]
    config = write_config(tmp_path, rows, repeated=True)
    original_config = config.read_bytes()
    original_manifest = (tmp_path / "samples.jsonl").read_bytes()
    copies = []

    def transfer(source, destination):
        copies.append(source)
        destination.write_bytes(b"unchanged audio")

    monkeypatch.setattr(audio_cache, "_run_bbb_transfer", transfer)
    summary = prefetch.cache_eval_audio(str(config), workers=2)
    assert summary == {"cache_root": str(cache_root), "unique_files": 2, "already_local": 1, "downloaded": 1}
    assert copies == [REMOTE + "a.wav"]
    assert (cache_root / "Evaluation/a.wav").read_bytes() == b"unchanged audio"
    assert audio_cache.resolve_audio_source(REMOTE + "a.wav#0:1") == str(cache_root / "Evaluation/a.wav") + "#0:1"
    assert prefetch.cache_eval_audio(str(config))["downloaded"] == 0
    assert len(copies) == 1
    assert config.read_bytes() == original_config
    assert (tmp_path / "samples.jsonl").read_bytes() == original_manifest


def test_failed_download_is_logged_and_raises(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(audio_cache, "LOCAL_DATA_ROOT", tmp_path / "cache")
    config = write_config(tmp_path, [{"audio_path": REMOTE + "missing.wav"}])

    def fail(source):
        raise audio_cache.RemoteAudioCacheError("transfer failed")

    monkeypatch.setattr(prefetch, "cache_audio_source", fail)
    with pytest.raises(RuntimeError, match="Failed to cache 1/1"):
        prefetch.cache_eval_audio(str(config))
    assert "missing.wav" in caplog.text
    assert "transfer failed" in caplog.text


def test_missing_download_output_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_cache, "LOCAL_DATA_ROOT", tmp_path / "cache")
    config = write_config(tmp_path, [{"audio_path": REMOTE + "missing.wav"}])
    monkeypatch.setattr(prefetch, "cache_audio_source", lambda source: str(tmp_path / "missing.wav"))
    with pytest.raises(RuntimeError, match="Failed to cache 1/1"):
        prefetch.cache_eval_audio(str(config))


@pytest.mark.parametrize(
    "row,error",
    [
        ({}, ValueError),
        ({"audio_path": 42}, ValueError),
        ({"audio_path": "az://other/data/a.wav"}, ValueError),
        ({"audio_path": "/nonexistent/verl-cache-test.wav"}, FileNotFoundError),
    ],
)
def test_invalid_audio_references_fail(tmp_path, row, error):
    config = write_config(tmp_path, [row])
    with pytest.raises(error):
        prefetch.collect_audio_files(str(config))


def test_custom_audio_fields(tmp_path):
    config = write_config(tmp_path, [{"recording": REMOTE + "a.wav", "url": "not-an-audio-file"}])
    assert prefetch.collect_audio_files(str(config), fields=("recording",)) == [REMOTE + "a.wav"]


@pytest.mark.parametrize("content", ["[]", "null", "dataset_name: parquet", "dataset_name: jsonl\njsonl_paths: []"])
def test_invalid_dataset_yaml_fails(tmp_path, content):
    config = tmp_path / "invalid.yaml"
    config.write_text(content)
    with pytest.raises(ValueError):
        prefetch.collect_audio_files(str(config))


def test_malformed_json_has_manifest_line_context(tmp_path):
    config = write_config(tmp_path, [])
    (tmp_path / "samples.jsonl").write_text("\nnot json\n")
    with pytest.raises(ValueError, match="samples.jsonl:2"):
        prefetch.collect_audio_files(str(config))


def test_empty_manifest_fails(tmp_path):
    config = write_config(tmp_path, [])
    with pytest.raises(ValueError, match="no audio references"):
        prefetch.collect_audio_files(str(config))


def test_missing_manifest_fails(tmp_path):
    config = write_config(tmp_path, [])
    (tmp_path / "samples.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="No JSONL files matched"):
        prefetch.collect_audio_files(str(config))


def test_worker_count_must_be_positive():
    with pytest.raises(ValueError, match="workers must be positive"):
        prefetch.cache_eval_audio("unused.yaml", workers=0)


def test_cli_prints_summary(tmp_path, monkeypatch, capsys):
    local = tmp_path / "local.wav"
    local.touch()
    config = write_config(tmp_path, [{"audio_path": str(local)}])
    monkeypatch.setattr("sys.argv", ["cache_eval_audio", str(config), "--workers", "2"])
    prefetch.main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["unique_files"] == 1
    assert summary["already_local"] == 1
    assert summary["downloaded"] == 0

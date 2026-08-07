from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from verl import audio_cache


REMOTE_ROOT = "az://orngwus2cresco/data/"


def test_copy_remote_wav_syncs_parent_folder(tmp_path, monkeypatch):
    local_path = tmp_path / "Evaluation" / "sample.wav"
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        local_path.write_bytes(b"audio")

    monkeypatch.setattr(audio_cache.subprocess, "run", fake_run)

    audio_cache._copy_remote_file(f"{REMOTE_ROOT}Evaluation/sample.wav", local_path)

    assert commands == [
        (
            ["bbb", "sync", f"{REMOTE_ROOT}Evaluation/", f"{local_path.parent}/"],
            {
                "check": True,
                "capture_output": True,
                "text": True,
                "timeout": audio_cache.BLOB_READ_TIMEOUT_SECONDS,
            },
        )
    ]


def test_copy_remote_non_wav_copies_atomically(tmp_path, monkeypatch):
    local_path = tmp_path / "speech" / "chunk.audio"
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"audio")

    monkeypatch.setattr(audio_cache.subprocess, "run", fake_run)

    audio_cache._copy_remote_file(f"{REMOTE_ROOT}speech/chunk.audio", local_path)

    assert commands[0][:3] == ["bbb", "cp", f"{REMOTE_ROOT}speech/chunk.audio"]
    assert commands[0][-1].endswith(".tmp")
    assert local_path.read_bytes() == b"audio"


def test_local_audio_source_preserves_relative_path_and_chunk_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_cache, "LOCAL_DATA_ROOT", tmp_path)

    result = audio_cache.local_audio_source(f"{REMOTE_ROOT}speech/set/chunk.audio:20:4")

    assert result == f"{tmp_path}/speech/set/chunk.audio:20:4"


def test_resolve_audio_source_prefers_existing_local_file(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_cache, "LOCAL_DATA_ROOT", tmp_path)
    local_file = tmp_path / "Evaluation" / "sample.wav"
    local_file.parent.mkdir(parents=True)
    local_file.touch()

    result = audio_cache.resolve_audio_source(f"{REMOTE_ROOT}Evaluation/sample.wav")

    assert result == str(local_file)


def test_localize_audio_source_does_not_cache_non_orange_remote(monkeypatch):
    def unexpected_cache(remote_path: str, local_path: Path):
        raise AssertionError(f"unexpected cache from {remote_path} to {local_path}")

    monkeypatch.setattr(audio_cache, "_ensure_cached_remote_file", unexpected_cache)
    source = "az://other-container/audio/sample.wav#1.5:3.0"

    assert audio_cache.localize_audio_source(source) == source


def test_localize_audio_source_caches_missing_orange_file(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_cache, "LOCAL_DATA_ROOT", tmp_path)
    copied = []

    def fake_copy(remote_path: str, local_path: Path):
        copied.append((remote_path, local_path))
        local_path.parent.mkdir(parents=True)
        local_path.write_bytes(b"audio")

    monkeypatch.setattr(audio_cache, "_copy_remote_file", fake_copy)
    source = f"{REMOTE_ROOT}speech/chunk.audio:20:4"

    result = audio_cache.localize_audio_source(source)

    assert result == f"{tmp_path}/speech/chunk.audio:20:4"
    assert copied == [(f"{REMOTE_ROOT}speech/chunk.audio", tmp_path / "speech" / "chunk.audio")]


def test_cache_audio_source_copies_file_and_rewrites_time_range(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_cache, "LOCAL_DATA_ROOT", tmp_path)
    copied = []

    def fake_copy(remote_path, local_path):
        copied.append((remote_path, local_path))
        local_path.parent.mkdir(parents=True)
        local_path.write_bytes(b"audio")

    monkeypatch.setattr(audio_cache, "_copy_remote_file", fake_copy)
    source = f"{REMOTE_ROOT}Evaluation/sample.wav#1.5:3.0"

    result = audio_cache.cache_audio_source(source)

    assert result == f"{tmp_path}/Evaluation/sample.wav#1.5:3.0"
    assert copied == [(f"{REMOTE_ROOT}Evaluation/sample.wav", tmp_path / "Evaluation" / "sample.wav")]


def test_cache_audio_source_does_not_copy_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_cache, "LOCAL_DATA_ROOT", tmp_path)
    local_file = tmp_path / "speech" / "sample.wav"
    local_file.parent.mkdir(parents=True)
    local_file.touch()

    def unexpected_copy(remote_path: str, local_path: Path):
        raise AssertionError(f"unexpected copy from {remote_path} to {local_path}")

    monkeypatch.setattr(audio_cache, "_copy_remote_file", unexpected_copy)

    assert audio_cache.cache_audio_source(f"{REMOTE_ROOT}speech/sample.wav") == str(local_file)


def test_cache_audio_source_copies_shared_chunk_file_once(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_cache, "LOCAL_DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(audio_cache, "LOCK_ROOT", tmp_path / "locks")
    copy_count = 0

    def fake_copy(remote_path, local_path):
        nonlocal copy_count
        copy_count += 1
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"audio")

    monkeypatch.setattr(audio_cache, "_copy_remote_file", fake_copy)
    sources = [f"{REMOTE_ROOT}speech/chunk.audio:8:{index}" for index in range(8)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(audio_cache.cache_audio_source, sources))

    assert copy_count == 1
    assert results == [f"{tmp_path}/data/speech/chunk.audio:8:{index}" for index in range(8)]
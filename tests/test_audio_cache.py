from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from verl import audio_cache


REMOTE_ROOT = "az://orngwus2cresco/data/"


def test_copy_remote_wav_copies_single_file_atomically(tmp_path, monkeypatch):
    local_path = tmp_path / "Evaluation" / "sample.wav"
    commands = []

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            commands.append((command, kwargs))
            self.command = command

        def communicate(self, timeout):
            assert timeout == audio_cache.BLOB_READ_TIMEOUT_SECONDS
            Path(self.command[-1]).write_bytes(b"audio")
            return "", ""

    monkeypatch.setattr(audio_cache.subprocess, "Popen", FakeProcess)

    audio_cache._copy_remote_file(f"{REMOTE_ROOT}Evaluation/sample.wav", local_path)

    assert commands[0][0][:3] == ["bbb", "cp", f"{REMOTE_ROOT}Evaluation/sample.wav"]
    assert commands[0][0][-1].endswith(".tmp")
    assert commands[0][1] == {
        "stdout": audio_cache.subprocess.PIPE,
        "stderr": audio_cache.subprocess.PIPE,
        "text": True,
        "start_new_session": True,
    }
    assert local_path.read_bytes() == b"audio"


def test_copy_remote_non_wav_copies_atomically(tmp_path, monkeypatch):
    local_path = tmp_path / "speech" / "chunk.audio"
    commands = []

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            commands.append(command)
            self.command = command

        def communicate(self, timeout):
            Path(self.command[-1]).write_bytes(b"audio")
            return "", ""

    monkeypatch.setattr(audio_cache.subprocess, "Popen", FakeProcess)

    audio_cache._copy_remote_file(f"{REMOTE_ROOT}speech/chunk.audio", local_path)

    assert commands[0][:3] == ["bbb", "cp", f"{REMOTE_ROOT}speech/chunk.audio"]
    assert commands[0][-1].endswith(".tmp")
    assert local_path.read_bytes() == b"audio"


def test_bbb_timeout_kills_process_group_before_retry(tmp_path, monkeypatch):
    local_path = tmp_path / "sample.wav"
    attempts = []
    killed = []

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            self.pid = 100 + len(attempts)
            self.command = command
            self.timed_out = not attempts
            attempts.append(self)

        def communicate(self, timeout=None):
            if self.timed_out and timeout is not None:
                self.timed_out = False
                raise audio_cache.subprocess.TimeoutExpired(self.command, timeout)
            if len(attempts) == 2:
                local_path.write_bytes(b"audio")
            return "", ""

    monkeypatch.setattr(audio_cache.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(audio_cache.os, "killpg", lambda pid, sig: killed.append((pid, sig)))

    audio_cache._run_bbb_transfer("az://orngwus2cresco/data/sample.wav", local_path)

    assert len(attempts) == 2
    assert killed == [(100, audio_cache.signal.SIGKILL)]
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


def test_submit_audio_cache_starts_daemon_and_reuses_queue(monkeypatch):
    queued = []
    processes = []

    class FakeQueue:
        def put_nowait(self, item):
            queued.append(item)

    class FakeProcess:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = False
            processes.append(self)

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started

    monkeypatch.setattr(audio_cache.multiprocessing, "Queue", FakeQueue)
    monkeypatch.setattr(audio_cache.multiprocessing, "Process", FakeProcess)
    monkeypatch.setattr(audio_cache, "_CACHE_SERVER_PROCESS", None)
    monkeypatch.setattr(audio_cache, "_CACHE_SERVER_QUEUE", None)
    monkeypatch.setattr(audio_cache, "_CACHE_SERVER_OWNER_PID", None)

    audio_cache.submit_audio_cache(["a.wav", "a.wav", "", "b.wav"], max_workers=7)
    audio_cache.submit_audio_cache(["c.wav"], max_workers=3)

    assert len(processes) == 1
    assert processes[0].started
    assert processes[0].kwargs["target"] is audio_cache._audio_cache_server
    assert processes[0].kwargs["name"] == "verl-audio-cache"
    assert processes[0].kwargs["daemon"] is True
    assert queued == [("sources", ("a.wav", "b.wav"), 7), ("sources", ("c.wav",), 3)]


def test_submit_audio_cache_dataset_does_not_read_columns(monkeypatch):
    queued = []

    class NonReadableDataset:
        column_names = ["audio_path", "text"]

        def __getitem__(self, key):
            raise AssertionError(f"caller read dataset column {key}")

    monkeypatch.setattr(audio_cache, "_submit_audio_cache_request", queued.append)
    ds = NonReadableDataset()

    audio_cache.submit_audio_cache_dataset(ds, ["audio_path", "audio_chunk"], max_workers=5)

    assert queued == [("dataset", (ds, ("audio_path",)), 5)]
"""Node-local shared cache correctness, including independent processes."""
import hashlib
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from recipe.phimm.utils.remote_cache import shared_remote_cache
from recipe.phimm.utils import shared


def test_shared_file_and_version(tmp_path):
    calls = []

    def populate(dest):
        calls.append(dest)
        Path(dest).write_bytes(b"model")

    first = shared_remote_cache("az://account/container/model.pt", tmp_path, "v1", populate)
    again = shared_remote_cache("az://account/container/model.pt", tmp_path, "v1", populate)
    second = shared_remote_cache("az://account/container/model.pt", tmp_path, "v2", populate)
    assert first == again and first != second
    assert Path(first).name == "model.pt"
    assert Path(first).read_bytes() == b"model"
    assert len(calls) == 2


def test_directory_and_failed_download_retry(tmp_path):
    def fail(dest):
        Path(dest).mkdir()
        (Path(dest) / "partial").write_bytes(b"bad")
        raise OSError("network failed")

    with pytest.raises(OSError, match="network failed"):
        shared_remote_cache("az://a/c/model", tmp_path, "v1", fail)
    assert not list(tmp_path.rglob("*.staging-*"))
    assert not list(tmp_path.rglob("complete.json"))

    def populate(dest):
        Path(dest).mkdir()
        (Path(dest) / "config.json").write_text("{}")

    result = shared_remote_cache("az://a/c/model", tmp_path, "v1", populate)
    assert (Path(result) / "config.json").read_text() == "{}"
    assert sorted(p.name for p in Path(result).iterdir()) == ["config.json"]


def test_missing_payload_and_corrupt_entry_fail_closed(tmp_path):
    with pytest.raises(RuntimeError, match="without a payload"):
        shared_remote_cache("az://a/c/model", tmp_path, "v1", lambda dest: None)
    assert not list(tmp_path.rglob("complete.json"))
    result = shared_remote_cache("az://a/c/model", tmp_path, "v1", lambda dest: Path(dest).touch())
    Path(result).unlink()
    with pytest.raises(RuntimeError, match="requires review"):
        shared_remote_cache("az://a/c/model", tmp_path, "v1", lambda dest: pytest.fail("must not overwrite"))


def test_full_url_is_part_of_identity(tmp_path):
    def populate(dest):
        Path(dest).touch()

    a = shared_remote_cache("az://a/c/model", tmp_path, "v1", populate)
    b = shared_remote_cache("az://b/c/model", tmp_path, "v1", populate)
    assert a != b


CHILD = '''
from pathlib import Path
import hashlib,sys,time
from recipe.phimm.utils.remote_cache import shared_remote_cache
root=Path(sys.argv[1])
def populate(dest):
    with (root/'downloads').open('a') as f: f.write('download\\n')
    time.sleep(0.1)
    Path(dest).write_bytes(b'x' * 1024 * 1024)
p=shared_remote_cache('az://a/c/model.bin',root/'cache','v1',populate)
print(p)
'''


def test_eight_processes_and_restart_download_once(tmp_path):
    def run(_):
        return subprocess.check_output([sys.executable, "-c", CHILD, str(tmp_path)], text=True).strip()

    with ThreadPoolExecutor(max_workers=8) as executor:
        paths = list(executor.map(run, range(8)))
    paths.append(run(0))
    assert len(set(paths)) == 1
    assert (tmp_path / "downloads").read_text().splitlines() == ["download"]
    assert hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest() == hashlib.sha256(b"x" * 1024 * 1024).hexdigest()
    assert len(list((tmp_path / "cache").rglob("model.bin"))) == 1


def test_killed_downloader_never_publishes_partial(tmp_path):
    child = '''
from pathlib import Path
import sys,time
from recipe.phimm.utils.remote_cache import shared_remote_cache
root=Path(sys.argv[1])
def populate(dest):
    Path(dest).write_bytes(b'partial')
    (root/'started').touch()
    time.sleep(60)
shared_remote_cache('az://a/c/model',root/'cache','v1',populate)
'''
    proc = subprocess.Popen([sys.executable, "-c", child, str(tmp_path)])
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "started").exists():
            assert proc.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.02)
        proc.kill()
        proc.wait(timeout=5)
        assert not list(tmp_path.rglob("complete.json"))
        result = shared_remote_cache(
            "az://a/c/model", tmp_path / "cache", "v1", lambda dest: Path(dest).write_bytes(b"complete")
        )
        assert Path(result).read_bytes() == b"complete"
        # SIGKILL orphan stays isolated; do not delete files belonging to another process.
        assert len(list((tmp_path / "cache").glob("*.staging-*"))) == 0
        assert len(list((tmp_path / "cache" / "shared-v1").glob("*.staging-*"))) == 1
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


@pytest.fixture
def blob(monkeypatch):
    calls = []
    shared._legacy_cache_remote_path.cache_clear()
    monkeypatch.setattr(shared.bf, "exists", lambda src: True)
    monkeypatch.setattr(shared.bf, "isdir", lambda src: False)

    def copy(src, dest, overwrite):
        calls.append((src, dest))
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"weights")

    monkeypatch.setattr(shared.bf, "copy", copy)
    yield calls
    shared._legacy_cache_remote_path.cache_clear()


def test_wrapper_namespace_changes_in_same_process(tmp_path, monkeypatch, blob):
    monkeypatch.setenv("VERL_REMOTE_CACHE_NAMESPACE", "v1")
    first = shared.cache_remote_path("az://a/c/model.pt", cache_dir=str(tmp_path))
    monkeypatch.setenv("VERL_REMOTE_CACHE_NAMESPACE", "v2")
    second = shared.cache_remote_path("az://a/c/model.pt", cache_dir=str(tmp_path))
    assert first != second and len(blob) == 2
    monkeypatch.setenv("VERL_REMOTE_CACHE_NAMESPACE", "v1")
    # A disk hit must work even when the remote service is unavailable.
    monkeypatch.setattr(shared.bf, "exists", lambda src: pytest.fail("unexpected remote lookup"))
    assert shared.cache_remote_path("az://a/c/model.pt", cache_dir=str(tmp_path)) == first


def test_wrapper_legacy_explicit_and_local(tmp_path, monkeypatch, blob):
    monkeypatch.delenv("VERL_REMOTE_CACHE_NAMESPACE", raising=False)
    first = shared.cache_remote_path("az://a/c/model.pt", cache_dir=str(tmp_path))
    assert shared.cache_remote_path("az://a/c/model.pt", cache_dir=str(tmp_path)) == first
    assert "shared-v1" not in first and len(blob) == 1
    monkeypatch.setenv("VERL_REMOTE_CACHE_NAMESPACE", "v1")
    explicit = str(tmp_path / "explicit.pt")
    assert shared.cache_remote_path("az://a/c/model.pt", local_path=explicit) == explicit
    assert shared.cache_remote_path("/existing/model") == "/existing/model"


def test_wrapper_missing_remote_can_retry(tmp_path, monkeypatch, blob):
    monkeypatch.setenv("VERL_REMOTE_CACHE_NAMESPACE", "v1")
    monkeypatch.setattr(shared.bf, "exists", lambda src: False)
    with pytest.raises(FileNotFoundError):
        shared.cache_remote_path("az://a/c/model.pt", cache_dir=str(tmp_path))
    monkeypatch.setattr(shared.bf, "exists", lambda src: True)
    result = shared.cache_remote_path("az://a/c/model.pt", cache_dir=str(tmp_path))
    assert Path(result).read_bytes() == b"weights"

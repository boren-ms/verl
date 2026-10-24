"""Keep optional packages optional and cover the affected import ordering."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def compat():
    path = Path(__file__).parents[1] / "src/hf_qwen35_audio/runtime_compat.py"
    spec = importlib.util.spec_from_file_location("audio_runtime_compat_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_affected_runtime_imports_flashinfer(compat, monkeypatch):
    versions = {"vllm": "0.24.0", "flashinfer-python": "0.6.12",
                "tilelang": "0.1.9", "apache-tvm-ffi": "0.1.9"}
    imported = []
    monkeypatch.setattr(compat, "version", versions.__getitem__)
    monkeypatch.setattr(compat, "import_module", imported.append)
    assert compat.prepare_optional_flashinfer()
    assert imported == ["flashinfer"]


def test_hf_only_runtime_needs_no_flashinfer(compat, monkeypatch):
    def missing(package):
        raise compat.PackageNotFoundError(package)

    monkeypatch.setattr(compat, "version", missing)
    monkeypatch.setattr(compat, "import_module", lambda name: pytest.fail(name))
    assert not compat.prepare_optional_flashinfer()


def test_other_versions_do_not_apply_workaround(compat, monkeypatch):
    monkeypatch.setattr(compat, "version", lambda name: "99.0")
    monkeypatch.setattr(compat, "import_module", lambda name: pytest.fail(name))
    assert not compat.prepare_optional_flashinfer()

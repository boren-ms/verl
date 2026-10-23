import hashlib
import json

import pytest

from recipe.phimm.prepare_model_directory import prepare_model_directory


@pytest.fixture
def model(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"test"}')
    (source / "tokenizer.json").write_text('{}')
    (source / "preprocessor_config.json").write_text('{}')
    (source / "model-01.safetensors").write_bytes(b"weights")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-01.safetensors"}})
    )
    archive = source / "baseline_split_example"
    archive.mkdir()
    (archive / "baseline.z001").write_bytes(b"archive" * 100)
    return source


def test_compact_preserves_payload_and_source(model, tmp_path):
    before = {p.relative_to(model).as_posix(): p.read_bytes() for p in model.rglob('*') if p.is_file()}
    destination = tmp_path / "compact"
    result = prepare_model_directory(model, destination, ["baseline_split_example"])
    assert not (destination / "baseline_split_example").exists()
    assert result["total_bytes"] == sum(len(data) for name, data in before.items() if '/' not in name)
    for item in result["files"]:
        assert (destination / item["path"]).read_bytes() == before[item["path"]]
        assert hashlib.sha256((destination / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert before == {p.relative_to(model).as_posix(): p.read_bytes() for p in model.rglob('*') if p.is_file()}
    assert json.loads((destination / "verl-model-manifest.json").read_text()) == result
    with pytest.raises(FileExistsError):
        prepare_model_directory(model, destination, ["baseline_split_example"])


def test_required_shard_cannot_be_excluded(model, tmp_path):
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "baseline_split_example/required.bin"}})
    )
    with pytest.raises(ValueError, match="Required weight shard"):
        prepare_model_directory(model, tmp_path / "compact", ["baseline_split_example"])
    assert not (tmp_path / "compact").exists()


def test_referenced_asset_cannot_be_excluded(model, tmp_path):
    (model / "config.json").write_text('{"asset":"./baseline_split_example/required.bin"}')
    with pytest.raises(ValueError, match="JSON references excluded"):
        prepare_model_directory(model, tmp_path / "compact", ["baseline_split_example"])


def test_destination_and_symlink_guards(model, tmp_path):
    with pytest.raises(ValueError, match="outside"):
        prepare_model_directory(model, model / "compact", ["baseline_split_example"])
    (model / "linked").symlink_to(model / "config.json")
    with pytest.raises(ValueError, match="Symlinks"):
        prepare_model_directory(model, tmp_path / "compact", ["baseline_split_example"])


def test_explicit_existing_directory_required(model, tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        prepare_model_directory(model, tmp_path / "compact", ["../source"])
    with pytest.raises(ValueError, match="existing source"):
        prepare_model_directory(model, tmp_path / "compact", ["missing"])

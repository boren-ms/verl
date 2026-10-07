import json
from pathlib import Path

import pytest

from scripts.list_training_chunk_paths import (
    chunk_file_path,
    collect_chunk_paths,
    find_missing_paths,
    load_training_sources,
    write_outputs,
)


def test_recipe_resolves_training_manifest_only():
    config = Path(__file__).parents[2] / (
        "recipe/phimm/config/v2609_entity/remax_2609r2_name_en13m_s1k_bs128_n4_r256_g32.yaml"
    )
    sources = load_training_sources(config)
    assert len(sources) == 1
    assert sources[0]["jsonl_paths"].endswith("all_gpt_tagged_person.train.jsonl")
    assert sources[0]["pre_process"]["path_map"]["src_part"] == "am_data/"


def test_selecting_validation_manifest(tmp_path):
    config = tmp_path / "recipe.yaml"
    config.write_text(
        "data:\n"
        "  train_data:\n"
        "    - dataset_name: jsonl\n"
        "      jsonl_paths: train.jsonl\n"
        "  val_data:\n"
        "    - dataset_name: jsonl\n"
        "      jsonl_paths: val.jsonl\n",
        encoding="utf-8",
    )
    assert load_training_sources(config)[0]["jsonl_paths"] == "train.jsonl"
    assert load_training_sources(config, "val")[0]["jsonl_paths"] == "val.jsonl"
    with pytest.raises(ValueError, match="Unsupported data split"):
        load_training_sources(config, "test")


def test_validation_requires_nonempty_config(tmp_path):
    config = tmp_path / "recipe.yaml"
    config.write_text("data:\n  val_data: null\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"data.val_data"):
        load_training_sources(config, "val")


def test_mapping_and_sample_suffix_removal():
    path_map = {"src_part": "am_data/", "dst_part": "az://orngwus2cresco/data/speech/am_data/"}
    assert chunk_file_path({"audio_path": "am_data/en/ChunkFiles/chunk.audio:98:13"}, path_map) == (
        "az://orngwus2cresco/data/speech/am_data/en/ChunkFiles/chunk.audio"
    )
    assert chunk_file_path({"audio_chunk": "az://account/container/chunk.audio:2:1"}, {}) == (
        "az://account/container/chunk.audio"
    )


@pytest.mark.parametrize(
    "reference", ["chunk.audio:2:2", "chunk.audio:0:0", "chunk.audio:x:0", "chunk.audios:2:0", "sample.wav", None]
)
def test_invalid_reference_is_explicit_error(reference):
    with pytest.raises(ValueError):
        chunk_file_path({"audio_path": reference}, {})


def test_collect_deduplicates_and_reports_missing(tmp_path):
    existing = tmp_path / "existing.audio"
    existing.touch()
    missing = tmp_path / "missing.audio"
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"audio_path": reference})
            for reference in [f"{missing}:2:0", f"{existing}:2:0", f"{existing}:2:1"]
        ),
        encoding="utf-8",
    )
    paths, rows, manifests = collect_chunk_paths([{"jsonl_paths": str(manifest)}])
    assert paths == sorted([str(existing), str(missing)])
    assert rows == 3
    assert manifests == [{"path": str(manifest), "rows": 3}]
    absent = find_missing_paths(paths, workers=2)
    assert absent == [str(missing)]
    write_outputs(tmp_path / "out", paths, absent, {"training_rows": rows})
    assert (tmp_path / "out/chunk_paths.txt").read_text().splitlines() == paths
    assert (tmp_path / "out/missing_chunk_paths.txt").read_text().splitlines() == absent
    assert json.loads((tmp_path / "out/summary.json").read_text()) == {"training_rows": 3}


def test_storage_failure_is_not_treated_as_missing(monkeypatch):
    def fail(path):
        raise PermissionError(path)

    monkeypatch.setattr("scripts.list_training_chunk_paths.bf.exists", fail)
    with pytest.raises(PermissionError):
        find_missing_paths(["az://account/container/chunk.audio"], workers=1)


def test_bad_manifest_reports_line_number(tmp_path):
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text('{"audio_path": "sample.wav"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.jsonl:1:"):
        collect_chunk_paths([{"jsonl_paths": str(manifest)}])

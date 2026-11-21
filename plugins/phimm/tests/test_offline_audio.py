"""Offline preparation produces standalone training inputs without changing validation."""

import json

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from recipe.phimm.agentic.dataset import ASRDataset
from recipe.phimm.data.dataset import augment, jsonl_dataset
from recipe.phimm.tools.offline_audio import materialize
from recipe.phimm.tools.random_merge_list import random_merge
from recipe.phimm.tools.random_segment import random_segment


@pytest.mark.parametrize("mode", ["random_segment", "random_merge"])
def test_offline_snapshot_reloads_and_preserves_task_order(tmp_path, mode):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "".join(
            json.dumps({"audio_path": f"{i}.wav", "display": f"hello <NE>Name{i}</NE>", "audio_length": 1}) + "\n"
            for i in range(6)
        )
    )
    source = {
        "dataset_name": "jsonl",
        "jsonl_paths": str(raw),
        "pre_process": {"rename_fields": {"mappings": {"text": "display"}}, "clean_tagged_text": {}},
        "add_task_info": {"style": "verb", "language": "English"},
        "post_process": {"input_egs_limit": 4, "verl_format": {"prompt_key": "prompt"}},
    }
    config = OmegaConf.create(
        {
            "data": {
                "train_files": "train",
                "val_files": "val",
                "train_data": [[source]],
                "val_data": [[source]],
            },
            "trainer": {"val_only": False, "total_training_steps": 800},
            "asr": {"context_aware": True},
            "offline_data": {mode: {"seed": 7}},
        }
    )
    before = OmegaConf.to_container(config, resolve=True)
    out = tmp_path / "prepared"
    snapshot = materialize(config, out)
    assert OmegaConf.to_container(config, resolve=True) == before
    assert "offline_data" not in snapshot
    with initialize_config_dir(config_dir=str(out), version_base=None):
        loaded = compose(config_name="training")
    assert loaded == snapshot
    assert loaded.data.val_data == config.data.val_data
    assert loaded.trainer == config.trainer
    assert loaded.asr == config.asr
    assert "pre_process" not in loaded.data.train_data[0]
    assert "input_egs_limit" not in loaded.data.train_data[0].post_process
    rows = [json.loads(line) for line in (out / "source_000.jsonl").read_text().splitlines()]
    assert all("prompt" not in row and "gt_output" not in row for row in rows)
    assert all("<NE>" not in row["text"] for row in rows)
    assert all(isinstance(row["audio_path"], list) for row in rows)
    train = ASRDataset("train", None, loaded.data)
    assert sum(len(row["asr_segments"]) for row in train) >= 4
    assert len(ASRDataset("val", None, loaded.data)) == 4
    assert len(rows) == (4 if mode == "random_segment" else 1)
    if mode == "random_segment":
        legacy = random_segment(augment(jsonl_dataset(str(raw)), **source), seed=7)
        assert train.ds.to_list() == legacy.to_list()
    if mode == "random_merge":
        assert train[0]["reward_model"]["ground_truth"] == rows[0]["text"]
        assert len(train[0]["extra_info"]["keywords"]) == 6
    with pytest.raises(FileExistsError):
        materialize(config, out)


@pytest.mark.parametrize("preserve_keywords", [False, True])
def test_offline_merge_preserves_parent_limit_and_keyword_policy(tmp_path, preserve_keywords):
    from recipe.phimm.data.dataset import process_ds

    path = tmp_path / "raw.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "id": str(index),
                    "audio_path": f"{index}.wav",
                    "audio_length": 1,
                    "text": f"Hello <NE>Name{index}</NE>",
                    "keywords": ["producer"],
                }
            )
            + "\n"
            for index in range(6)
        )
    )
    source = {
        "dataset_name": "jsonl",
        "jsonl_paths": str(path),
        "pre_process": {"clean_tagged_text": {"preserve_existing_keywords": preserve_keywords}},
        "add_task_info": {"style": "verb", "language": "English"},
        "post_process": {"input_egs_limit": 2, "verl_format": {"prompt_key": "prompt"}},
    }
    options = {"merge_count": [2, 2], "seed": 42}
    legacy = augment(
        random_merge(process_ds(jsonl_dataset(str(path)), **source["pre_process"]), **options),
        add_task_info=source["add_task_info"],
        post_process=source["post_process"],
    )
    config = OmegaConf.create({"data": {"train_data": [source]}, "offline_data": {"random_merge": options}})
    snapshot = materialize(config, tmp_path / "prepared")
    prepared = OmegaConf.to_container(snapshot.data.train_data[0], resolve=True)
    actual = augment(jsonl_dataset(prepared["jsonl_paths"]), **prepared)
    assert len(actual) == 2
    assert actual.to_list() == legacy.to_list()
    assert all(("producer" in row["extra_info"]["keywords"]) == preserve_keywords for row in actual)


@pytest.mark.parametrize("mode", ["random_segment", "random_merge"])
@pytest.mark.parametrize("stage", ["pre_process", "post_process"])
def test_generation_is_not_available_as_a_training_transform(mode, stage):
    from datasets import Dataset
    from recipe.phimm.data.dataset import augment

    with pytest.raises(ValueError, match="Unsupported Base3a data transforms"):
        augment(Dataset.from_list([{"audio_path": "a.wav"}]), **{stage: {mode: {}}})

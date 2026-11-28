# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Exercise static JSONL through prompt construction and reward formatting."""

import json

import pytest

from recipe.phimm.data.dataset import augment, jsonl_dataset
from recipe.phimm.data.prompts import BIAS_INSTRUCTION
from recipe.phimm.data.simulate_biasing import dataset_block, suite_dataset_config


@pytest.mark.parametrize("position", [None, "before_audio", "after_audio"])
def test_static_dataset_block_preserves_candidates_and_reward_keywords(tmp_path, position):
    row = {
        "id": "sample",
        "audio_path": "sample.flac",
        "language": "English",
        "text": "We use Seattle.",
        "biasing_words": ["Seattle", "London"],
        "keywords": ["Seattle"],
    }
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(row) + "\n")
    block = dataset_block(path, "sample_source", position=position, style="verb", lang_hint=True)
    ds = jsonl_dataset(block.pop("jsonl_paths"))
    block.pop("dataset_name")
    result = augment(ds, **block)[0]
    prompt = result["prompt"][0]["content"]
    assert result["reward_model"]["ground_truth"] == row["text"]
    assert result["reward_model"]["gt_output"].endswith(row["text"])
    assert result["extra_info"]["keywords"] == row["keywords"]
    assert result["extra_info"]["biasing_words"] == row["biasing_words"]
    assert result["data_source"] == "sample_source"
    assert prompt.count("<audio>") == 1
    if position is None:
        assert BIAS_INSTRUCTION not in prompt
    elif position == "before_audio":
        assert prompt.index(BIAS_INSTRUCTION) < prompt.index("<audio>")
    else:
        assert prompt.index(BIAS_INSTRUCTION) > prompt.index("<audio>")


def test_suite_replaces_old_mix_and_maps_all_validation_sources(tmp_path):
    validation = [(tmp_path / "clean_0.jsonl", "clean_0", 0), (tmp_path / "Gaming_100.jsonl", "Gaming_100", 100)]
    config = suite_dataset_config(
        tmp_path / "train.jsonl", validation, position="before_audio", style="verb", lang_hint=True, training_rows=12
    )
    assert config["data"]["mix"][0]["datasource"] == [{"name": "asr", "epoch": 1, "shuffle": True}]
    assert config["reward_map"] == {
        "asr": "train_librispeech",
        "clean_0": "val_openasr_en",
        "Gaming_100": "val_openasr_en",
    }
    assert config["data"]["val_data"][0]["add_task_info"]["biasing"] is None
    assert config["data"]["val_data"][1]["add_task_info"]["biasing"] == "before_audio"
    assert config["asr"]["expected_training_rows"] == 12
    assert config["data"]["max_prompt_length"] == 8192
    assert config["actor_rollout_ref"]["actor"]["ppo_max_token_len_per_gpu"] == "${data.max_model_length}"

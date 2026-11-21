"""Whole-utterance list packing preserves reference, metadata and ASR boundaries."""

import json
from pathlib import Path

import datasets
import numpy as np
import pytest
import soundfile as sf
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from recipe.phimm.agentic.dataset import ASRDataset, _prepare_data_confs
from recipe.phimm.data.dataset import augment, process_ds
from recipe.phimm.tools.offline_audio import materialize
from recipe.phimm.tools.random_merge_list import random_merge
from recipe.phimm.utils.audio import load_audio


def rows(count):
    return [
        {
            "audio_path": f"{i}.wav",
            "text": f"word{i}",
            "id": i,
            "language": "en" if i % 2 else "zh",
            "keywords": [str(i), "shared"],
            "audio_length": 1.0,
            "custom": {"value": i if i % 2 else None},
            "gt_output": "stale reference",
        }
        for i in range(count)
    ]


def test_exact_coverage_reproducibility_and_metadata_order():
    originals = rows(17)
    ds = datasets.Dataset.from_list(originals)
    merged = random_merge(ds, max_audio_length=4, buffer_size=5, seed=7)
    assert merged.to_list() == random_merge(ds, max_audio_length=4, buffer_size=5, seed=7).to_list()
    assert merged.to_list() != random_merge(ds, max_audio_length=4, buffer_size=5, seed=8).to_list()
    assert sorted(i for row in merged for i in row["random_merge_source_indices"]) == list(range(17))
    assert sum(merged["audio_length"]) == 17
    for row in merged:
        order = row["random_merge_source_indices"]
        assert row["audio_path"] == [f"{i}.wav" for i in order]
        assert row["text"] == " ".join(f"word{i}" for i in order)
        assert row["keywords"] == list(dict.fromkeys(word for i in order for word in originals[i]["keywords"]))
        languages = ["English" if i % 2 else "Chinese" for i in order]
        assert row["language"] == " ".join(dict.fromkeys(languages))
        assert row["source_languages"] == languages
        assert [c["language"] for c in row["components"]] == languages
        assert [c["text"] for c in row["components"]] == [originals[i]["text"] for i in order]
        assert row["source_audio_lengths"] == [1.0] * len(order)
        assert [json.loads(value) for value in row["custom"]] == [originals[i]["custom"] for i in order]
        assert [json.loads(value) for value in row["source_metadata_json"]] == [originals[i] for i in order]
        assert "gt_output" not in row
        assert row["random_merge_count"] == len(order)
        assert row["audio_length"] < 4
    assert any(len(row["audio_path"]) == 3 for row in merged)


def test_random_count_bounds_conserve_every_source():
    ds = datasets.Dataset.from_list(rows(100))
    result = random_merge(ds, merge_count=[2, 3], seed=17)
    assert result.to_list() == random_merge(ds, merge_count=[2, 3], seed=17).to_list()
    assert {row["random_merge_count"] for row in result} >= {2, 3}
    assert max(result["random_merge_count"]) == 3
    assert sorted(i for row in result for i in row["random_merge_source_indices"]) == list(range(100))
    # A duration cap can prevent a pair, but must never discard the source.
    limited = random_merge(ds, merge_count=[2, 3], max_audio_length=2)
    assert limited["random_merge_count"] == [1] * 100


@pytest.mark.parametrize("count", [[0, 3], [3, 2], [2], [2, True], [2, 3.0], "2,3"])
def test_invalid_merge_count(count):
    with pytest.raises(ValueError, match="merge_count"):
        random_merge(datasets.Dataset.from_list(rows(2)), merge_count=count)


def test_post_process_merge_rejected_before_task_formatting():
    with pytest.raises(ValueError, match="Unsupported Base3a data transforms"):
        augment(datasets.Dataset.from_list(rows(2)), post_process={"random_merge": {}})


def test_components_and_languages_are_merged_even_when_first_row_lacks_keywords():
    data = [
        dict(
            audio_path="a.wav",
            text="你好",
            audio_length=1,
            keywords=None,
            components=[dict(language="zh", text="你好")],
        ),
        dict(audio_path="b.wav", text="bonjour", audio_length=1, keywords=["Paris"], components=None),
    ]
    result = random_merge(datasets.Dataset.from_list(data), default_language="fr")
    assert len(result) == 1
    row = result[0]
    assert set(row["language"].split()) == {"Chinese", "French"}
    assert row["keywords"] == ["Paris"]
    assert set(c["language"] for c in row["components"]) == {"Chinese", "French"}
    assert row["text"] == " ".join(c["text"] for c in row["components"])


@pytest.mark.parametrize("audio_field", ["audio_path", "audio_file", "audio_chunk"])
def test_paths_stay_separate_and_audio_is_not_concatenated(tmp_path, monkeypatch, audio_field):
    data, waveforms = [], {}
    for i, size in enumerate([8000, 12000, 16000]):
        path = tmp_path / f"{i}.wav"
        wave = np.full(size, (i + 1) / 8, dtype=np.float32)
        sf.write(path, wave, 16000, subtype="FLOAT")
        waveforms[str(path)] = wave
        data.append({audio_field: str(path), "text": f"word{i}", "audio_length": size / 16000})
    with monkeypatch.context() as patch:
        patch.setattr(sf, "read", lambda *a, **k: pytest.fail("packing must not decode audio"))
        patch.setattr(sf, "info", lambda *a, **k: pytest.fail("duration already provided"))
        merged = random_merge(datasets.Dataset.from_list(data))
    row = merged[0]
    assert isinstance(row["audio_path"], list) and len(row["audio_path"]) == 3
    for path in row["audio_path"]:
        assert ";" not in path
        actual, sr = load_audio({"audio_path": path})
        assert sr == 16000
        np.testing.assert_array_equal(actual, waveforms[path])


def test_global_block_produces_full_reference_segments_and_leaves_validation_alone(tmp_path):
    originals = [{**row, "id": str(row["id"])} for row in rows(3)]
    path = tmp_path / "raw.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in originals))
    source = {
        "dataset_name": "jsonl",
        "jsonl_paths": str(path),
        "add_task_info": {"style": "verb", "language": "English", "lang_hint": True},
        "post_process": {"verl_format": {"extra_keys": ["random_merge_source_indices", "source_languages"]}},
    }
    config = OmegaConf.create(
        {
            "train_files": "train",
            "val_files": "val",
            "train_data": [source],
            "val_data": [source],
            "version": 2609,
        }
    )
    full_config = OmegaConf.create(
        {"data": OmegaConf.to_container(config), "offline_data": {"random_merge": {"max_audio_length": 10, "seed": 7}}}
    )
    config = materialize(full_config, tmp_path / "prepared").data
    train = ASRDataset("train", None, config)
    assert len(train) == 1
    sample = train[0]
    order = sample["extra_info"]["random_merge_source_indices"]
    assert sample["audio_path"] == [f"{i}.wav" for i in order]
    assert [segment["audio_path"] for segment in sample["asr_segments"]] == sample["audio_path"]
    assert len(sample["asr_segments"]) == 3
    assert sample["reward_model"]["ground_truth"] == " ".join(originals[i]["text"] for i in order)
    assert "stale reference" not in sample["reward_model"]["gt_output"]
    for i in order:
        assert f"word{i}" in sample["reward_model"]["gt_output"]
    assert set(sample["extra_info"]["language"].split()) == {"English", "Chinese"}
    assert "The languages are" in sample["prompt"][0]["content"]
    assert all(segment["asr_full_reference"] for segment in sample["asr_segments"])
    val = ASRDataset("val", None, config)
    assert len(val) == 3
    assert all(isinstance(val[i]["audio_path"], str) for i in range(len(val)))


def test_source_opt_out_and_conflicting_segmentation():
    config = OmegaConf.create(
        {
            "train_data": [{"pre_process": {"random_merge": {"enabled": False}}}, {}],
            "random_merge": {"max_audio_length": 10},
            "random_segment": {"enabled": False},
        }
    )
    with pytest.raises(ValueError, match="Materialize random segment/merge"):
        _prepare_data_confs(config, is_train=True)


def test_source_block_inherits_hint_language_before_formatting(tmp_path):
    data = [dict(audio_path=f"{i}.wav", text=f"mot{i}", audio_length=1) for i in range(2)]
    ds = datasets.Dataset.from_list(data)
    result = augment(
        random_merge(ds, default_language="French"),
        add_task_info={"style": "verb", "language": "French", "lang_hint": True},
        post_process={"verl_format": {"prompt_key": "prompt"}},
    )
    assert len(result) == 1
    assert result[0]["extra_info"]["language"] == "French"
    assert result[0]["reward_model"]["gt_output"].count("<src=French>") == 2


def test_empty_disabled_and_formatted_input():
    ds = datasets.Dataset.from_list(rows(1))
    assert random_merge(ds, enabled=False) is ds
    empty = random_merge(ds.select([]))
    assert len(empty) == 0
    assert hasattr(empty.features["audio_path"], "feature")
    with pytest.raises(ValueError, match="before add_task_info"):
        random_merge(datasets.Dataset.from_list([dict(audio_path="a.wav", reward_model={"ground_truth": "a"})]))


def test_hydra_random_merge_block_enables_context_and_disables_random_cuts():
    import recipe.phimm

    directory = Path(recipe.phimm.__file__).parent / "config/v2611_dev/earnings"
    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        config = compose(config_name="remax_earnings_short_baseline", overrides=["+asr_training=random_merge"])
    assert config.offline_data.random_merge.enabled
    assert not config.offline_data.random_segment.enabled
    assert config.asr.context_aware.enable
    assert config.data.max_audio_dur is None
    assert config.data.max_prompt_length == 2048
    assert config.asr.expected_training_rows is None


@pytest.mark.parametrize(
    "preset", ["remax_earnings_short_random_merge23_no_context", "remax_earnings_short_random_merge23_no_train_context"]
)
def test_no_context_preset_preserves_exact_data_and_other_controls(preset):
    import recipe.phimm

    directory = Path(recipe.phimm.__file__).parent / "config/v2611_dev/earnings"
    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        control = compose(config_name="remax_earnings_short_random_merge23_context")
        ablation = compose(config_name=preset)
    assert ablation.asr.context_aware.enable is False
    # Normalize only the intended flag and experiment-derived output paths.
    ablation.asr.context_aware.enable = control.asr.context_aware.enable
    ablation.trainer.experiment_name = control.trainer.experiment_name
    assert OmegaConf.to_container(ablation, resolve=True) == OmegaConf.to_container(control, resolve=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_merge_keeps_producer_keywords_alongside_cleaned_tags(enabled):
    ds = datasets.Dataset.from_list(
        [
            dict(audio_path="a.wav", text="hello <NE>Bob</NE>", keywords=["Alice"], audio_length=1),
            dict(audio_path="b.wav", text="visit <NE>Paris</NE>", keywords=["Bob"], audio_length=1),
        ]
    )
    result = augment(
        random_merge(process_ds(ds, clean_tagged_text={"preserve_existing_keywords": enabled}), enabled=enabled),
        add_task_info={"style": "verb", "language": "English"},
        post_process={"verl_format": {"prompt_key": "prompt"}},
    )
    if enabled:
        assert len(result) == 1
        assert set(result[0]["extra_info"]["keywords"]) == {"Alice", "Bob", "Paris"}
        assert "<NE>" not in result[0]["reward_model"]["ground_truth"]
    else:
        assert [row["extra_info"]["keywords"] for row in result] == [["Bob"], ["Paris"]]

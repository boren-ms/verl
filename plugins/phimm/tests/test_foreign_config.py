"""Check the native Base1 recipe and each effective train/validation data path."""

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from recipe.phimm.agentic.dataset import ASRDataset
from recipe.phimm.data.dataset import create_audio_dataset


def load_config(name="remax_foreign_base1_2611r2"):
    directory = Path(__file__).resolve().parents[1] / "src/recipe/phimm/config/v2611_dev/foreign"
    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        config = compose(config_name=name)
    OmegaConf.resolve(config)
    return config


def test_quartr_fl_recipe_uses_language_hints_and_foreign_validation():
    config = load_config("remax_foreign_quartr_fl")
    assert len(config.data.train_data) == 1
    source = config.data.train_data[0][0]
    assert source.jsonl_paths.endswith("/quartr_fl_segments.jsonl")
    assert source.add_task_info.language == "English"
    assert source.pre_process.clean_tagged_text == {}
    assert source.post_process.add_field.fields.data_source == "quartr_fl_segments"
    assert config.reward_map.quartr_fl_segments == "train_librispeech"
    mix = config.data.mix[0].datasource
    assert [(entry.name, entry.sample) for entry in mix] == [("quartr_fl_segments", 2732)]
    assert config.asr.expected_training_rows == 2732
    assert config.trainer.total_epochs == 2
    assert config.data.train_batch_size == config.actor_rollout_ref.actor.ppo_mini_batch_size == 32
    assert config.data.gen_batch_size == 32
    assert config.trainer.total_training_steps == config.asr.expected_training_steps == 170
    assert (2732 // config.data.train_batch_size) * config.trainer.total_epochs == 170
    assert [source.post_process.add_field.fields.data_source for group in config.data.val_data for source in group] == [
        "earnings22_parent4",
        "enhint_en",
        "enhint_noen",
    ]


def test_base1_preserves_mix_and_validation_routing():
    config = load_config()
    assert "reward_functions" not in config
    assert "val_reward" not in config
    assert [(entry.name, entry.sample) for entry in config.data.mix[0].datasource] == [
        ("foreign_1k", 6700),
        ("foreign_mixed_10k", 6100),
    ]
    assert config.data.train_batch_size == config.actor_rollout_ref.actor.ppo_mini_batch_size == 128
    assert config.trainer.total_epochs == 2
    assert config.trainer.total_training_steps == config.asr.expected_training_steps == 200
    assert config.asr.expected_training_rows == 12800
    assert config.actor_rollout_ref.model.path.endswith("/38000/qwen_hf/")
    assert config.actor_rollout_ref.model.lora_rank == 320
    assert config.actor_rollout_ref.actor.optim.lr == 6e-7
    assert config.actor_rollout_ref.rollout.n == 8
    assert "offline_data" not in config
    for source in ("foreign_1k", "foreign_mixed_10k"):
        assert config.reward_map[source] == "train_librispeech"
    assert config.reward_map.enhint_noen == "val_foreign_language"
    assert config.rewards.val_foreign_language.name == "foreign_language_eval"
    for group in config.data.val_data:
        for source in group:
            name = source.post_process.add_field.fields.data_source
            assert source.add_task_info.style == "verb"
            if name != "enhint_noen":
                assert config.reward_map[name] == "val_openasr_en"
    assert len(config.data.train_data) == 2
    assert [source.post_process.add_field.fields.data_source for group in config.data.val_data for source in group] == [
        "earnings22_parent4",
        "enhint_en",
        "enhint_noen",
    ]


@pytest.mark.parametrize(
    "name",
    [
        "remax_foreign_base1_2611r2",
        "remax_foreign_quartr_fl",
        "remax_foreign_quartr_fl_earnings",
        "remax_foreign_quartr_fl_earnings_ppo2",
    ],
)
def test_all_effective_data_configs_load_and_preserve_parent_metadata(tmp_path, name):
    config = load_config(name)
    for split in ("train_data", "val_data"):
        for group in config.data[split]:
            for source in group:
                kwargs = OmegaConf.to_container(source, resolve=True)
                masked = "mask_foreign_language" in kwargs
                row = {
                    "id": "example",
                    "audio_path": "audio.wav",
                    "text": "Bonjour" if masked else "Hello",
                    "display": "Hello",
                    "display_ne_tagged": "Hello",
                    "language": "fr" if masked else "en_fr",
                    "data_source": "foreign_1k",
                    "source_language": "fr",
                    "source_corpus": "fleurs",
                    "source_split": "train",
                    "source_id": "parent4",
                    "parent_audio_path": "parent.wav",
                    "seg_index": 0,
                    "seg_start": 0.0,
                    "seg_end": 1.0,
                    "n_segments": 184,
                }
                path = tmp_path / "sample.jsonl"
                path.write_text(json.dumps(row) + "\n")
                kwargs["jsonl_paths"] = str(path)
                result = create_audio_dataset(**kwargs)[0]
                assert result["extra_info"]["language"] == "English"
                assert "Only include speech" not in result["prompt"][0]["content"]
                assert result["reward_model"]["ground_truth"] == ("<foreign_language>" if masked else "Hello")
                if result["data_source"] == "earnings22_parent4":
                    assert result["extra_info"]["parent_audio_path"] == "parent.wav"
                    assert result["extra_info"]["n_segments"] == 184


def test_foreign_validation_sources_combine_with_parent_metadata(tmp_path):
    config = load_config("remax_foreign_quartr_fl")
    for group in config.data.val_data:
        for source in group:
            name = source.post_process.add_field.fields.data_source
            row = {"id": name, "audio_path": "audio.wav", "text": "Hello", "language": "en"}
            if name == "earnings22_parent4":
                row.update(parent_audio_path="parent.wav", seg_index=0, n_segments=1, seg_start=0.0, seg_end=1.0)
            path = tmp_path / f"{name}.jsonl"
            path.write_text(json.dumps(row) + "\n")
            source.jsonl_paths = str(path)
    ds = ASRDataset(config.data.val_files, None, config.data)
    assert len(ds) == len(ds.ds) == 3
    by_source = {row["data_source"]: row for row in ds.ds}
    assert by_source["earnings22_parent4"]["extra_info"]["n_segments"] == 1
    assert by_source["earnings22_parent4"]["extra_info"]["parent_audio_path"] == "parent.wav"
    for name in ("enhint_en", "enhint_noen"):
        assert by_source[name]["extra_info"]["n_segments"] is None
        assert by_source[name]["extra_info"]["parent_audio_path"] is None

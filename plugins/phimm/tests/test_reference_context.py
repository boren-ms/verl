"""Fixed reference histories preserve parent rewards and cannot leak future labels."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from recipe.phimm.agentic.dataset import ASRDataset
from recipe.phimm.tools.offline_reference_context import materialize, reference_contexts


def merged_row():
    originals = [
        {"audio_path": "first.wav", "text": "First label.", "language": "English"},
        {"audio_path": "foreign.wav", "text": "", "language": "Spanish"},
        {"audio_path": "last.wav", "text": "Last label.", "language": "English"},
    ]
    return {
        "id": "parent",
        "audio_path": [row["audio_path"] for row in originals],
        "text": " ".join(row["text"] for row in originals),
        "source_metadata_json": [json.dumps(row) for row in originals],
        "audio_length": 3,
        "keywords": ["producer"],
        "random_merge_source_indices": [4, 9, 2],
    }


def test_materialized_data_roundtrips_through_real_loader_without_changing_parent_contract(tmp_path):
    row = merged_row()
    path = tmp_path / "merged.jsonl"
    path.write_text(json.dumps(row) + "\n")
    source = {
        "dataset_name": "jsonl",
        "jsonl_paths": str(path),
        "add_task_info": {"style": "verb", "language": "English", "lang_hint": True},
        "post_process": {"add_field": {"fields": {"data_source": "earn"}}, "verl_format": {}},
    }
    config = OmegaConf.create(
        {
            "data": {
                "version": 2609,
                "train_files": "train",
                "val_files": "val",
                "train_data": [[source]],
                "val_data": [[source]],
            },
            "asr": {"context_aware": {"enable": True, "training_source": "reference"}},
        }
    )
    before = deepcopy(config)
    output = tmp_path / "fixed"
    snapshot = materialize(config, output)
    assert config == before and snapshot.data.val_data == config.data.val_data
    with initialize_config_dir(config_dir=str(output), version_base=None):
        loaded = compose(config_name="training")
    assert loaded == snapshot
    raw = json.loads((output / "source_000.jsonl").read_text())
    histories = raw.pop("segment_reference_contexts")
    assert raw == row
    assert histories == [[], ["First label."], ["First label.", ""]]
    train = ASRDataset("train", None, loaded.data)
    assert len(train) == 1
    sample = train[0]
    assert sample["reward_model"]["ground_truth"] == row["text"]
    assert sample["extra_info"]["keywords"] == ["producer"]
    segments = sample["asr_segments"]
    assert [seg["audio_path"] for seg in segments] == row["audio_path"]
    assert [seg["asr_reference_context"] for seg in segments] == histories
    assert all(seg["reward_model"] == sample["reward_model"] for seg in segments)
    assert all("asr_reference_context" not in seg for seg in ASRDataset("val", None, loaded.data)[0]["asr_segments"])
    audit = json.loads((output / "preparation.json").read_text())["sources"][0]
    assert audit["parents"] == 1 and audit["segments"] == 3
    with pytest.raises(FileExistsError):
        materialize(config, output)


@pytest.mark.parametrize("corruption", ["missing", "path", "text", "length", "parent", "duplicate"])
def test_unaligned_or_repeated_preparation_is_rejected(corruption):
    row = merged_row()
    if corruption == "missing":
        row.pop("source_metadata_json")
    elif corruption in {"path", "text"}:
        original = json.loads(row["source_metadata_json"][0])
        original["audio_path" if corruption == "path" else "text"] = "wrong.wav" if corruption == "path" else None
        row["source_metadata_json"][0] = json.dumps(original)
    elif corruption == "length":
        row["source_metadata_json"].pop()
    elif corruption == "parent":
        row["text"] = "different target"
    else:
        row["segment_reference_contexts"] = []
    with pytest.raises(ValueError):
        reference_contexts(row)


@pytest.mark.parametrize("contexts", [None, [], [[], []], [[], ["ok"], ["future"]], [[], [1], ["ok", ""]]])
def test_dataset_rejects_misaligned_fixed_contexts(contexts):
    ds = ASRDataset.__new__(ASRDataset)
    ds.ds = [{"audio_path": ["a", "b", "c"], "prompt": [], "extra_info": {"segment_reference_contexts": contexts}}]
    ds.groups, ds.prompt_key = None, "prompt"
    with pytest.raises(ValueError, match="segment_reference_contexts"):
        ds[0]


def test_reference_preset_preserves_original_experiment_budget_and_changes_only_training_context():
    config_dir = Path(__file__).resolve().parents[1] / "src/recipe/phimm/config/v2611_dev/earnings"
    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="remax_fl_earnings_short_random_merge23_x4_reference_context")
        base = compose(config_name="remax_fl_earnings_short_random_merge23_x4_text_history2")
    # Rollout logging paths interpolate the experiment name.
    base.trainer.experiment_name = config.trainer.experiment_name
    original_sources = OmegaConf.to_container(base.data.train_data[0], resolve=True)
    sources = OmegaConf.to_container(config.data.train_data[0], resolve=True)
    assert len(sources) == len(original_sources) == 6
    root = "az://orngwus2cresco/data/boren/data/Training/fl_earnings_short_random_merge23_x4_reference_context_v1"
    for i, (source, original) in enumerate(zip(sources, original_sources, strict=True)):
        original["jsonl_paths"] = f"{root}/source_{i:03d}.jsonl"
        original["post_process"]["verl_format"]["extra_keys"].append("segment_reference_contexts")
        assert source == original
    base.data.train_data = config.data.train_data
    assert config.data == base.data and config.actor_rollout_ref == base.actor_rollout_ref
    assert config.reward_map == base.reward_map and config.rewards == base.rewards
    assert config.asr.context_aware.training_source == "reference"
    assert config.asr.context_aware.max_history == 1
    assert config.asr.expected_training_rows == 111888
    assert config.trainer.total_epochs == 2
    assert config.trainer.total_training_steps == config.asr.expected_training_steps == 1748

"""Fixed reference histories preserve parent rewards and cannot leak future labels."""

import gzip
import json
from copy import deepcopy
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from recipe.phimm.agentic.dataset import ASRDataset
from recipe.phimm.tools.offline_reference_context import materialize, reference_texts


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


@pytest.mark.parametrize(
    "validation_source",
    [None, "generated", "reference", [{"start_step": 1, "source": None}, {"start_step": 201, "source": "reference"}]],
)
@pytest.mark.parametrize("compressed", [False, True])
def test_materialized_data_roundtrips_through_real_loader_without_changing_parent_contract(
    tmp_path, validation_source, compressed
):
    row = merged_row()
    path = tmp_path / ("merged.jsonl.gz" if compressed else "merged.jsonl")
    payload = (json.dumps(row) + "\n").encode()
    path.write_bytes(gzip.compress(payload) if compressed else payload)
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
            "asr": {
                "context_aware": {
                    "enable": True,
                    "training_source": "reference",
                    "validation_source": validation_source,
                }
            },
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
    references = raw.pop("segment_reference_texts")
    assert raw == row
    assert references == ["First label.", "", "Last label."]
    train = ASRDataset("train", None, loaded.data)
    assert len(train) == 1
    sample = train[0]
    assert sample["reward_model"]["ground_truth"] == row["text"]
    assert sample["extra_info"]["keywords"] == ["producer"]
    segments = sample["asr_segments"]
    assert [seg["audio_path"] for seg in segments] == row["audio_path"]
    assert [seg["asr_reference_text"] for seg in segments] == references
    assert all("asr_reference_context" not in seg for seg in segments)
    assert all(seg["reward_model"] == sample["reward_model"] for seg in segments)
    assert all("asr_reference_context" not in seg for seg in ASRDataset("val", None, loaded.data)[0]["asr_segments"])
    preparation = json.loads((output / "preparation.json").read_text())
    assert preparation["validation_source"] == validation_source
    audit = preparation["sources"][0]
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
        reference_texts(row)


@pytest.mark.parametrize("references", [None, [], ["a", "b"], ["a", None, "c"], ["a", 1, "c"]])
def test_dataset_rejects_misaligned_own_segment_texts(references):
    ds = ASRDataset.__new__(ASRDataset)
    ds.ds = [{"audio_path": ["a", "b", "c"], "prompt": [], "extra_info": {"segment_reference_texts": references}}]
    ds.groups, ds.prompt_key = None, "prompt"
    with pytest.raises(ValueError, match="segment_reference_texts"):
        ds[0]


def test_normalized_component_references_match_contig5_provenance():
    row = merged_row()
    row["components"] = [{"text": text} for text in ["First label.", "", "Last label."]]
    row["source_ids"] = ["first", "foreign", "last"]
    for i, value in enumerate(row["source_metadata_json"]):
        original = json.loads(value)
        original["display"] = original.pop("text")
        original["id"] = row["source_ids"][i]
        row["source_metadata_json"][i] = json.dumps(original)
        row["audio_path"][i] = "az://account/data/" + original["audio_path"]
    assert reference_texts(row) == ["First label.", "", "Last label."]
    row["source_ids"][1] = "wrong segment"
    with pytest.raises(ValueError, match="segment IDs"):
        reference_texts(row)


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


def test_context_schedule_preset_uses_one_epoch_and_three_phases():
    from recipe.phimm.agentic.context_prompt import context_source

    config_dir = Path(__file__).resolve().parents[1] / "src/recipe/phimm/config/v2611_dev/earnings"
    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="remax_fl_earnings_short_random_merge23_x4_context_schedule_history2_1epoch")
        base = compose(config_name="remax_fl_earnings_short_random_merge23_x4_reference_context")
    base.trainer.experiment_name = config.trainer.experiment_name
    config_data = OmegaConf.to_container(config.data)
    base_data = OmegaConf.to_container(base.data)
    config_data.pop("val_data")
    base_data.pop("val_data")
    assert config_data == base_data and config.actor_rollout_ref == base.actor_rollout_ref
    assert config.trainer.total_epochs == 1
    assert config.trainer.total_training_steps == config.asr.expected_training_steps == 874
    assert config.trainer.save_freq == 100
    assert config.trainer.test_freq == 400
    assert [source.post_process.add_field.fields.data_source for group in config.data.val_data for source in group] == [
        "earnings21",
        "earnings22",
        "coraal",
    ]
    assert config.asr.context_aware.max_history == 2
    assert [
        context_source(config.asr.context_aware, training=True, global_steps=step)
        for step in (1, 291, 292, 583, 584, 874)
    ] == [
        "none",
        "none",
        "reference",
        "reference",
        "generated",
        "generated",
    ]


def test_context_schedule_refext_preset_extends_to_two_epochs_with_reference_context():
    from recipe.phimm.agentic.context_prompt import context_source

    config_dir = Path(__file__).resolve().parents[1] / "src/recipe/phimm/config/v2611_dev/earnings"
    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="remax_fl_earnings_short_random_merge23_x4_context_schedule_history2_2epoch_refext")
        oracle = compose(config_name="remax_fl_earnings_short_random_merge23_x4_reference_context_openlong")
        stage = compose(config_name="remax_fl_earnings_short_random_merge23_x4_context_schedule_history2_1epoch")
    stage.trainer.experiment_name = config.trainer.experiment_name
    assert config.data.train_data == oracle.data.train_data and config.data.val_data == oracle.data.val_data
    assert config.actor_rollout_ref == stage.actor_rollout_ref
    assert config.trainer.total_epochs == 2
    assert config.trainer.total_training_steps == config.asr.expected_training_steps == 1748
    assert config.asr.expected_training_rows == 111888
    assert (config.trainer.save_freq, config.trainer.test_freq) == (100, 400)
    options = config.asr.context_aware
    assert options.max_history == 2
    assert [context_source(options, training=True, global_steps=step) for step in (1, 292, 583, 584, 874, 875, 1748)] == [
        "none",
        "reference",
        "reference",
        "generated",
        "generated",
        "reference",
        "reference",
    ]
    assert context_source(options, training=False, global_steps=874) == "reference"

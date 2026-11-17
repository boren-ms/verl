"""U2 defaults use complete parent references with either decoding policy."""

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_u2_context_config_preserves_training_and_clip_inputs():
    from pathlib import Path

    import recipe.phimm

    directory = Path(recipe.phimm.__file__).parent / "config/v2611_dev/earnings"
    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        base = compose(config_name="remax_earnings_short_context_aware")
        config = compose(config_name="remax_earnings_short_context_aware_u2_context")
    OmegaConf.resolve(config)
    assert config.asr.context_aware
    assert config.data.get("validation_reference_scope", "parent") == "parent"
    for key in ("train_data", "mix", "train_batch_size"):
        assert config.data[key] == base.data[key]
    assert "offline_data" not in config and "offline_data" not in base
    assert config.rewards == base.rewards
    assert config.algorithm == base.algorithm
    assert config.trainer.total_training_steps == base.trainer.total_training_steps == 394
    for old, new in zip(base.data.val_data[0], config.data.val_data[0], strict=True):
        assert old.jsonl_paths == new.jsonl_paths
        assert "2parents_fullref_" in new.jsonl_paths
        assert new.jsonl_paths.endswith(".jsonl.gz")
        source = new.post_process.add_field.fields.data_source
        assert source == old.post_process.add_field.fields.data_source
        assert config.reward_map[source] == "val_openasr_en"
        assert {"parent_audio_path", "seg_start", "seg_index"} <= set(new.post_process.verl_format.extra_keys)


@pytest.mark.parametrize("bad", ["missing", "different", "duplicate"])
def test_full_reference_builder_rejects_incomplete_or_inconsistent_inputs(bad):
    from recipe.phimm.data.u2_full_reference import full_reference_rows

    rows = [
        {
            "id": str(i),
            "audio_path": f"audio#{i}:{i + 1}",
            "parent_audio_path": "parent",
            "seg_index": i,
            "n_segments": 2,
            "seg_start": i,
            "seg_end": i + 1,
            "text": text,
        }
        for i, text in enumerate(["hello", "world"])
    ]
    refs = [{**rows[0], "text": "hello world", "seg_end": 2}]
    if bad == "missing":
        refs = []
    if bad == "different":
        refs[0]["text"] = "different full reference"
    if bad == "duplicate":
        rows[1]["seg_index"] = 0
    with pytest.raises(ValueError):
        full_reference_rows(rows, refs, lambda text: " ".join(text.lower().split()))


def test_full_reference_builder_retains_clips_and_uses_original_parent_text():
    from recipe.phimm.data.u2_full_reference import full_reference_rows

    rows = [
        {
            "id": str(i),
            "audio_path": f"audio#{i}:{i + 1}",
            "parent_audio_path": "parent",
            "seg_index": i,
            "n_segments": 2,
            "seg_start": i,
            "seg_end": i + 1,
            "text": text,
            "baseline_wer": 0.1,
        }
        for i, text in enumerate(["hello", "world"])
    ]
    refs = [{**rows[0], "text": "Hello World", "seg_end": 2}]
    output, audit = full_reference_rows(rows[::-1], refs, lambda text: " ".join(text.lower().split()))
    assert [r["audio_path"] for r in output] == [r["audio_path"] for r in rows]
    assert [r["text"] for r in output] == ["Hello World", "Hello World"]
    assert [r["legacy_segment_text"] for r in output] == ["hello", "world"]
    assert all(r["reference_scope"] == "parent" for r in output)
    assert "baseline_wer" not in output[0]
    assert output[0]["legacy_segment_baseline_wer"] == 0.1
    assert audit[0]["normalized_join_matches_original"]


def test_full_u2_dataset_groups_and_keeps_parent_reference_once(monkeypatch):
    import datasets
    from recipe.phimm.agentic import dataset

    rows = [
        {
            "data_source": "u2",
            "audio_path": f"parent#{i}:{i + 1}",
            "prompt": "task",
            "reward_model": {"ground_truth": "Hello World"},
            "extra_info": {"parent_audio_path": "parent", "seg_start": i, "reference_scope": "parent"},
        }
        for i in [1, 0]
    ]
    monkeypatch.setattr(dataset, "create_audio_dataset", lambda **kwargs: datasets.Dataset.from_list(rows))
    config = OmegaConf.create({"train_files": "train", "train_data": [], "val_data": [{}], "version": 2609})
    val = dataset.ASRDataset("val", tokenizer=None, config=config)
    assert len(val) == 1
    assert [r["extra_info"]["seg_start"] for r in val[0]["asr_segments"]] == [0, 1]
    assert val[0]["reward_model"]["ground_truth"] == "Hello World"


def test_u2_decode_pair_only_changes_context_and_identity():
    from pathlib import Path

    import recipe.phimm

    directory = Path(recipe.phimm.__file__).parent / "config/v2611_dev/eval"
    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        configs = [
            compose(
                config_name="eval_u2_context",
                overrides=[
                    "actor_rollout_ref.model.path=/verified/step80",
                    f"trainer.experiment_name=u2-{mode}",
                    f"asr.context_aware={mode}",
                ],
            )
            for mode in ("true", "false")
        ]
    for config in configs:
        assert config.trainer.val_only and config.trainer.resume_mode == "disable"
        assert config.actor_rollout_ref.model.lora_rank == 0
        assert config.data.get("validation_reference_scope", "parent") == "parent"
        assert config.trainer.nnodes == 2 and config.trainer.n_gpus_per_node == 8
    # Normalize identity before resolution so all derived artifact paths match.
    configs[1].trainer.experiment_name = configs[0].trainer.experiment_name
    configs[1].asr.context_aware = True
    for config in configs:
        OmegaConf.resolve(config)
    assert configs[0] == configs[1]


def test_every_u2_manifest_defaults_to_full_parent_references():
    from pathlib import Path

    import recipe.phimm

    directory = Path(recipe.phimm.__file__).parent / "config/data/val_data/openasr"
    manifests = list(directory.glob("openlong_all_2parents_error_2609r2b_u2*.yaml"))
    assert len(manifests) == 1
    for path in manifests:
        sources = OmegaConf.load(path)
        assert len(sources) == 4
        for source in sources:
            assert "2parents_fullref_" in source.jsonl_paths
            assert source.jsonl_paths.endswith(".jsonl.gz")
            assert {"parent_audio_path", "seg_start", "seg_index", "reference_scope"} <= set(
                source.post_process.verl_format.extra_keys
            )

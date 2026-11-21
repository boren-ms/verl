"""Protect source identity and model-only native SFT validation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from recipe.phimm.sft.validation_state import checkpoint_snapshot, checkpoint_steps, source_revision


def test_source_revision_is_independent_of_ray_worker_cwd(tmp_path, monkeypatch):
    expected = source_revision()
    monkeypatch.chdir(tmp_path)
    assert source_revision() == expected
    assert len(expected) == 40


def test_dai_sft_checkpoint_evaluation_keeps_speaker_prompt_and_der_scoring():
    import recipe.phimm

    directory = Path(recipe.phimm.__file__).parent / "config/v2611_dev/sft"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        cfg = compose(
            config_name="eval_sft_dai_diarization_2611r2",
            overrides=[
                "trainer.experiment_name=dai-periodic",
                "actor_rollout_ref.model.path=az://test/exports/step1000/qwen_hf",
            ],
        )
    assert cfg.trainer.val_only and cfg.trainer.resume_mode == "disable"
    assert (cfg.trainer.nnodes, cfg.trainer.n_gpus_per_node) == (1, 8)
    assert cfg.actor_rollout_ref.model.path.endswith("/step1000/qwen_hf")
    assert cfg.actor_rollout_ref.model.lora_rank == 0
    assert cfg.data.val_data[0].post_process.verl_format.target_format == "speaker_jsonl"
    assert cfg.rewards.val_diarization.reward_kwargs.prediction_format == "speaker_jsonl"
    assert cfg.rewards.val_diarization.reward_kwargs.compute_wer


def test_native_sft_validation_preserves_base3a_decoding_and_has_no_training_resume():
    import recipe.phimm

    directory = Path(recipe.phimm.__file__).parent / "config/v2611_dev/sft"
    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        candidate = compose(
            config_name="eval_sft_checkpoints_base3a_2611r2",
            overrides=["asr.sft_validation.checkpoint_root=/checkpoints/sft", "trainer.experiment_name=test-sft-val"],
        )
        reference = compose(
            config_name="eval_sft_base3a_2611r2",
            overrides=["actor_rollout_ref.model.path=/merged", "trainer.experiment_name=test-merged-val"],
        )
    OmegaConf.resolve(candidate)
    assert candidate.data.val_data == reference.data.val_data
    assert candidate.reward_map == reference.reward_map
    assert candidate.actor_rollout_ref.rollout.val_kwargs == reference.actor_rollout_ref.rollout.val_kwargs
    assert candidate.actor_rollout_ref.rollout.custom == reference.actor_rollout_ref.rollout.custom
    assert candidate.trainer.val_only and candidate.trainer.resume_mode == "disable"
    assert candidate.actor_rollout_ref.actor.checkpoint.load_contents == ["model"]
    assert candidate.actor_rollout_ref.model.lora.merge
    assert (candidate.actor_rollout_ref.model.lora_rank, candidate.actor_rollout_ref.model.lora_alpha) == (320, 640)
    assert checkpoint_steps(candidate.asr.sft_validation.steps) == list(range(20, 869, 20)) + [868]


@pytest.mark.parametrize("steps", [[], [20, 20], [40, 20], [0, 20], [20.0], [True]])
def test_reject_ambiguous_checkpoint_sequence(steps):
    with pytest.raises(ValueError):
        checkpoint_steps(steps)


@pytest.fixture
def checkpoint(tmp_path):
    producer = {
        "model": {"path": "az://test/models/base/", "lora_rank": 320, "lora_alpha": 640},
        "trainer": {"nnodes": 2, "n_gpus_per_node": 2},
    }
    (tmp_path / "effective_config.yaml").write_text(OmegaConf.to_yaml(producer))
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("40")
    directory = tmp_path / "global_step_40"
    directory.mkdir()
    (directory / "lora_train_meta.json").write_text(json.dumps({"r": 320, "lora_alpha": 640}))
    for rank in range(4):
        (directory / f"model_world_size_4_rank_{rank}.pt").write_bytes(b"nonempty model shard")
    model = SimpleNamespace(path="az://test/models/base", lora_rank=320, lora_alpha=640)
    return tmp_path, model


def test_checkpoint_readiness_accepts_model_shards_without_optimizer_or_loader(checkpoint):
    root, model = checkpoint
    snapshot = checkpoint_snapshot(str(root), 40, model, 4)
    assert len(snapshot["model_shards"]) == 4
    assert snapshot["source_optimizer_step"] == 40
    assert snapshot["checkpoint"] == str(root / "global_step_40")
    assert checkpoint_snapshot(str(root), 60, model, 4) is None


@pytest.mark.parametrize("field,value", [("path", "wrong-base"), ("lora_rank", 160), ("lora_alpha", 320)])
def test_wrong_checkpoint_producer_fails_closed(checkpoint, field, value):
    root, model = checkpoint
    setattr(model, field, value)
    with pytest.raises(ValueError, match="producer mismatch"):
        checkpoint_snapshot(str(root), 40, model, 4)


def test_committed_checkpoint_with_missing_rank_is_an_error(checkpoint):
    root, model = checkpoint
    (root / "global_step_40/model_world_size_4_rank_2.pt").unlink()
    with pytest.raises(FileNotFoundError):
        checkpoint_snapshot(str(root), 40, model, 4)


def test_lora_metadata_is_checked_independently(checkpoint):
    root, model = checkpoint
    (root / "global_step_40/lora_train_meta.json").write_text(json.dumps({"r": 320, "lora_alpha": 320}))
    with pytest.raises(ValueError, match="metadata mismatch"):
        checkpoint_snapshot(str(root), 40, model, 4)

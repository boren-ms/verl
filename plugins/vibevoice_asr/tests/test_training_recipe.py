"""Exercise the complete Hydra inheritance used by the full training job."""

from pathlib import Path


def test_composed_training_recipe_bounds_padded_actor_activation_memory(monkeypatch):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    monkeypatch.setenv("VIBEVOICE_RL_MODEL", "/seed")
    monkeypatch.setenv("VIBEVOICE_RL_DATA_ROOT", "/data")
    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)
    directory = Path(__file__).resolve().parents[2] / "phimm/src/recipe/phimm/config/v2611_dev/vibevoice"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        config = compose(config_name="remax_vibevoice_ami_ls")

    # Check the composed actor setting: the shared base disables checkpointing.
    assert config.actor_rollout_ref.model.enable_gradient_checkpointing is True
    assert config.actor_rollout_ref.model.use_remove_padding is False
    assert config.actor_rollout_ref.actor.use_dynamic_bsz is False
    assert config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
    assert config.data.train_batch_size == config.actor_rollout_ref.actor.ppo_mini_batch_size == 128
    assert config.actor_rollout_ref.rollout.n == 8


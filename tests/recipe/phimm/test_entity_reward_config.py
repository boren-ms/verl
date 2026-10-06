from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_ROOT = Path(__file__).parents[3] / "recipe/phimm/config"
TRAINER_CONFIG_DIR = Path(__file__).parents[3] / "verl/trainer/config"
ENTITY_CONFIG_DIR = CONFIG_ROOT / "v2609_entity"


@pytest.mark.parametrize(
    "config_name",
    sorted(path.stem for path in ENTITY_CONFIG_DIR.glob("*.yaml")),
)
def test_entity_configs_use_local_2609_defaults(config_name):
    searchpath = f"hydra.searchpath=[file://{CONFIG_ROOT},file://{TRAINER_CONFIG_DIR}]"
    local = OmegaConf.load(ENTITY_CONFIG_DIR / f"{config_name}.yaml")
    assert all("v2609_asr" not in str(default) for default in local.defaults)
    with initialize_config_dir(config_dir=str(ENTITY_CONFIG_DIR), version_base=None):
        config = compose(config_name=config_name, overrides=[searchpath])
        base = compose(config_name="base", overrides=[searchpath])

    assert config.data.version == 2609
    assert config.actor_rollout_ref.model.path == base.actor_rollout_ref.model.path
    assert "phi-fastllm-2609" in config.actor_rollout_ref.model.path
    assert config.actor_rollout_ref.rollout.gt_rollout is False
    assert config.actor_rollout_ref.rollout.n == (4 if config_name.startswith("remax") else 1)
    assert config.actor_rollout_ref.model.lora_rank == 256
    assert config.val_reward.reward_function_by_data_source.name_val == "openasr_en"
    assert all(reward.reward_kwargs.version == 2609 for reward in config.val_reward.reward_functions.values())
    assert len(config.data.val_data) == 1
    assert config.data.val_data[0].post_process.add_field.fields.data_source == "name_val"

    if config_name.startswith("eval"):
        assert config.trainer.project_name == "v2609_entity_eval"
        assert config.trainer.val_only is True
        assert config.trainer.validation_resume is True
        assert OmegaConf.to_container(config.data.train_data) == OmegaConf.to_container(config.data.val_data)
    else:
        assert config.trainer.project_name == "v2609_entity"

    if config_name.startswith("remax"):
        assert config.reward_functions.asr_measure.reward_kwargs.version == 2609
        reward_kwargs = config.reward_functions.asr_measure.reward_kwargs
        assert OmegaConf.to_container(reward_kwargs.reduce) == {"mode": "average", "total": 4}
        assert OmegaConf.to_container(reward_kwargs.measures) == {
            "char": {"beta": 1.0},
            "word": {"beta": 1.0},
            "keyword": {"beta": 1.0},
            "lang": {"beta": 0.5},
            "fmt": {"beta": 0.5},
        }
        assert config.reward_function_by_data_source.name_train == "asr_measure"
        assert "name_enhc" in config_name
        assert config.data.train_batch_size == 128
        assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 128
        assert config.trainer.total_training_steps == 1000
        assert config.trainer.nnodes == 4
        assert config.trainer.n_gpus_per_node * config.trainer.nnodes == 32
        assert OmegaConf.to_container(config.trainer, resolve=False)["ngpus"] == (
            "${eval:${trainer.n_gpus_per_node}*${trainer.nnodes}}"
        )

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.trainer.config import AlgoConfig, BadCaseReplayConfig, BadCaseReplaySelectionConfig
from verl.utils.config import omega_conf_to_dataclass


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("name", ["ppo_trainer", "ppo_megatron_trainer"])
def test_replay_config_hydra_dataclass_and_snapshots_agree(name):
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        config = compose(config_name=name)
    replay = omega_conf_to_dataclass(config.algorithm).bad_case_replay
    assert isinstance(replay, BadCaseReplayConfig)
    assert isinstance(replay.selection, BadCaseReplaySelectionConfig)
    assert not replay.enable
    assert replay.capacity == 400
    assert not hasattr(replay, "fill_size")
    assert not hasattr(replay, "add_ratio")
    assert replay.replay_size == 16
    assert not hasattr(replay, "replay_ratio")
    assert replay.sampling == "random"
    assert replay.selection.weights == {"word": 1.0}
    assert replay.seed == 42

    for path in (
        ROOT / f"verl/trainer/config/_generated_{name}.yaml",
        ROOT / f"tests/trainer/config/legacy_{name}.yaml",
    ):
        snapshot = OmegaConf.load(path).algorithm.bad_case_replay
        assert OmegaConf.to_container(snapshot) == OmegaConf.to_container(config.algorithm.bad_case_replay)


def test_replay_defaults_do_not_share_mutable_metric_weights():
    first = AlgoConfig()
    second = AlgoConfig()
    first.bad_case_replay.selection.weights["word"] = 0.5
    first.bad_case_replay.selection.weights["char"] = 0.5
    assert second.bad_case_replay.selection.weights == {"word": 1.0}


@pytest.mark.parametrize("strategy", ["random", "worst"])
def test_asr_replay_usage_can_select_a_single_raw_metric(strategy):
    config_dir = ROOT / "recipe/phimm/config/v2611_dev/dev"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(
            config_name="remax_dev_base0_v1h2",
            overrides=[
                "algorithm.bad_case_replay.enable=true",
                "algorithm.bad_case_replay.replay_size=16",
                f"algorithm.bad_case_replay.sampling={strategy}",
                "algorithm.bad_case_replay.selection.weights.word=1.0",
            ],
        )
    replay = omega_conf_to_dataclass(config.algorithm).bad_case_replay
    assert replay.enable
    assert replay.replay_size == 16
    assert replay.sampling == strategy
    assert replay.selection.weights == {"word": 1.0}
    assert config.algorithm.adv_estimator == "remax"
    assert "word" in config.reward_functions.librispeech.reward_kwargs.measures
    restored = OmegaConf.to_object(OmegaConf.structured(replay))
    assert isinstance(restored, BadCaseReplayConfig)
    assert isinstance(restored.selection, BadCaseReplaySelectionConfig)
    assert restored.selection.weights == {"word": 1.0}


def test_replay_recipe_reserves_sixteen_of_128_inputs():
    config_dir = ROOT / "recipe/phimm/config/v2611_dev/dev"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="remax_dev_base0_v1h2a_fix_bcr")
    replay = omega_conf_to_dataclass(config.algorithm).bad_case_replay
    assert replay.enable
    assert not hasattr(replay, "fill_size")
    assert replay.replay_size == 16
    assert config.data.train_batch_size == 128
    assert config.trainer.nnodes == 2
    assert config.trainer.n_gpus_per_node == 8
    assert config.data.train_batch_size - replay.replay_size == 112
    assert replay.sampling == "random"
    assert replay.selection.weights == {"word": 1.0}


def test_replay_metric_weights_can_explicitly_add_character_accuracy():
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "algorithm.bad_case_replay.selection.weights.word=0.7",
                "+algorithm.bad_case_replay.selection.weights.char=0.3",
            ],
        )
    replay = omega_conf_to_dataclass(config.algorithm).bad_case_replay
    assert replay.selection.weights == {"word": 0.7, "char": 0.3}

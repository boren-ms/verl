from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_earning_recipe_preserves_base0a_settings_and_routes_quartr_and_earnings():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2611_dev/dev"), version_base=None):
        donor = compose(config_name="remax_dev_base0a_r0", overrides=[searchpath])
    with initialize_config_dir(config_dir=str(config_root / "v2611_dev/earning"), version_base=None):
        config = compose(config_name="remax_earning_base0a_r0", overrides=[searchpath])

    for section in ("actor_rollout_ref", "algorithm", "trainer", "reward_functions"):
        assert OmegaConf.to_container(config[section], resolve=False) == OmegaConf.to_container(
            donor[section], resolve=False
        )
    config_data = OmegaConf.to_container(config.data, resolve=False)
    donor_data = OmegaConf.to_container(donor.data, resolve=False)
    for key in donor_data.keys() - {"data_source", "train_data", "val_data"}:
        assert config_data[key] == donor_data[key]

    assert OmegaConf.to_container(config.data.data_source) == {
        "quartr_20261016": {"num_epoch": 1, "shuffle": True},
    }
    train_data = OmegaConf.to_container(config.data.train_data, resolve=True)
    assert len(train_data) == 1
    train = train_data[0]
    assert train == OmegaConf.load(config_root / "data/train_data/quartr_20261016_wer05.yaml")
    assert train["pre_process"]["rename_fields"]["mappings"] == {"wer": "wer_maiears"}
    assert train["post_process"]["filter_by_wer"]["wer_range"] == [0.0, 0.05]
    assert train["add_task_info"] == {
        "task": "lang_asr_verb", "language": "English", "prefix_prob": 0.0, "lang_hint": True,
    }
    train_source = train["post_process"]["add_field"]["fields"]["data_source"]
    assert OmegaConf.to_container(config.reward_function_by_data_source) == {train_source: "librispeech"}

    val_data = OmegaConf.to_container(config.data.val_data, resolve=True)
    assert len(val_data) == 1
    assert val_data[0] == OmegaConf.load(config_root / "data/val_data/earnings22_verb_langhint.yaml")
    assert len(val_data[0]) == 1
    val = val_data[0][0]
    assert "Earnings22-Cleaned-AA-chunked/" in val["jsonl_paths"]
    assert val["add_task_info"] == train["add_task_info"]
    val_source = val["post_process"]["add_field"]["fields"]["data_source"]
    reward_name = config.val_reward.reward_function_by_data_source[val_source]
    assert reward_name == "openasr_en"
    assert config.val_reward.reward_functions[reward_name] == donor.val_reward.reward_functions[reward_name]
    assert config.val_reward.reward_manager == donor.val_reward.reward_manager == "long_audio_grouped"

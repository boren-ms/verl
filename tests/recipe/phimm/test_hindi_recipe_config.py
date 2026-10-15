from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_hindi_recipe_preserves_base0_training_settings_and_routes_datasets():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2611_dev/dev"), version_base=None):
        donor = compose(config_name="remax_dev_base0_r0", overrides=[searchpath])
    with initialize_config_dir(config_dir=str(config_root / "v2611_dev/hindi"), version_base=None):
        config = compose(config_name="remax_hindi_base0_r0", overrides=[searchpath])
    with initialize_config_dir(config_dir=str(config_root / "v2611_dev/eval"), version_base=None):
        evaluation = compose(config_name="eval_2609_openml_hi_nl", overrides=[searchpath])

    for section in ("actor_rollout_ref", "algorithm", "trainer", "reward_functions", "val_reward"):
        assert OmegaConf.to_container(config[section], resolve=False) == OmegaConf.to_container(
            donor[section], resolve=False
        )
    for key in donor.data.keys() - {"data_source", "train_data", "val_data"}:
        assert OmegaConf.to_container(config.data, resolve=False)[key] == OmegaConf.to_container(
            donor.data, resolve=False
        )[key]

    assert OmegaConf.to_container(config.data.data_source) == {
        "indic_voice": {"num_epoch": 1, "shuffle": True},
    }
    train_data = OmegaConf.to_container(config.data.train_data, resolve=True)
    assert len(train_data) == 1
    train = train_data[0]
    assert train == OmegaConf.load(config_root / "data/train_data/indian/indic_voice.yaml")
    assert train["specs"][0]["manifest_file"].endswith("/IndicVoices/hi_verbatim_c96f9088/file_set_train.json")
    assert train["add_task_info"] == {"task": "lang_asr_verb", "language": "Hindi", "lang_hint": True}
    train_source = train["post_process"]["add_field"]["fields"]["data_source"]
    assert OmegaConf.to_container(config.reward_function_by_data_source) == {train_source: "librispeech"}

    val_data = OmegaConf.to_container(config.data.val_data, resolve=True)
    assert len(val_data) == 1
    assert val_data[0] == OmegaConf.load(config_root / "data/val_data/openasr_ml_hi_in_langhint.yaml")
    assert len(val_data[0]) == 1
    val = val_data[0][0]
    assert val["jsonl_paths"].endswith("/openasr_ml_jsonl/monsoon/hi_in/data.jsonl")
    assert val["add_task_info"] == {"task": "lang_asr", "language": "Hindi", "lang_hint": True}
    assert val["post_process"]["verl_format"]["extra_keys"] == ["lattice"]
    val_source = val["post_process"]["add_field"]["fields"]["data_source"]
    reward_name = config.val_reward.reward_function_by_data_source[val_source]
    reward = config.val_reward.reward_functions[reward_name]
    assert reward.path == "recipe/phimm/reward/asr_eval.py"
    assert reward.name == "openasr_hi_in_eval"
    assert reward.reward_kwargs.version == 2609
    assert evaluation.val_reward.reward_function_by_data_source[val_source] == reward_name
    assert evaluation.val_reward.reward_functions[reward_name] == reward

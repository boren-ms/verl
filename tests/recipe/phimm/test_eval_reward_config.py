from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_ROOT = Path(__file__).parents[3] / "recipe/phimm/config"
EVAL_CONFIG_DIR = CONFIG_ROOT / "v2609_asr/eval"
TRAINER_CONFIG_DIR = Path(__file__).parents[3] / "verl/trainer/config"
EVAL_CONFIG_NAMES = sorted(path.stem for path in EVAL_CONFIG_DIR.glob("eval_*.yaml"))

REWARD_SOURCES = {
    "openasr_hi_in": ["monsoon_hi_in"],
    "inhouse_zh": ["mixlang_fy26q2"],
    "openasr_en": [
        "name_val",
        "ls_clean",
        "ls_other",
        "spgispeech",
        "ami",
        "gigaspeech",
        "earnings22_cleaned_aa_chunked",
        "monsoon_en_in",
    ],
    "openasr_ml": [
        "de_fleurs",
        "fr_fleurs",
        "it_fleurs",
        "es_fleurs",
        "pt_fleurs",
        "de_mcv",
        "fr_mcv",
        "it_mcv",
        "es_mcv",
        "fr_mls",
        "it_mls",
        "es_mls",
        "pt_mls",
        "nl_fleurs",
        "nl_mcv",
        "nl_mls",
    ],
    "inhouse_all": [
        "enus_conv_fy21q1",
        "enus_conv_om_fy25q3",
        "enus_dict_office_fy24q3",
        "dadk_conv_fy21q3",
        "dadk_conv_om_fy23q1",
        "dadk_dict_fy23q4",
        "huhu_conv_fy22q4",
        "huhu_conv_om_fy24q2",
        "huhu_dict_fy25q2",
        "nbno_conv_fy21q3",
        "nbno_conv_om_fy23q1",
        "nbno_dict_fy23q4",
        "nlnl_conv_fy23q2",
        "nlnl_conv_om_fy23q1",
        "nlnl_dict_fy23q4",
        "cscz_conv_fy23q2",
        "cscz_conv_om_fy24q2",
        "cscz_dict_fy24q2",
    ],
}


@pytest.mark.parametrize("config_name", EVAL_CONFIG_NAMES)
@pytest.mark.parametrize("nnodes", [1, 2, 4])
def test_eval_batch_size_scales_with_nodes(config_name, nnodes):
    registered_eval = not OmegaConf.has_resolver("eval")
    if registered_eval:
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {}, {}))
    try:
        with initialize_config_dir(config_dir=str(EVAL_CONFIG_DIR), version_base=None):
            config = compose(
                config_name=config_name,
                overrides=[
                    f"hydra.searchpath=[file://{CONFIG_ROOT},file://{TRAINER_CONFIG_DIR}]",
                    f"trainer.nnodes={nnodes}",
                ],
            )
        assert config.data.val_batch_size == 256 * nnodes
    finally:
        if registered_eval:
            OmegaConf.clear_resolver("eval")


@pytest.mark.parametrize("config_name", EVAL_CONFIG_NAMES)
def test_eval_configs_inherit_reward_assignments(config_name):
    with initialize_config_dir(config_dir=str(EVAL_CONFIG_DIR), version_base=None):
        config = compose(
            config_name=config_name,
            overrides=[f"hydra.searchpath=[file://{CONFIG_ROOT},file://{TRAINER_CONFIG_DIR}]"],
        )

    assignments = OmegaConf.to_container(config.val_reward.reward_function_by_data_source, resolve=True)
    assert config.trainer.validation_resume is True
    for reward_function, sources in REWARD_SOURCES.items():
        for source in sources:
            assert assignments[source] == reward_function

    aa_reward_function = "openasr_ml" if config_name == "eval_2609_mix_openml_aa_ter30" else "openasr_en"
    assert assignments["earnings22_cleaned_aa"] == aa_reward_function
    assert assignments["voxpopuli_cleaned_aa"] == aa_reward_function
    assert set(assignments.values()) <= set(config.val_reward.reward_functions)
    for reward_function in config.val_reward.reward_functions.values():
        assert reward_function.reward_kwargs.version == 2609

    if config_name != "eval_base":
        local_config = OmegaConf.load(EVAL_CONFIG_DIR / f"{config_name}.yaml")
        assert "reward_function_by_data_source" not in local_config.get("val_reward", {})


@pytest.mark.parametrize(
    "config_path",
    [
        "v2609_asr/eval/eval_2609_openml_hi_nl",
        "v2609_asr/eval/eval_2609_openml_verb",
        "v2609_asr/eval/eval_2609_openml_verb_langhint",
        "v2609_asr/eval/eval_2609_mix_openml_aa",
        "v2609_asr/eval/eval_2609_mix_openml_aa_ter30",
    ],
)
def test_hindi_eval_configs_select_dedicated_oiwer_function(config_path):
    from verl.trainer.ppo.reward import get_custom_reward_fn

    path = CONFIG_ROOT / config_path
    with initialize_config_dir(config_dir=str(path.parent), version_base=None):
        config = compose(config_name=path.name)

    hindi_rows = [
        row for row in config.data.val_data
        if row.post_process.add_field.fields.data_source == "monsoon_hi_in"
    ]
    assert len(hindi_rows) == 1
    assert "lattice" in hindi_rows[0].post_process.verl_format.extra_keys
    registration = config.val_reward.reward_function_by_data_source.monsoon_hi_in
    function_config = config.val_reward.reward_functions[registration]
    assert function_config.name == "openasr_hi_in_eval"
    scorer = get_custom_reward_fn(config.val_reward, function_config)
    result = scorer(
        "<TXT>world today</TXT>",
        "hello today",
        data_source="monsoon_hi_in",
        extra_info={"language": "Hindi", "lattice": [["hello", "world"], ["today"]]},
    )
    assert result == {"score": 1.0, "wer": 0.0, "n_err": 0, "n_ref": 2}


def test_openml_hi_nl_contains_only_existing_hindi_and_dutch_datasets():
    with initialize_config_dir(config_dir=str(EVAL_CONFIG_DIR), version_base=None):
        config = compose(config_name="eval_2609_openml_hi_nl")
        full_config = compose(config_name="eval_2609_openml_verb")

    rows = OmegaConf.to_container(config.data.val_data, resolve=True)
    sources = [row["post_process"]["add_field"]["fields"]["data_source"] for row in rows]
    assert sources == ["monsoon_hi_in", "nl_fleurs", "nl_mcv", "nl_mls"]
    full_rows = {
        row["post_process"]["add_field"]["fields"]["data_source"]: row
        for row in OmegaConf.to_container(full_config.data.val_data, resolve=True)
    }
    assert rows == [full_rows[source] for source in sources]
    assert OmegaConf.to_container(config.data.train_data, resolve=True) == rows
    assert config.data.version == 2609
    assert config.trainer.val_only is True
    assert config.trainer.val_before_train is True
    assert config.actor_rollout_ref.model.lora_rank == full_config.actor_rollout_ref.model.lora_rank
    for source in sources:
        registration = config.val_reward.reward_function_by_data_source[source]
        expected = "openasr_hi_in_eval" if source == "monsoon_hi_in" else "openasr_eval"
        assert config.val_reward.reward_functions[registration].name == expected

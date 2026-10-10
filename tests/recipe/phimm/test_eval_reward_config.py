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
        "urgent2024",
        "urgent2024_clean",
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


def test_r2_openall_mix_changes_only_model_checkpoint():
    reference = OmegaConf.load(
        CONFIG_ROOT / "v2609_asr/remax_2609r2_earning_ml_verb_hint_s400_bs128_n8_r256_g32_smp8_avg4.yaml"
    )
    with initialize_config_dir(config_dir=str(EVAL_CONFIG_DIR), version_base=None):
        original = compose(config_name="eval_2609_openall_mix")
        r2 = compose(config_name="eval_2609r2_openall_mix")

    assert r2.actor_rollout_ref.model.path == reference.actor_rollout_ref.model.path
    assert r2.actor_rollout_ref.model.path != original.actor_rollout_ref.model.path
    r2.actor_rollout_ref.model.path = original.actor_rollout_ref.model.path
    assert OmegaConf.to_container(r2, resolve=False) == OmegaConf.to_container(original, resolve=False)


def test_r2_ls_clean_uses_zip_manifest_without_path_rewriting():
    with initialize_config_dir(config_dir=str(EVAL_CONFIG_DIR), version_base=None):
        config = compose(config_name="eval_2609r2_openall_mix_zip")
    entries = [
        row for group in config.data.val_data for row in group
        if row.post_process.add_field.fields.data_source == "ls_clean"
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.jsonl_paths == "az://orngwus2cresco/data/boren/data/openasr_jsonl/ls-clean/data_zip.jsonl"
    assert entry.cache_name == "auto_openasr_2606_ls_clean_zip_verb_langhint"
    assert "pre_process" not in entry
    assert entry.add_task_info.task == "lang_asr_verb"
    assert entry.add_task_info.language == "English"
    assert entry.add_task_info.lang_hint is True


def test_r2_all_datasets_use_zip_manifests_and_fresh_caches():
    with initialize_config_dir(config_dir=str(EVAL_CONFIG_DIR), version_base=None):
        config = compose(config_name="eval_2609r2_openall_mix_zip")
    entries = [row for group in config.data.val_data for row in group]
    assert len(entries) == 26
    assert len({row.cache_name for row in entries}) == 26
    for row in entries:
        assert row.dataset_name == "jsonl"
        assert row.jsonl_paths.endswith("_zip.jsonl")
        assert "zip" in row.cache_name
        assert "path_map" not in row.get("pre_process", {})
    by_source = {row.post_process.add_field.fields.data_source: row for row in entries}
    assert by_source["mixlang_fy26q2"].pre_process.rename_fields.mappings.audio_path == "WavPath"
    assert "parent_audio_path" in by_source["mixlang_fy26q2"].post_process.verl_format.extra_keys
    for source in ("voxpopuli_cleaned_aa", "earnings22_cleaned_aa_chunked"):
        assert by_source[source].pre_process.rename_fields.mappings.audio_path == "url"
        assert "parent_audio_path" in by_source[source].post_process.verl_format.extra_keys
    assert "lattice" in by_source["monsoon_hi_in"].post_process.verl_format.extra_keys


@pytest.mark.parametrize("config_name", ["eval_2609_openall_mix", "eval_2609r2_openall_mix"])
def test_zip_eval_preserves_original_settings_and_dataset_behavior(config_name):
    with initialize_config_dir(config_dir=str(EVAL_CONFIG_DIR), version_base=None):
        original = compose(config_name=config_name)
        packed = compose(config_name=f"{config_name}_zip")
    original_entries = [row for group in original.data.val_data for row in group]
    packed_entries = [row for group in packed.data.val_data for row in group]
    assert len(original_entries) == len(packed_entries) == 26
    for old, new in zip(original_entries, packed_entries, strict=True):
        old = OmegaConf.to_container(old, resolve=True)
        new = OmegaConf.to_container(new, resolve=True)
        assert "_zip.jsonl" not in old.get("jsonl_paths", "")
        old.pop("cache_name", None)
        new.pop("cache_name")
        old.pop("jsonl_paths", None)
        new.pop("jsonl_paths")
        if old["dataset_name"] == "parquet":
            old["dataset_name"] = "jsonl"
            old.pop("parquet_paths")
            old.pop("decode_audio")
        pre_process = old.get("pre_process", {})
        pre_process.pop("path_map", None)
        if not pre_process:
            old.pop("pre_process", None)
        assert old == new
    for group in ("_mixlang", "_openasr", "_openml"):
        packed[group] = original[group]
    assert OmegaConf.to_container(packed, resolve=False) == OmegaConf.to_container(original, resolve=False)


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


@pytest.mark.parametrize("config_name", ["eval_2609r2_urgent_zip", "eval_2609r2a_urgent_zip"])
def test_urgent_eval_sources_dispatch_to_english_scorer(config_name):
    from verl.trainer.ppo.reward import get_reward_fn_dispatcher

    with initialize_config_dir(config_dir=str(EVAL_CONFIG_DIR), version_base=None):
        config = compose(config_name=config_name)
    scorer = get_reward_fn_dispatcher(config.val_reward)
    entries = [row for group in config.data.val_data for row in group]
    assert [row.post_process.add_field.fields.data_source for row in entries] == [
        "urgent2024", "urgent2024_clean",
    ]
    for row in entries:
        source = row.post_process.add_field.fields.data_source
        assert config.val_reward.reward_function_by_data_source[source] == "openasr_en"
        result = scorer(
            source,
            "<TXT>hello world</TXT>",
            "hello world",
            extra_info={"language": "English"},
        )
        assert result["score"] == 1.0
        assert result["n_err"] == 0
        assert result["n_ref"] == 2


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

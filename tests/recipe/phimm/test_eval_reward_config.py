from pathlib import Path
import sys

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
        "hiin_conversation_dtest_fy23q3_hi_in_dtest",
        "hiin_conversation_domainset_dtest_banking_entity_fy25q2_hi_in_dtest_offlinedatacollection",
        "hiin_conversation_domainset_dtest_medical_entity_fy24q1_hi_in_dtest",
        "hiin_conversation_onlinemeetings_dtest_fy24q2_hi_in",
        "hiin_dictation_dtest_l_d_fy23q2_hi_in_dtest",
    ],
}


@pytest.mark.parametrize(
    ("group_index", "entry_index", "language", "words"),
    [
        (0, 0, "English", ["hello", "world"]),
        (1, 0, "German", ["hallo", "welt"]),
        (2, 0, "English", ["hello", "world"]),
        (2, 1, "English", ["hello", "world"]),
    ],
    ids=["earnings", "openml", "ls_clean", "ls_other"],
)
def test_base0a_nohint_validation_scores_separately_from_hinted(
    group_index, entry_index, language, words
):
    import numpy as np
    import torch
    from datasets import Dataset

    from recipe.phimm.data.dataset import add_task_info
    from recipe.phimm.reward.long_audio_grouped import LongAudioGroupedRewardManager
    from verl import DataProto
    from verl.trainer.ppo.reward import get_reward_fn_dispatcher

    searchpath = f"hydra.searchpath=[file://{CONFIG_ROOT},file://{TRAINER_CONFIG_DIR}]"
    with initialize_config_dir(config_dir=str(CONFIG_ROOT / "v2611_dev/dev"), version_base=None):
        config = compose(config_name="remax_dev_base0a_r0", overrides=[searchpath])

    groups = OmegaConf.to_container(config.data.val_data, resolve=True)
    assert len(groups) == 6
    hinted = groups[group_index][entry_index]
    nohint = groups[group_index + 3][entry_index]
    assert hinted["jsonl_paths"] == nohint["jsonl_paths"]
    if "cache_name" in hinted:
        assert hinted["cache_name"] != nohint["cache_name"]

    responses = {}
    sources = []
    extra_info = []
    for entry, use_hint in [(hinted, True), (nohint, False)]:
        task_info = entry["add_task_info"]
        assert task_info["task"] == ("lang_asr_verb" if use_hint else "lang_asr")
        assert task_info["lang_hint"] is use_hint
        dataset = add_task_info(
            Dataset.from_dict({"text": words, "language": [language] * 2}),
            **task_info,
            version=config.data.version,
        )
        source = entry["post_process"]["add_field"]["fields"]["data_source"]
        for index, row in enumerate(dataset):
            assert ("The language is" in row["prompt"]) is use_hint
            assert ("<VERBATIM>" in row["gt_output"]) is use_hint
            assert row["prefix"] == ""
            responses[len(responses) + 1] = row["gt_output"]
            sources.append(source)
            extra_info.append({
                "parent_audio_path": "shared-parent.wav",
                "seg_start": float(index),
                "language": language,
                "keywords": words,
            })

    assert sources[0] != sources[2]
    routes = config.val_reward.reward_function_by_data_source
    assert routes[sources[0]] == routes[sources[2]]
    scorer = get_reward_fn_dispatcher(config.val_reward)
    calls = []

    def compute_score(**kwargs):
        calls.append(kwargs)
        return scorer(**kwargs)

    class Tokenizer:
        eos_token = None

        def decode(self, token_ids, skip_special_tokens=True):
            return responses[int(token_ids[0])]

    data = DataProto.from_dict(
        tensors={
            "prompts": torch.ones((4, 1), dtype=torch.long),
            "responses": torch.arange(1, 5).reshape(4, 1),
            "attention_mask": torch.ones((4, 2), dtype=torch.long),
        },
        non_tensors={
            "extra_info": np.array(extra_info, dtype=object),
            "reward_model": np.array(
                [{"ground_truth": " ".join(words)} for _ in range(4)], dtype=object
            ),
            "data_source": np.array(sources, dtype=object),
        },
    )
    manager = LongAudioGroupedRewardManager(
        tokenizer=Tokenizer(),
        num_examine=0,
        compute_score=compute_score,
        version=config.val_reward.reward_kwargs.version,
    )
    result = manager(data, return_dict=True)

    assert len(calls) == 2
    assert [call["data_source"] for call in calls] == [sources[0], sources[2]]
    assert all(call["solution_str"] == "\n".join(words) for call in calls)
    assert all(call["extra_info"]["n_segments"] == 2 for call in calls)
    assert result["reward_extra_info"]["wer"] == [0.0, None, 0.0, None]
    assert result["reward_extra_info"]["n_err"] == [0, None, 0, None]
    assert result["reward_extra_info"]["n_ref"] == [2, None, 2, None]
    assert torch.all(result["reward_tensor"] == 1.0)


@pytest.mark.parametrize(
    "config_path",
    sorted((CONFIG_ROOT / "data/val_data/inhouse/indian").glob("*.yaml")),
    ids=lambda path: path.stem,
)
def test_indian_inhouse_datasets_enable_language_hints(config_path):
    entries = OmegaConf.load(config_path)

    assert entries
    for entry in entries:
        assert entry.add_task_info.task == "lang_asr"
        assert entry.add_task_info.language
        assert entry.add_task_info.lang_hint is True


def test_r2b_in_matches_india_benchmark_datasets_and_r2b_checkpoint():
    with initialize_config_dir(config_dir=str(EVAL_CONFIG_DIR), version_base=None):
        config = compose(config_name="eval_2609r2b_in")
        original = compose(config_name="eval_2609r2b_openall_zip")

    expected_corpora = {
        "en-IN": [
            "Conversation_DTEST_FY23Q3_en-IN_DTEST",
            "Conversation_OnlineMeetings_DTEST_FY24Q2_en-IN",
            "Dictation_DTEST_L_D_FY23Q2_en-IN_DTEST",
        ],
        "hi-IN": [
            "Conversation_DTEST_FY23Q3_hi-IN_DTEST",
            "Conversation_OnlineMeetings_DTEST_FY24Q2_hi-IN",
            "Dictation_DTEST_L_D_FY23Q2_hi-IN_DTEST",
        ],
        "as-IN": ["Dictation_DisplayGPT"],
        "bn-IN": ["DeviceCommands_DisplayGPT", "Dictation_DisplayGPT"],
        "gu-IN": ["DeviceCommands_DisplayGPT", "Dictation_DisplayGPT"],
        "kn-IN": ["DeviceCommands_DisplayGPT", "Dictation_DisplayGPT"],
        "ml-IN": ["DeviceCommands_DisplayGPT", "Dictation_DisplayGPT"],
        "mr-IN": ["DeviceCommands_DisplayGPT"],
        "or-IN": ["Dictation_DisplayGPT"],
        "pa-IN": ["Dictation_DisplayGPT"],
        "ta-IN": ["DeviceCommands_DisplayGPT", "Dictation_DisplayGPT"],
        "te-IN": ["DeviceCommandsDisplayGPT_Clean", "DictationDisplayGPT_Clean"],
        "ur-IN": ["Dictation_DisplayGPT"],
    }
    groups = OmegaConf.to_container(config.data.val_data, resolve=True)
    assert OmegaConf.to_container(config.data.train_data, resolve=True) == groups
    assert len(groups) == len(expected_corpora) == 13
    assert sum(map(len, groups)) == 23
    sources = []
    for group, (locale, corpora) in zip(groups, expected_corpora.items(), strict=True):
        assert [row["jsonl_paths"].split("/")[-2] for row in group] == corpora
        code = locale.replace("-", "").lower()
        filename = f"{code}_ter"
        original_rows = OmegaConf.to_container(
            OmegaConf.load(CONFIG_ROOT / f"data/val_data/inhouse/indian/{filename}.yaml"),
            resolve=True,
        )
        assert group == original_rows
        for row in group:
            assert row["jsonl_paths"].split("/")[-3] == locale
            source = row["post_process"]["add_field"]["fields"]["data_source"]
            assert source.startswith(f"{code}_")
            sources.append(source)
            registration = config.val_reward.reward_function_by_data_source[source]
            scorer = config.val_reward.reward_functions[registration]
            assert scorer.name == "eval_score"
            assert scorer.path == "recipe/phimm/reward/asr_inhouse_measure.py"
            if locale == "hi-IN":
                assert registration == "inhouse_all"
            else:
                assert scorer.reward_kwargs.locale == locale.lower()
            if locale in ("en-IN", "hi-IN"):
                assert "svad_explode" not in row["pre_process"]
                assert "/InhouseASR_2605_seg_presegment/" in row["jsonl_paths"]
                segmented_rows = OmegaConf.to_container(
                    OmegaConf.load(CONFIG_ROOT / f"data/val_data/inhouse/indian/{code}_seg.yaml"),
                    resolve=True,
                )
                expected_segment = next(
                    entry for entry in segmented_rows if entry["jsonl_paths"] == row["jsonl_paths"]
                )
                assert {key: value for key, value in row.items() if key != "val_batch_size"} == expected_segment
            else:
                assert row["pre_process"]["svad_explode"]["max_len_sec"] == 40.0
            if locale in ("en-IN", "hi-IN") or (
                locale == "bn-IN" and row["jsonl_paths"].split("/")[-2] == "Dictation_DisplayGPT"
            ):
                assert row["val_batch_size"] == -1
            assert "parent_audio_path" in row["post_process"]["verl_format"]["extra_keys"]
    assert len(set(sources)) == 23
    assert config.actor_rollout_ref.model.path == original.actor_rollout_ref.model.path
    assert config.trainer.experiment_name == "v2609r2b_in"
    assert config.trainer.nnodes == 1
    assert config.trainer.val_only is True
    assert config.trainer.val_before_train is True


def test_r2b_in_dispatches_each_dataset_to_its_ter_locale(monkeypatch):
    from verl.trainer.ppo.reward import get_reward_fn_dispatcher

    with initialize_config_dir(config_dir=str(EVAL_CONFIG_DIR), version_base=None):
        config = compose(config_name="eval_2609r2b_in")
    scorer = get_reward_fn_dispatcher(config.val_reward)
    path = str(Path(__file__).parents[3] / "recipe/phimm/reward/asr_inhouse_measure.py")
    module = sys.modules[f"custom_reward_module::{path}"]
    locales = []

    def compute_dter(ref, hyp, locale):
        locales.append(locale)
        return 0, 1, 0.0, None

    monkeypatch.setattr(module, "ensure_pack_dir", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_compute_dter", compute_dter)
    for group in config.data.val_data:
        for row in group:
            source = row.post_process.add_field.fields.data_source
            result = scorer(
                source, "<TXT>hello</TXT>", "hello",
                extra_info={"language": row.add_task_info.language},
            )
            assert result["score"] == 1.0
            assert locales[-1] == row.jsonl_paths.split("/")[-3].lower()


def test_in1_selects_first_ter_dataset_from_each_locale():
    with initialize_config_dir(config_dir=str(EVAL_CONFIG_DIR), version_base=None):
        config = compose(config_name="in1")
        full = compose(config_name="eval_2609r2b_in")

    groups = OmegaConf.to_container(config.data.val_data, resolve=True)
    full_groups = OmegaConf.to_container(full.data.val_data, resolve=True)
    assert groups == [[group[0]] for group in full_groups]
    assert len(groups) == 13
    assert len({group[0]["jsonl_paths"].split("/")[-3] for group in groups}) == 13
    assert OmegaConf.to_container(config.data.train_data, resolve=True) == groups
    assert config.trainer.experiment_name == "v2609r2b_in1"
    assert config.trainer.experiment_name != full.trainer.experiment_name
    assert config.trainer.val_only is True
    assert config.trainer.nnodes == full.trainer.nnodes == 1
    assert config.actor_rollout_ref.model.path == full.actor_rollout_ref.model.path
    assert OmegaConf.to_container(config.val_reward, resolve=True) == OmegaConf.to_container(
        full.val_reward, resolve=True
    )


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

    if config_name not in ("eval_base", "eval_2609r2b_in"):
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

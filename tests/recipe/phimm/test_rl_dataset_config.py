import ast
import logging
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import datasets
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict


def test_ls_only_word_reward_recipe_preserves_donor_settings():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    donor_name = "remax_2609r2_earning_ml0_ls_bad_verb_hint_e2_bs64_n4_r256_g16_ds1e_sfl_avg6_flr_rm0b_v0a"
    recipe_name = "remax_2609r2_ls_bad_e1_g16_n4_rw0"
    with initialize_config_dir(config_dir=str(config_root / "v2609_asr"), version_base=None):
        donor = compose(config_name=donor_name, overrides=[searchpath])
        config = compose(config_name=recipe_name, overrides=[searchpath])

    assert config.trainer.total_epochs == 1
    donor.trainer.total_epochs = 1
    for section in ("actor_rollout_ref", "algorithm", "trainer", "val_reward"):
        assert OmegaConf.to_container(config[section], resolve=False) == OmegaConf.to_container(
            donor[section], resolve=False
        )
    assert OmegaConf.to_container(config.data.data_source) == {
        "ls_rare_verb": {"num_epoch": 1, "shuffle": True},
    }
    assert OmegaConf.to_container(config.data.train_data, resolve=True) == [OmegaConf.to_container(donor._ls_train)]
    donor.data.data_source = config.data.data_source
    donor.data.train_data = config.data.train_data
    assert OmegaConf.to_container(config.data) == OmegaConf.to_container(donor.data)
    assert config.data.train_data[0].pre_process.remove_brackets == {}
    assert OmegaConf.to_container(config.reward_function_by_data_source) == {"ls_rare_verb": "librispeech"}
    reward = config.reward_functions.librispeech
    assert reward.path == donor.reward_functions.librispeech.path
    assert reward.name == "compute_score"
    assert reward.reward_kwargs.version == 2609
    assert OmegaConf.to_container(reward.reward_kwargs.reduce) == {"mode": "average", "total": 4}
    assert OmegaConf.to_container(reward.reward_kwargs.measures) == {
        "char": {"beta": 1.0},
        "word": {"beta": 1.0},
        "word_kw": {"beta": 10.0},
        "word_del": {"beta": 2.0},
        "word_ins": {"beta": 2.0},
        "word_lang": {"beta": 5.0},
        "word_fmt": {"beta": 5.0},
    }


@pytest.mark.parametrize(
    "recipe_name, masked_recipe_name",
    [
        (
            "remax_2609v0_ml_hint_s200_bs128_n8_flr_g16",
            "remax_2609v0_ml_hint_s200_bs128_n8_flr_g16_am",
        ),
        (
            "remax_2609v0_earning_verb_hint_s1k_bs128_n8_r256_g32_flr",
            "remax_2609v0_earning_verb_hint_s1k_bs128_n8_r256_g16_flr_am",
        ),
        (
            "remax_2609v0_earning_verb_hint_s400_bs128_n8_r256_g16_flr",
            "remax_2609v0_earning_verb_hint_s400_bs128_n8_r256_g16_flr_am",
        ),
    ],
)
def test_greedy_advantage_mask_recipe_preserves_reference(recipe_name, masked_recipe_name):
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2609_asr"), version_base=None):
        base = compose(config_name=recipe_name, overrides=[searchpath])
        masked = compose(config_name=masked_recipe_name, overrides=[searchpath])

    assert base.algorithm.remax_advantage_mask is None
    assert masked.algorithm.remax_advantage_mask == "edit_boundary"
    assert masked.actor_rollout_ref.rollout.gt_rollout
    assert masked.actor_rollout_ref.rollout.n == 8
    assert masked.trainer.save_freq == 50
    assert masked.trainer.nnodes * masked.trainer.n_gpus_per_node == 16
    base.algorithm.remax_advantage_mask = "edit_boundary"
    base.trainer.nnodes = 2
    assert OmegaConf.to_container(masked) == OmegaConf.to_container(base)


def test_r2_smp8_avg4_sfl_changes_only_global_shuffle():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    recipe_name = "remax_2609r2_earning_ml_verb_hint_s400_bs128_n8_r256_g32_smp8_avg4"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2609_asr"), version_base=None):
        original = compose(config_name=recipe_name, overrides=[searchpath])
        shuffled = compose(config_name=f"{recipe_name}_sfl", overrides=[searchpath])

    assert original.data.shuffle is False
    assert shuffled.data.shuffle is True
    shuffled.data.shuffle = False
    assert OmegaConf.to_container(shuffled, resolve=False) == OmegaConf.to_container(original, resolve=False)


def test_existing_edit_mask_recipe_uses_advantage_mask():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = (
        f"hydra.searchpath=[file://{config_root},file://{config_root / 'base'},"
        f"file://{root / 'verl/trainer/config'}]"
    )
    with initialize_config_dir(config_dir=str(config_root / "remax"), version_base=None):
        config = compose(
            config_name="remax_r2_punc_p0_7_n12_s1k_mean_binary_adv_edit_mask",
            overrides=[searchpath],
        )
    assert config.algorithm.adv_estimator == "remax"
    assert config.algorithm.remax_advantage_mask == "edit_boundary"
    assert not config.algorithm.use_kl_in_reward
    assert config.actor_rollout_ref.actor.policy_loss.loss_mode == "vanilla"


@pytest.mark.parametrize("mode", [None, "edit_boundary", "first_diff"])
def test_400_step_recipe_accepts_string_advantage_mask_modes(mode):
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2609_asr"), version_base=None):
        config = compose(
            config_name="remax_2609v0_earning_verb_hint_s400_bs128_n8_r256_g16_flr_am",
            overrides=[searchpath, f"algorithm.remax_advantage_mask={mode or 'null'}"],
        )
    assert config.algorithm.remax_advantage_mask == mode
    assert "remax_advantage_mask_mode" not in config.algorithm
    assert config.trainer.total_training_steps == 400
    assert config.trainer.nnodes * config.trainer.n_gpus_per_node == 16


@pytest.mark.parametrize(
    "source_ratio,earnings_samples,total_steps,steps_per_pass",
    [("021", 0, 616, 308), ("121", 13169, 822, 411)],
)
def test_ml0_ls_recipe_preserves_training_settings_and_weights_sources(
    source_sampler, source_ratio, earnings_samples, total_steps, steps_per_pass
):
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    donor_name = "remax_2609r2_earning_ml0_verb_hint_s306_bs128_n16_r128_g32_ds0_sfl_avg4_flr1"
    recipe_name = (
        f"remax_2609r2_earning_ml0_ls_verb_hint_s{total_steps}_bs128_n16_r256_g32_ds{source_ratio}_sfl_avg4_flr1"
    )
    original_name = "remax_2609r2_earning_ml_ls_verb_hint_s753_bs128_n8_r256_g32_ds1s_sfl_avg4_edit_smts_lr0"
    with initialize_config_dir(config_dir=str(config_root / "v2609_asr"), version_base=None):
        donor = compose(config_name=donor_name, overrides=[searchpath])
        config = compose(config_name=recipe_name, overrides=[searchpath])
        original = compose(config_name=original_name, overrides=[searchpath])

    assert config.actor_rollout_ref.model.lora_rank == 256
    donor.actor_rollout_ref.model.lora_rank = 256
    assert config.trainer.total_training_steps == total_steps
    assert config.trainer.total_epochs == 2
    donor.trainer.total_training_steps = total_steps
    donor.trainer.total_epochs = 2
    for section in ("actor_rollout_ref", "algorithm", "trainer"):
        assert OmegaConf.to_container(config[section], resolve=False) == OmegaConf.to_container(
            donor[section], resolve=False
        )
    assert config.data.train_batch_size == donor.data.train_batch_size == 128
    assert config.data.shuffle == donor.data.shuffle is True
    assert OmegaConf.to_container(config.data.data_source) == {
        "earnings_fy27": {"num_sample": earnings_samples, "shuffle": True},
        "openml": {"num_sample": 26338, "shuffle": True},
        "ls_rare_verb": {"num_sample": 13169, "shuffle": True},
    }
    assert OmegaConf.to_container(config._ml_train) == OmegaConf.to_container(donor._ml_train)
    assert len(config._ml_train[0].jsonl_paths) == 2
    assert OmegaConf.to_container(config._ls_train) == OmegaConf.to_container(original._ls_train)
    assert len(config.data.train_data) == len(config.data.val_data) == 3
    assert OmegaConf.to_container(config.reward_functions) == OmegaConf.to_container(original.reward_functions)
    assert OmegaConf.to_container(config.reward_function_by_data_source) == OmegaConf.to_container(
        original.reward_function_by_data_source
    )
    for name in ("_earning_val", "_ml_val", "_ls_val"):
        assert OmegaConf.to_container(config[name]) == OmegaConf.to_container(original[name])
    inputs = [
        datasets.Dataset.from_dict({"id": [source], "data_source": [source]})
        for source in config.data.data_source
    ]
    sampled = source_sampler(inputs, config.data.data_source)
    expected_sources = (["earnings_fy27"] if earnings_samples else []) + ["openml", "ls_rare_verb"]
    expected_samples = ([earnings_samples] if earnings_samples else []) + [26338, 13169]
    assert [group[0]["data_source"] for group in sampled] == expected_sources
    assert [len(group) for group in sampled] == expected_samples
    assert sum(len(group) for group in sampled) // config.data.train_batch_size == steps_per_pass
    assert config.trainer.total_training_steps == 2 * steps_per_pass


@pytest.mark.parametrize(
    "recipe_name,openml_samples,ls_samples,total_steps,shuffle",
    [
        (
            "remax_2609r2_earning_ml0_ls_bad_verb_hint_s308_bs128_n16_r256_g32_ds0105_sfl_avg4_flr1",
            13169,
            6584,
            308,
            True,
        ),
        (
            "remax_2609r2_earning_ml0_ls_bad_verb_hint_s616_bs128_n16_r256_g32_ds021_avg4_flr1",
            26338,
            13169,
            616,
            False,
        ),
        (
            "remax_2609r2_earning_ml0_ls_bad_verb_hint_s616_bs128_n16_r256_g32_ds021_sfl_avg4_flr1",
            26338,
            13169,
            616,
            True,
        ),
    ],
)
def test_ml0_ls_bad_recipes_change_only_ls_manifest_preprocessing_budgets_and_shuffle(
    source_sampler, recipe_name, openml_samples, ls_samples, total_steps, shuffle
):
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    original_name = "remax_2609r2_earning_ml0_ls_verb_hint_s616_bs128_n16_r256_g32_ds021_sfl_avg4_flr1"
    with initialize_config_dir(config_dir=str(config_root / "v2609_asr"), version_base=None):
        original = compose(config_name=original_name, overrides=[searchpath])
        config = compose(config_name=recipe_name, overrides=[searchpath])

    bad_manifest = (
        "az://orngwus2cresco/data/boren/data/verl/gen_qwen/2607v1a_earning_step650/"
        "ls_rare_nonempty_verb_g32_nonzero_n_err.jsonl"
    )
    data_config = OmegaConf.load(config_root / "data/train_data/ls_rare_bad_verb_langhint.yaml")
    recipe = OmegaConf.load(config_root / "v2609_asr" / f"{recipe_name}.yaml")
    assert "/data/train_data/ls_rare_bad_verb_langhint@_ls_train" in recipe.defaults
    assert "_ls_train" not in recipe
    assert OmegaConf.to_container(config._ls_train) == OmegaConf.to_container(data_config)
    assert config._ls_train.jsonl_paths == bad_manifest
    assert config._ls_train.pre_process is None
    assert config.data.train_data[2].jsonl_paths == bad_manifest
    assert config.data.train_data[2].pre_process is None
    assert config._ls_train.add_task_info.lang_hint is True
    source = config._ls_train.post_process.add_field.fields.data_source
    assert source == "ls_rare_verb"
    assert config.reward_function_by_data_source[source] == "librispeech"
    assert OmegaConf.to_container(config.data.data_source) == {
        "earnings_fy27": {"num_sample": 0, "shuffle": True},
        "openml": {"num_sample": openml_samples, "shuffle": True},
        "ls_rare_verb": {"num_sample": ls_samples, "shuffle": True},
    }
    assert config.data.shuffle is shuffle
    assert config.data.use_interleave is False
    assert config.data.data_source.ls_rare_verb.num_sample == math.floor(
        config.data.data_source.openml.num_sample * 0.5
    )
    inputs = [
        datasets.Dataset.from_dict({"id": [name], "data_source": [name]})
        for name in config.data.data_source
    ]
    sampled = source_sampler(inputs, config.data.data_source)
    assert [group[0]["data_source"] for group in sampled] == ["openml", "ls_rare_verb"]
    assert [len(group) for group in sampled] == [openml_samples, ls_samples]
    combined = datasets.concatenate_datasets(sampled)
    assert list(combined["data_source"]) == ["openml"] * openml_samples + ["ls_rare_verb"] * ls_samples
    steps_per_pass = sum(len(group) for group in sampled) // config.data.train_batch_size
    assert steps_per_pass == total_steps // 2
    assert config.trainer.total_epochs == 2
    assert config.trainer.total_training_steps == 2 * steps_per_pass == total_steps

    config._ls_train.jsonl_paths = original._ls_train.jsonl_paths
    config._ls_train.pre_process = original._ls_train.pre_process
    config.data.data_source = original.data.data_source
    config.data.shuffle = original.data.shuffle
    config.trainer.total_training_steps = original.trainer.total_training_steps
    assert OmegaConf.to_container(config, resolve=False) == OmegaConf.to_container(original, resolve=False)


def _load_flatten_data_confs():
    module_path = Path(__file__).parents[3] / "recipe/phimm/data/rl_dataset.py"
    module = ast.parse(module_path.read_text())
    function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_flatten_data_confs"
    )
    namespace = {"Sequence": Sequence}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(module_path), "exec"), namespace)
    return namespace["_flatten_data_confs"]


def _load_audio_retry_helper():
    module_path = Path(__file__).parents[3] / "recipe/phimm/data/rl_dataset.py"
    module = ast.parse(module_path.read_text())
    function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_load_audio_with_retries"
    )
    namespace = {"logger": SimpleNamespace(warning=lambda *args, **kwargs: None)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(module_path), "exec"), namespace)
    return namespace["_load_audio_with_retries"]


def test_flatten_data_confs_expands_nested_omegaconf_groups():
    flatten_data_confs = _load_flatten_data_confs()
    grouped = OmegaConf.create(
        [
            [{"dataset_name": "jsonl", "cache_name": "first"}, {"dataset_name": "jsonl", "cache_name": "second"}],
            [{"dataset_name": "jsonl", "cache_name": "third"}],
        ]
    )

    flattened = flatten_data_confs(grouped)

    assert [config.cache_name for config in flattened] == ["first", "second", "third"]


def test_flatten_data_confs_preserves_flat_and_scalar_inputs():
    flatten_data_confs = _load_flatten_data_confs()
    flat = OmegaConf.create([{"dataset_name": "jsonl"}])

    assert flatten_data_confs(flat) == list(flat)
    assert flatten_data_confs("data.jsonl") == ["data.jsonl"]


@pytest.mark.parametrize("is_train", [False, True])
def test_dataset_loader_preserves_options_and_overrides_version(
    dataset_class, dataset_namespace, monkeypatch, is_train
):
    configs = OmegaConf.create([{"dataset_name": "jsonl", "val_batch_size": -1, "version": 2607}])
    calls = []

    def create_dataset(**kwargs):
        calls.append(kwargs)
        return datasets.Dataset.from_dict({"id": [0], "data_source": ["test"]})

    monkeypatch.setitem(dataset_namespace, "create_audio_dataset", create_dataset)
    dataset_class(configs, None, OmegaConf.create({"num_proc": None, "version": 2609}), is_train=is_train)
    assert calls == [{"dataset_name": "jsonl", "val_batch_size": -1, "version": 2609}]
    assert configs[0].val_batch_size == -1


def test_2609_validation_configs_use_full_batches_only_for_parent_audio():
    root = Path(__file__).parents[3] / "recipe/phimm/config"
    references = set()
    for path in (root / "v2609_asr").glob("*.yaml"):
        config = OmegaConf.load(path)
        for item in config.get("defaults", []):
            if isinstance(item, str) and item.startswith("/data/val_data/"):
                references.add(item.split("@")[0].lstrip("/"))

    parent_count = short_count = 0
    for reference in references:
        for item in _load_flatten_data_confs()(OmegaConf.load(root / f"{reference}.yaml")):
            extra_keys = item.get("post_process", {}).get("verl_format", {}).get("extra_keys", [])
            if "parent_audio_path" in extra_keys:
                parent_count += 1
                assert item.val_batch_size == -1, reference
            else:
                short_count += 1
                assert "val_batch_size" not in item, reference
    assert parent_count > 0 and short_count > 0
    assert OmegaConf.load(root / "v2609_asr/base.yaml").data.val_batch_size == 256


@pytest.fixture
def dataset_namespace():
    module_path = Path(__file__).parents[3] / "recipe/phimm/data/rl_dataset.py"
    module = ast.parse(module_path.read_text())
    definitions = [
        node for node in module.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name != "main"
    ]
    namespace = {
        "Dataset": object,
        "Sequence": Sequence,
        "Mapping": Mapping,
        "math": math,
        "random": random,
        "datasets": datasets,
        "logger": logging.getLogger(__name__),
        "get_num_proc": lambda value: value,
        "set_chunk_load_mode": lambda value: None,
    }
    code = ast.Module(body=ast.parse("from __future__ import annotations").body + definitions, type_ignores=[])
    exec(compile(code, str(module_path), "exec"), namespace)
    from recipe.phimm.data.dataset import create_audio_dataset

    namespace["create_audio_dataset"] = create_audio_dataset
    random_state = random.getstate()
    random.seed(42)
    try:
        yield namespace
    finally:
        random.setstate(random_state)


@pytest.fixture
def dataset_class(dataset_namespace):
    return dataset_namespace["RLHFDataset"]


@pytest.fixture
def source_sampler(dataset_namespace):
    return dataset_namespace["group_datasets"]


@pytest.fixture
def cached_sources(tmp_path):
    paths = []
    for name, source, count in (("first", "earning", 3), ("other", "openml", 2), ("second", "earning", 2)):
        path = tmp_path / name
        datasets.Dataset.from_dict(
            {"id": [f"{name}-{i}" for i in range(count)], "data_source": [source] * count}
        ).save_to_disk(str(path))
        paths.append(str(path))
    return [{"dataset_name": "cached", "cache_path": path} for path in paths]


def test_collect_datasources_returns_combined_datasets_by_name(dataset_namespace):
    first = datasets.Dataset.from_dict({"id": ["a", "b"], "data_source": ["earning"] * 2})
    other = datasets.Dataset.from_dict({"id": ["c"], "data_source": ["openml"]})
    second = datasets.Dataset.from_dict({"id": ["d"], "data_source": ["earning"]})
    collected = dataset_namespace["_collect_datasources"]([first, other, second])

    assert isinstance(collected, dict)
    assert list(collected) == ["earning", "openml"]
    assert all(isinstance(ds, datasets.Dataset) for ds in collected.values())
    assert list(collected["earning"]["id"]) == ["a", "b", "d"]
    assert collected["openml"] is other
    assert [len(ds) for ds in (first, other, second)] == [2, 1, 1]


def test_collect_datasources_skips_empty_inputs(dataset_namespace, caplog):
    empty = datasets.Dataset.from_dict({"id": [], "data_source": []})
    other = datasets.Dataset.from_dict({"id": ["a"], "data_source": ["openml"]})
    with caplog.at_level(logging.WARNING):
        collected = dataset_namespace["_collect_datasources"]([empty, other])

    assert collected == {"openml": other}
    assert "Skipping empty training dataset 0" in caplog.text


def test_dataset_budgets_combined_datasources_and_preserves_caches(dataset_class, cached_sources):
    config = OmegaConf.create(
        {
            "use_interleave": False,
            "num_proc": None,
            "data_source": {"earning": {"num_epoch": 2}, "openml": {"num_sample": 3}},
        }
    )
    dataset = dataset_class(
        OmegaConf.create([[cached_sources[0]], cached_sources[1:]]),
        None,
        config,
    )
    earning = ["first-0", "first-1", "first-2", "second-0", "second-1"]
    assert list(dataset.ds["id"]) == earning * 2 + ["other-0", "other-1", "other-0"]
    assert Counter(dataset.ds["data_source"]) == {"earning": 10, "openml": 3}
    config.data_source.earning.num_epoch = 0.5
    resized = dataset_class(cached_sources, None, config)
    resized_ids = list(resized.ds["id"])
    assert len(resized_ids) == 5
    assert len(set(resized_ids[:2])) == 2
    assert set(resized_ids[:2]) <= set(earning)
    assert resized_ids[2:4] == ["other-0", "other-1"]
    assert resized_ids[4] in {"other-0", "other-1"}
    assert [len(datasets.Dataset.load_from_disk(conf["cache_path"])) for conf in cached_sources] == [3, 2, 2]


@pytest.mark.parametrize(
    ("dataset_order", "source_order"),
    [
        ([0, 1, 2], ["openml", "earning"]),
        ([1, 0, 2], ["openml", "earning"]),
    ],
)
def test_datasources_stay_serial_in_configured_key_order(dataset_class, cached_sources, dataset_order, source_order):
    config = OmegaConf.create(
        {
            "use_interleave": False,
            "shuffle": False,
            "num_proc": None,
            "data_source": {"openml": {"num_sample": 3}, "earning": {"num_epoch": 2}},
        }
    )
    dataset = dataset_class([cached_sources[i] for i in dataset_order], None, config)
    blocks = {
        "earning": ["first-0", "first-1", "first-2", "second-0", "second-1"] * 2,
        "openml": ["other-0", "other-1", "other-0"],
    }
    assert list(dataset.ds["id"]) == [row for source in source_order for row in blocks[source]]
    assert list(dataset.ds["data_source"]) == [source for source in source_order for _ in blocks[source]]


@pytest.mark.parametrize(
    "interleave_options",
    [
        {},
        {"stopping_strategy": "all_exhausted"},
        {"stopping_strategy": "all_exhausted", "probabilities": [0.3, 0.7], "seed": 7},
    ],
)
@pytest.mark.parametrize("source_order", [("earning", "openml"), ("openml", "earning")])
def test_interleaving_uses_weighted_datasource_groups(dataset_class, cached_sources, interleave_options, source_order):
    budgets = {"earning": {"num_epoch": 2}, "openml": {"num_sample": 3}}
    config = OmegaConf.create(
        {
            "use_interleave": True,
            "num_proc": None,
            "interleave_ds": interleave_options,
            "data_source": {source: budgets[source] for source in source_order},
        }
    )
    dataset = dataset_class(cached_sources, None, config)
    earning = datasets.concatenate_datasets(
        [datasets.Dataset.load_from_disk(cached_sources[i]["cache_path"]) for i in [0, 2, 0, 2]]
    )
    openml = datasets.Dataset.load_from_disk(cached_sources[1]["cache_path"]).select([0, 1, 0])
    groups = {"earning": earning, "openml": openml}
    expected = datasets.interleave_datasets(
        [groups[source] for source in source_order],
        **interleave_options,
    )
    assert dataset.ds.to_dict() == expected.to_dict()


def test_validation_and_generation_ignore_training_budgets(dataset_class, cached_sources):
    dataset = dataset_class(
        cached_sources,
        None,
        OmegaConf.create(
            {
                "use_interleave": True,
                "num_proc": None,
                "data_source": {"earning": {"num_epoch": 0}, "absent_from_validation": {"num_sample": 20}},
            }
        ),
        is_train=False,
    )
    assert list(dataset.ds["id"]) == ["first-0", "first-1", "first-2", "other-0", "other-1", "second-0", "second-1"]


def test_mix_orders_groups_and_combines_matching_sources(dataset_class, cached_sources):
    config = OmegaConf.create(
        {
            "num_proc": None,
            "shuffle": False,
            "use_interleave": True,
            "interleave_ds": {"probabilities": [1]},
            "train_batch_size": 2,
            "mix": [
                {"datasource": [{"name": "earning", "sample": 7}, {"name": "openml", "batch": 2}]},
                {"datasource": [{"name": "earning", "epoch": 0.5}]},
            ],
        }
    )
    original = OmegaConf.to_container(config)
    dataset = dataset_class(cached_sources, None, config)
    earning = ["first-0", "first-1", "first-2", "second-0", "second-1"]
    assert list(dataset.ds["id"]) == earning + earning[:2] + ["other-0", "other-1"] * 2 + earning[:2]
    assert Counter(dataset.ds["data_source"]) == {"earning": 9, "openml": 4}
    assert OmegaConf.to_container(config) == original
    assert [len(datasets.Dataset.load_from_disk(conf["cache_path"])) for conf in cached_sources] == [3, 2, 2]


@pytest.mark.parametrize("seed", [None, 7])
def test_mix_shuffle_is_seeded_and_stays_within_groups(dataset_class, cached_sources, seed):
    config = OmegaConf.create(
        {
            "num_proc": None,
            "seed": seed,
            "mix": [
                {
                    "shuffle": True,
                    "datasource": [{"name": "earning", "shuffle": True, "sample": 3}, {"name": "openml"}],
                },
                {"datasource": [{"name": "earning", "sample": 2}]},
            ],
        }
    )
    dataset = dataset_class(cached_sources, None, config)
    effective_seed = 42 if seed is None else seed
    earning = datasets.concatenate_datasets(
        [datasets.Dataset.load_from_disk(cached_sources[i]["cache_path"]) for i in (0, 2)]
    )
    openml = datasets.Dataset.load_from_disk(cached_sources[1]["cache_path"])
    first = datasets.concatenate_datasets(
        [earning.shuffle(seed=effective_seed).select(range(3)), openml]
    ).shuffle(seed=effective_seed)
    expected = datasets.concatenate_datasets([first, earning.select(range(2))])
    assert dataset.ds.to_dict() == expected.to_dict()
    assert dataset_class(cached_sources, None, config).ds.to_dict() == expected.to_dict()


@pytest.mark.parametrize("budget", [{"sample": 12}, {"sample": 12.0}, {"batch": 6.0}, {"epoch": 2.4}])
def test_mix_budgets_repeat_passes_after_source_shuffle(dataset_class, cached_sources, budget):
    config = OmegaConf.create(
        {
            "num_proc": None,
            "train_batch_size": 2,
            "mix": [{"datasource": [{"name": "earning", "shuffle": True, **budget}]}],
        }
    )
    dataset = dataset_class(cached_sources, None, config)
    source = datasets.concatenate_datasets(
        [datasets.Dataset.load_from_disk(cached_sources[i]["cache_path"]) for i in (0, 2)]
    ).shuffle(seed=42)
    assert list(dataset.ds["id"]) == list(source["id"]) * 2 + list(source["id"])[:2]


@pytest.mark.parametrize(
    ("budget", "expected_count"),
    [
        ({"sample": 2.4}, 2),
        ({"sample": 2.5}, 2),
        ({"sample": 2.6}, 3),
        ({"sample": 3.5}, 4),
        ({"batch": 0.5}, 1),
        ({"batch": 1.2}, 2),
        ({"batch": 1.3}, 3),
        ({"epoch": 0.5}, 2),
        ({"epoch": 0.75}, 4),
        ({"epoch": 1.5}, 8),
    ],
)
def test_mix_rounds_fractional_budgets_after_scaling(dataset_class, cached_sources, budget, expected_count):
    config = OmegaConf.create(
        {
            "num_proc": None,
            "train_batch_size": 2,
            "mix": [{"datasource": [{"name": "earning", **budget}]}],
        }
    )
    dataset = dataset_class(cached_sources, None, config)
    earning = ["first-0", "first-1", "first-2", "second-0", "second-1"]
    assert len(dataset) == expected_count
    assert list(dataset.ds["id"]) == (earning * 2)[:expected_count]


@pytest.mark.parametrize("budget", [{"sample": 0.4}, {"batch": 0.2}, {"epoch": 0.09}])
def test_mix_fractional_budgets_can_round_to_zero(dataset_class, cached_sources, budget):
    config = OmegaConf.create(
        {
            "num_proc": None,
            "train_batch_size": 2,
            "mix": [{"datasource": [{"name": "earning", **budget}, {"name": "openml"}]}],
        }
    )
    dataset = dataset_class(cached_sources, None, config)
    assert list(dataset.ds["id"]) == ["other-0", "other-1"]


@pytest.mark.parametrize("budget", [{"batch": 1e308}, {"epoch": 1e308}])
def test_mix_rejects_non_finite_scaled_counts(dataset_class, cached_sources, budget):
    config = OmegaConf.create(
        {
            "num_proc": None,
            "train_batch_size": 2,
            "mix": [{"datasource": [{"name": "earning", **budget}]}],
        }
    )
    with pytest.raises(ValueError, match="produces a non-finite row count"):
        dataset_class(cached_sources, None, config)


def test_mix_omits_unlisted_and_zero_sources_and_warns_for_empty_groups(dataset_class, cached_sources, caplog):
    config = OmegaConf.create(
        {
            "num_proc": None,
            "mix": [
                {"datasource": [{"name": "earning", "epoch": 0.1}]},
                {"datasource": [{"name": "openml", "sample": None}]},
            ],
        }
    )
    with caplog.at_level(logging.WARNING):
        dataset = dataset_class(cached_sources, None, config)
    assert list(dataset.ds["id"]) == ["other-0", "other-1"]
    assert "Skipping mix group 0" in caplog.text
    config.mix[1].datasource[0].sample = 0
    with pytest.raises(ValueError, match="No samples remain"):
        dataset_class(cached_sources, None, config)


@pytest.mark.parametrize(
    ("mix", "message"),
    [
        ([], "non-empty list of groups"),
        ({}, "non-empty list of groups"),
        ("earning", "non-empty list of groups"),
        ([None], "must contain datasource"),
        ([{"datasource": [], "unexpected": 1}], "must contain datasource"),
        ([{}], "datasource must be a non-empty list"),
        ([{"datasource": "earning"}], "datasource must be a non-empty list"),
        ([{"datasource": {}}], "datasource must be a non-empty list"),
        ([{"datasource": [{"name": "earning"}], "shuffle": "true"}], "shuffle must be a boolean"),
        ([{"datasource": [None]}], "accepts name"),
        ([{"datasource": [{"name": "missing"}]}], "no non-empty training datasource"),
        ([{"datasource": [{"name": ["earning"]}]}], "no non-empty training datasource"),
        ([{"datasource": [{"name": "earning", "samples": 2}]}], "accepts name"),
        ([{"datasource": [{"name": "earning", "shuffle": 1}]}], "shuffle must be a boolean"),
        ([{"datasource": [{"name": "earning", "sample": 2, "epoch": 1}]}], "only one"),
    ],
)
def test_mix_rejects_invalid_structure(dataset_class, cached_sources, mix, message):
    with pytest.raises(ValueError, match=message):
        dataset_class(cached_sources, None, OmegaConf.create({"num_proc": None, "mix": mix}))


@pytest.mark.parametrize(
    "budget",
    [
        {"sample": -1},
        {"sample": True},
        {"sample": "1"},
        {"sample": float("nan")},
        {"sample": float("inf")},
        {"batch": -1},
        {"batch": True},
        {"batch": "1"},
        {"batch": float("nan")},
        {"batch": float("inf")},
        {"epoch": -0.1},
        {"epoch": True},
        {"epoch": "1"},
        {"epoch": float("nan")},
        {"epoch": float("inf")},
    ],
)
def test_mix_rejects_invalid_budgets(dataset_class, cached_sources, budget):
    config = OmegaConf.create({"num_proc": None, "mix": [{"datasource": [{"name": "earning", **budget}]}]})
    with pytest.raises(ValueError, match="finite non-negative"):
        dataset_class(cached_sources, None, config)


@pytest.mark.parametrize("batch_size", [None, 0, -1, True, 1.5])
def test_mix_batch_budget_requires_valid_batch_size(dataset_class, cached_sources, batch_size):
    config = OmegaConf.create(
        {
            "num_proc": None,
            "train_batch_size": batch_size,
            "mix": [{"datasource": [{"name": "earning", "batch": 1}]}],
        }
    )
    with pytest.raises(ValueError, match="train_batch_size must be a positive integer"):
        dataset_class(cached_sources, None, config)


@pytest.mark.parametrize(
    ("extra", "message"),
    [({"shuffle": True}, "data.shuffle=false"), ({"data_source": {}}, "cannot be configured together")],
)
def test_mix_rejects_conflicting_training_options(dataset_class, cached_sources, extra, message):
    config = OmegaConf.create(
        {"num_proc": None, "mix": [{"datasource": [{"name": "earning"}]}], **extra}
    )
    with pytest.raises(ValueError, match=message):
        dataset_class(cached_sources, None, config)


def test_validation_ignores_mix_and_conflicting_training_options(dataset_class, cached_sources):
    config = OmegaConf.create({"num_proc": None, "mix": [], "shuffle": True, "data_source": {}})
    dataset = dataset_class(cached_sources, None, config, is_train=False)
    assert list(dataset.ds["id"]) == ["first-0", "first-1", "first-2", "other-0", "other-1", "second-0", "second-1"]


@pytest.mark.parametrize("use_interleave", [True, False])
def test_null_mix_preserves_legacy_loader(dataset_class, cached_sources, use_interleave):
    config = OmegaConf.create({"num_proc": None, "mix": None, "use_interleave": use_interleave})
    dataset = dataset_class(cached_sources, None, config)
    assert len(dataset) == (6 if use_interleave else 7)


def test_mix_preserves_order_through_training_sampler(dataset_class, cached_sources):
    from torch.utils.data import SequentialSampler

    root = Path(__file__).parents[3]
    module_path = root / "verl/trainer/main_ppo.py"
    module = ast.parse(module_path.read_text())
    sampler_def = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "create_rl_sampler"
    )
    namespace = {}
    exec(compile(ast.Module(body=[sampler_def], type_ignores=[]), str(module_path), "exec"), namespace)
    config = OmegaConf.create(
        {
            "num_proc": None,
            "shuffle": False,
            "sampler": None,
            "mix": [{"datasource": [{"name": "openml"}]}, {"datasource": [{"name": "earning"}]}],
        }
    )
    dataset = dataset_class(cached_sources, None, config)
    sampler = namespace["create_rl_sampler"](config, dataset)
    assert isinstance(sampler, SequentialSampler)
    assert [dataset.ds[i]["data_source"] for i in sampler] == ["openml"] * 2 + ["earning"] * 5


def test_asr_base_accepts_grouped_mix_hydra_override():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "base"), version_base=None):
        config = compose(
            config_name="dapo_asr",
            overrides=[
                searchpath,
                "data.shuffle=false",
                "data.mix=[{shuffle:true,datasource:[{name:earning,sample:128,shuffle:true},{name:openml,batch:2}]}]",
            ],
        )
    assert config.data.shuffle is False
    assert OmegaConf.to_container(config.data.mix) == [
        {
            "shuffle": True,
            "datasource": [{"name": "earning", "sample": 128, "shuffle": True}, {"name": "openml", "batch": 2}],
        }
    ]


@pytest.mark.parametrize(
    ("recipe_name", "budget_key", "first_budget", "tail_budget"),
    [
        ("remax_2609r2_mx0_bs64_n4_g16_rw0", "sample", 13168, {"ls_rare_verb": 58539}),
        ("remax_2609r2_mx0a_bs64_n4_g16_rw0", "batch", 100, {"openml": 100, "ls_rare_verb": 1000}),
    ],
)
def test_mx0_recipe_preserves_donor_and_samples_ls_without_ranges(
    dataset_namespace, dataset_class, monkeypatch, recipe_name, budget_key, first_budget, tail_budget,
):
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    donor_name = "remax_2609r2_earning_ml0_ls_bad_verb_hint_e2_bs64_n4_r256_g16_ds1e_sfl_avg6_flr_rm0b_v0a"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2609_asr"), version_base=None):
        donor = compose(config_name=donor_name, overrides=[searchpath])
        config = compose(config_name=recipe_name, overrides=[searchpath])
    assert config.trainer.total_epochs == 1
    assert config.trainer.total_training_steps is None
    donor.trainer.total_epochs = 1
    with open_dict(donor.data):
        donor.data.pop("data_source")
    donor.data.shuffle = False
    OmegaConf.update(donor, "data.mix", OmegaConf.to_container(config.data.mix), force_add=True)
    expected_measures = {
        "char": {"beta": 1.0},
        "word": {"beta": 1.0},
        "word_kw": {"beta": 2.0},
        "word_lang": {"beta": 2.0},
        "word_fmt": {"beta": 2.0},
    }
    assert OmegaConf.to_container(config.reward_functions.librispeech.reward_kwargs.measures) == expected_measures
    donor.reward_functions.librispeech.reward_kwargs.measures = expected_measures
    source = OmegaConf.load(config_root / "v2609_asr/remax_2609r2_mx0_bs64_n4_g16_rw0.yaml")
    expected_reduction = source.reward_functions.librispeech.reward_kwargs.reduce
    assert OmegaConf.to_container(expected_reduction) == {"mode": "average", "total": 2}
    assert config.reward_functions.librispeech.reward_kwargs.reduce == expected_reduction
    donor.reward_functions.librispeech.reward_kwargs.reduce = expected_reduction
    assert OmegaConf.to_container(config, resolve=False) == OmegaConf.to_container(donor, resolve=False)
    assert [entry[budget_key] for entry in config.data.mix[0].datasource] == [first_budget] * 3
    unused_budget_key = "sample" if budget_key == "batch" else "batch"
    assert all(unused_budget_key not in entry for group in config.data.mix for entry in group.datasource)
    assert "range" not in config.data.mix[0].datasource[2]
    assert all("range" not in entry for entry in config.data.mix[1].datasource)
    assert {entry.name: entry[budget_key] for entry in config.data.mix[1].datasource} == tail_budget
    factor = config.data.train_batch_size if budget_key == "batch" else 1
    first_count = first_budget * factor
    tail_counts = {name: budget * factor for name, budget in tail_budget.items()}

    counts = {"earnings_fy27": 3, "openml": 13168, "ls_rare_verb": 71707}
    sources = {
        name: datasets.Dataset.from_dict({"id": [f"{name}-{i}" for i in range(count)], "data_source": [name] * count})
        for name, count in counts.items()
    }
    monkeypatch.setitem(dataset_namespace, "create_audio_dataset", lambda source, **kwargs: sources[source])
    configs = [{"source": name} for name in sources]
    dataset = dataset_class(configs, None, config.data)
    first_size = first_count * 3
    assert len(dataset) == first_size + sum(tail_counts.values())
    assert Counter(dataset.ds[:first_size]["data_source"]) == {
        "earnings_fy27": first_count, "openml": first_count, "ls_rare_verb": first_count
    }
    assert Counter(dataset.ds[first_size:]["data_source"]) == tail_counts
    ls_first = {
        row["id"] for row in dataset.ds.select(range(first_size)) if row["data_source"] == "ls_rare_verb"
    }
    assert ls_first == set(sources["ls_rare_verb"].shuffle(seed=44).select(range(first_count))["id"])
    tail_rows = dataset.ds[first_size:]
    for index, entry in enumerate(config.data.mix[1].datasource):
        name = entry.name
        count = tail_counts[name]
        selected = {
            row_id
            for row_id, source in zip(tail_rows["id"], tail_rows["data_source"], strict=True)
            if source == name
        }
        assert selected == set(sources[name].shuffle(seed=43 + index).select(range(count))["id"])
    assert dataset_class(configs, None, config.data).ds.to_dict() == dataset.ds.to_dict()


def test_name_env37k_rw0a_recipe_preserves_ls_settings_and_routes_name_data():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2609_asr"), version_base=None):
        source = compose(config_name="remax_2609r2_ls_bad_e1_g16_n4_rw0a", overrides=[searchpath])
        variant = compose(config_name="remax_2609r2_name_env37k_e1_g16_n4_rw0a", overrides=[searchpath])

    name_data = OmegaConf.load(config_root / "data/train_data/name_env37k.yaml")
    assert variant._name_train == name_data
    assert len(variant.data.train_data) == 1
    assert variant.data.train_data[0] == name_data
    assert name_data[0].post_process.add_field.fields.data_source == "name_train"
    assert OmegaConf.to_container(variant.data.data_source) == {"name_train": {"num_epoch": 1, "shuffle": True}}
    assert OmegaConf.to_container(variant.reward_function_by_data_source) == {"name_train": "asr_measure"}
    expected_reward = OmegaConf.create(OmegaConf.to_container(source.reward_functions.librispeech))
    expected_reward.reward_kwargs.reduce.total = 2
    assert variant.reward_functions.asr_measure == expected_reward
    name_val = OmegaConf.load(config_root / "data/val_data/name_1k.yaml")
    assert variant._name_val == name_val
    assert len(variant.data.val_data) == 1
    assert variant.data.val_data[0] == name_val
    assert variant.data.val_data[0][0].pre_process.input_egs_limit == 1000
    assert name_val[0].post_process.add_field.fields.data_source == "name_val"
    assert variant.val_reward.reward_function_by_data_source.name_val == "openasr_en"
    assert variant.val_reward.reward_functions.openasr_en.name == "openasr_en_eval"

    with open_dict(source):
        source.pop("_ls_train")
        source._name_train = name_data
        for name in ("_earning_val", "_ml_val", "_ls_val"):
            source.pop(name)
        source._name_val = name_val
    source.data.train_data = variant.data.train_data
    source.data.val_data = variant.data.val_data
    source.data.data_source = variant.data.data_source
    source.reward_functions = variant.reward_functions
    source.reward_function_by_data_source = variant.reward_function_by_data_source
    with open_dict(source.val_reward.reward_function_by_data_source):
        source.val_reward.reward_function_by_data_source.name_val = "openasr_en"
    assert OmegaConf.to_container(variant, resolve=False) == OmegaConf.to_container(source, resolve=False)


def test_mx0a_rw0b_recipe_preserves_mix_and_uses_keyword_rewards_without_lang_fmt():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2609_asr"), version_base=None):
        source = compose(config_name="remax_2609r2_mx0a_bs64_n4_g16_rw0", overrides=[searchpath])
        variant = compose(config_name="remax_2609r2_mx0a_bs64_n4_g16_rw0b", overrides=[searchpath])

    assert variant.reward_function_by_data_source == source.reward_function_by_data_source
    assert variant.reward_functions.librispeech.path == source.reward_functions.librispeech.path
    assert variant.reward_functions.librispeech.name == source.reward_functions.librispeech.name
    kwargs = variant.reward_functions.librispeech.reward_kwargs
    assert kwargs.version == 2609
    assert OmegaConf.to_container(kwargs.reduce) == {"mode": "average", "total": 2}
    assert OmegaConf.to_container(kwargs.measures) == {
        "char": {"beta": 1.0},
        "word": {"beta": 1.0},
        "word_kw": {"beta": 2.0},
    }
    source.reward_functions = variant.reward_functions
    assert OmegaConf.to_container(variant, resolve=False) == OmegaConf.to_container(source, resolve=False)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"sample": {"source": "openml"}}, "finite non-negative number"),
        ({"offset": 2}, "accepts name"),
        ({"range": {"source": "openml"}}, "integer bounds"),
        ({"range": "0,2"}, "integer bounds"),
        ({"range": [0]}, "integer bounds"),
        ({"range": [0, 2, 3]}, "integer bounds"),
        ({"range": [0, 1.5]}, "integer bounds"),
        ({"range": [False, 2]}, "integer bounds"),
        ({"range": [-1, 2]}, "must satisfy"),
        ({"range": [3, 2]}, "must satisfy"),
        ({"range": [0, 6]}, "must satisfy"),
        ({"shuffle_seed": True}, "shuffle_seed must be a non-negative integer"),
        ({"shuffle_seed": -1}, "shuffle_seed must be a non-negative integer"),
        ({"shuffle_seed": 1.5}, "shuffle_seed must be a non-negative integer"),
    ],
)
def test_mix_rejects_non_numeric_budgets_and_invalid_ranges(dataset_class, cached_sources, options, message):
    config = OmegaConf.create({"num_proc": None, "mix": [{"datasource": [{"name": "earning", **options}]}]})
    with pytest.raises(ValueError, match=message):
        dataset_class(cached_sources, None, config)


@pytest.mark.parametrize(
    ("source_range", "expected"),
    [
        ([2, 5], ["first-2", "second-0", "second-1", "first-2"]),
        ([1, 3], ["first-1", "first-2"] * 2),
        ([4, 5], ["second-1"] * 4),
        ([0, 5], ["first-0", "first-1", "first-2", "second-0"]),
        (None, ["first-0", "first-1", "first-2", "second-0"]),
    ],
)
def test_mix_ranges_select_subsets_and_repeat_selected_rows(dataset_class, cached_sources, source_range, expected):
    config = OmegaConf.create(
        {"num_proc": None, "mix": [{"datasource": [{"name": "earning", "sample": 4, "range": source_range}]}]}
    )
    assert list(dataset_class(cached_sources, None, config).ds["id"]) == expected


def test_mix_applies_range_before_source_shuffle(dataset_class, cached_sources):
    config = OmegaConf.create(
        {
            "num_proc": None,
            "mix": [{"datasource": [{"name": "earning", "sample": 4, "range": [1, 4], "shuffle": True}]}],
        }
    )
    source = datasets.concatenate_datasets(
        [datasets.Dataset.load_from_disk(cached_sources[i]["cache_path"]) for i in (0, 2)]
    )
    remaining = source.select(range(1, 4)).shuffle(seed=42)
    expected = list(remaining["id"]) + list(remaining["id"])[:1]
    actual = list(dataset_class(cached_sources, None, config).ds["id"])
    assert actual == expected
    assert not set(actual) & set(source.select([0, 4])["id"])


@pytest.mark.parametrize(("budget", "expected_count"), [({}, 3), ({"epoch": 0.5}, 2), ({"epoch": 2}, 6)])
def test_mix_range_default_and_epoch_budgets_use_subset_rows(dataset_class, cached_sources, budget, expected_count):
    config = OmegaConf.create(
        {"num_proc": None, "mix": [{"datasource": [{"name": "earning", "range": [2, 5], **budget}]}]}
    )
    dataset = dataset_class(cached_sources, None, config)
    assert len(dataset) == expected_count
    assert list(dataset.ds["id"]) == (["first-2", "second-0", "second-1"] * 2)[:expected_count]


@pytest.mark.parametrize("budget", [{}, {"sample": 0}, {"epoch": 1}])
def test_mix_empty_range_skips_empty_selection(dataset_class, cached_sources, budget):
    config = OmegaConf.create(
        {
            "num_proc": None,
            "mix": [{"datasource": [{"name": "earning", "range": [5, 5], **budget}, {"name": "openml"}]}],
        }
    )
    assert list(dataset_class(cached_sources, None, config).ds["id"]) == ["other-0", "other-1"]


def test_mix_empty_range_rejects_positive_budget(dataset_class, cached_sources):
    config = OmegaConf.create(
        {"num_proc": None, "mix": [{"datasource": [{"name": "earning", "range": [5, 5], "sample": 1}]}]}
    )
    with pytest.raises(ValueError, match="leaves no rows for a positive sample budget"):
        dataset_class(cached_sources, None, config)


def test_single_datasource_group_uses_all_weighted_rows(dataset_class, cached_sources):
    dataset = dataset_class(
        [cached_sources[0], cached_sources[2]],
        None,
        OmegaConf.create({"num_proc": None, "data_source": {"earning": {"num_epoch": 2}}}),
    )
    assert list(dataset.ds["id"]) == ["first-0", "first-1", "first-2", "second-0", "second-1"] * 2


def test_single_dataset_uses_concatenation_budgets(dataset_class, cached_sources):
    dataset = dataset_class(
        cached_sources[:1],
        None,
        OmegaConf.create({"num_proc": None, "data_source": {"earning": {"num_epoch": 2}}}),
    )
    assert list(dataset.ds["id"]) == ["first-0", "first-1", "first-2"] * 2


def test_zero_budget_omits_source_and_rejects_empty_mixture(dataset_class, cached_sources):
    config = OmegaConf.create(
        {
            "use_interleave": False,
            "num_proc": None,
            "data_source": {"earning": {"num_epoch": 0}, "openml": {"num_sample": 3}},
        }
    )
    dataset = dataset_class(cached_sources, None, config)
    assert list(dataset.ds["id"]) == ["other-0", "other-1", "other-0"]
    config.data_source.openml.num_sample = 0
    with pytest.raises(ValueError, match="No samples remain"):
        dataset_class(cached_sources, None, config)


@pytest.mark.parametrize("extra_config", [{}, {"data_source": {"earning": {}, "openml": {}}}, {"data_source": None}])
def test_training_interleaving_groups_only_with_source_config(dataset_class, cached_sources, extra_config):
    dataset = dataset_class(cached_sources, None, OmegaConf.create({"num_proc": None, **extra_config}))
    if extra_config.get("data_source") is None:
        assert list(dataset.ds["id"]) == ["first-0", "other-0", "second-0", "first-1", "other-1", "second-1"]
    else:
        assert list(dataset.ds["id"]) == ["first-0", "other-0", "first-1", "other-1"]


@pytest.mark.parametrize("extra_config", [{}, {"data_source": {"earning": {}, "openml": {}}}, {"data_source": None}])
def test_concatenation_groups_only_with_source_config(dataset_class, cached_sources, extra_config):
    dataset = dataset_class(
        cached_sources,
        None,
        OmegaConf.create({"num_proc": None, "use_interleave": False, **extra_config}),
    )
    if extra_config.get("data_source") is None:
        assert list(dataset.ds["id"]) == ["first-0", "first-1", "first-2", "other-0", "other-1", "second-0", "second-1"]
    else:
        assert list(dataset.ds["id"]) == ["first-0", "first-1", "first-2", "second-0", "second-1", "other-0", "other-1"]
    assert Counter(dataset.ds["data_source"]) == {"earning": 5, "openml": 2}


@pytest.mark.parametrize("extra_config", [{}, {"data_source": None}])
@pytest.mark.parametrize("use_interleave", [False, True])
def test_missing_or_null_source_config_skips_grouping(
    dataset_class, dataset_namespace, cached_sources, extra_config, use_interleave, monkeypatch
):
    def unexpected_grouping(*args, **kwargs):
        raise AssertionError("Grouping must not run without a non-null data_source setting")

    monkeypatch.setitem(dataset_namespace, "group_datasets", unexpected_grouping)
    dataset = dataset_class(
        cached_sources,
        None,
        OmegaConf.create({"num_proc": None, "use_interleave": use_interleave, **extra_config}),
    )
    assert len(dataset) == (6 if use_interleave else 7)


@pytest.mark.parametrize("key", ["num_epochs", "num_samples", "num_epoch", "num_sample"])
def test_loader_does_not_precheck_dataset_entries(dataset_class, cached_sources, key):
    cached_sources[0][key] = 1
    dataset = dataset_class(cached_sources[:1], None, OmegaConf.create({"num_proc": None}))
    assert list(dataset.ds["id"]) == ["first-0", "first-1", "first-2"]


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, ["a", "b", "c", "d"]),
        ({"num_epoch": None, "num_sample": None}, ["a", "b", "c", "d"]),
        ({"num_epoch": 0}, []),
        ({"num_epoch": 0.1}, []),
        ({"num_epoch": 0.75}, ["a", "d", "b"]),
        ({"num_epoch": 2.5}, ["a", "b", "c", "d"] * 2 + ["a", "d"]),
        ({"num_sample": 0}, []),
        ({"num_sample": 3}, ["a", "d", "b"]),
        ({"num_sample": 4}, ["a", "b", "c", "d"]),
        ({"num_sample": 5}, ["a", "b", "c", "d", "a"]),
    ],
)
def test_source_budget_is_applied_once_across_datasets(source_sampler, options, expected):
    inputs = [
        datasets.Dataset.from_dict({"id": ["a", "b"], "data_source": ["earning"] * 2}),
        datasets.Dataset.from_dict({"id": ["other"], "data_source": ["unbudgeted"]}),
        datasets.Dataset.from_dict({"id": ["c", "d"], "data_source": ["earning"] * 2}),
    ]
    grouped = source_sampler(inputs, OmegaConf.create({"earning": options, "unbudgeted": {}}))
    assert isinstance(grouped, list)
    assert all(isinstance(ds, datasets.Dataset) for ds in grouped)
    assert [len(ds) for ds in grouped] == ([len(expected), 1] if expected else [1])
    assert [ds[0]["data_source"] for ds in grouped] == (["earning", "unbudgeted"] if expected else ["unbudgeted"])
    result = datasets.concatenate_datasets(grouped)
    assert list(result["id"]) == expected + ["other"]
    assert result.features == inputs[0].features
    assert [len(ds) for ds in inputs] == [2, 1, 2]


@pytest.mark.parametrize("config", [None, []])
def test_source_config_must_be_a_mapping(source_sampler, config):
    ds = datasets.Dataset.from_dict({"id": ["a"], "data_source": ["earning"]})
    with pytest.raises(ValueError, match="must map"):
        source_sampler([ds], config)


@pytest.mark.parametrize("options", [{"num_epoch": -1}, {"num_sample": -1}])
def test_negative_result_size_raises_when_sampling(source_sampler, options):
    ds = datasets.Dataset.from_dict({"id": ["a"], "data_source": ["earning"]})
    with pytest.raises(ValueError, match="negative sampled size"):
        source_sampler([ds], {"earning": options})


@pytest.mark.parametrize("missing_options", [{"num_sample": 100}, {"num_sample": -1}, "unused"])
def test_missing_configured_sources_raise(source_sampler, missing_options):
    ds = datasets.Dataset.from_dict({"id": ["a"], "data_source": ["earning"]})
    with pytest.raises(ValueError, match=r"no non-empty training datasets: \['missing'\]"):
        source_sampler([ds], {"missing": missing_options, "earning": {"num_epoch": 2}})


@pytest.mark.parametrize("config", [{}, {"beta": {}, "alpha": {}}])
def test_unlisted_sources_raise(source_sampler, config):
    inputs = [
        datasets.Dataset.from_dict({"id": [index], "data_source": [source]})
        for index, source in enumerate(["unlisted-z", "alpha", "beta", "unlisted-a", "alpha"])
    ]
    with pytest.raises(ValueError, match="missing from data.data_source.*unlisted-z.*unlisted-a"):
        source_sampler(inputs, config)


def test_empty_source_mapping_rejects_loaded_training_sources(dataset_class, cached_sources):
    with pytest.raises(ValueError, match="missing from data.data_source"):
        dataset_class(cached_sources, None, OmegaConf.create({"num_proc": None, "data_source": {}}))


@pytest.mark.parametrize("configured", [
    {"earning": {}, "typo": {}},
    {"typo": {}, "another": {}},
    {None: {}, 1: {}},
])
def test_source_match_error_reports_both_directions(source_sampler, configured, monkeypatch):
    inputs = [
        datasets.Dataset.from_dict({"id": [index], "data_source": [source]})
        for index, source in enumerate(["earning", "openml"])
    ]

    def unexpected_sampling(*args, **kwargs):
        raise AssertionError("Datasource matching must be checked before sampling")

    monkeypatch.setattr(datasets.Dataset, "shuffle", unexpected_sampling)
    monkeypatch.setattr(random, "sample", unexpected_sampling)
    with pytest.raises(ValueError, match="Datasource names do not match") as exc_info:
        source_sampler(inputs, configured)
    missing = [source for source in configured if source not in {"earning", "openml"}]
    unlisted = [source for source in ["earning", "openml"] if source not in configured]
    assert f"no non-empty training datasets: {missing}" in str(exc_info.value)
    assert f"missing from data.data_source: {unlisted}" in str(exc_info.value)


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("budget", [{}, {"num_epoch": 2}, {"num_sample": 7}])
def test_each_sampled_source_can_be_shuffled(source_sampler, shuffle, budget, monkeypatch):
    source = datasets.Dataset.from_dict({"id": list(range(10)), "data_source": ["earning"] * 10})
    other = datasets.Dataset.from_dict({"id": [10, 11], "data_source": ["other"] * 2})
    calls = []
    original_shuffle = datasets.Dataset.shuffle

    def track_shuffle(ds, *args, **kwargs):
        calls.append((set(ds["data_source"]), args, kwargs))
        return original_shuffle(ds, *args, **kwargs)

    monkeypatch.setattr(datasets.Dataset, "shuffle", track_shuffle)
    expected_rng = random.Random()
    expected_rng.setstate(random.getstate())
    config = OmegaConf.create({"earning": {**budget, "shuffle": shuffle}, "other": {}})
    grouped = source_sampler([source.select(range(5)), other, source.select(range(5, 10))], config)
    expected = source
    if budget.get("num_epoch") == 2:
        expected = datasets.concatenate_datasets([source, source])
    elif budget.get("num_sample") == 7:
        expected = source.select(expected_rng.sample(range(10), 7))
    assert [ds[0]["data_source"] for ds in grouped] == ["earning", "other"]
    assert grouped[0].features == expected.features
    assert Counter(grouped[0]["id"]) == Counter(expected["id"])
    if not shuffle:
        assert grouped[0].to_dict() == expected.to_dict()
    assert calls == ([({"earning"}, (), {})] if shuffle else [])
    assert grouped[1].to_dict() == other.to_dict()


@pytest.mark.parametrize("budget", [{"num_sample": 7}, {"num_epoch": 0.35}, {"num_sample": 47}])
def test_partial_passes_use_global_random_without_replacement(source_sampler, budget):
    source = datasets.Dataset.from_dict({"id": list(range(20)), "data_source": ["earning"] * 20})
    config = {"earning": budget}
    global_state = random.getstate()
    expected_rng = random.Random()
    expected_rng.setstate(global_state)
    first = source_sampler([source], config)[0]
    assert random.getstate() != global_state
    repeated = source_sampler([source], config)[0]
    assert first.to_dict() != repeated.to_dict()
    expected_size = 47 if budget.get("num_sample") == 47 else 7
    assert len(first) == expected_size
    full_passes, remainder = divmod(expected_size, len(source))
    assert list(first["id"])[: full_passes * 20] == list(range(20)) * full_passes
    tail = list(first["id"])[full_passes * 20 :]
    assert len(set(tail)) == remainder
    assert set(tail) <= set(range(20))
    assert tail != list(range(remainder))
    assert tail == expected_rng.sample(range(20), remainder)


def test_shuffle_defaults_to_false(source_sampler, monkeypatch):
    source = datasets.Dataset.from_dict({"id": list(range(10)), "data_source": ["earning"] * 10})

    def unexpected_shuffle(*args, **kwargs):
        raise AssertionError("Dataset.shuffle must not run when shuffle is omitted")

    monkeypatch.setattr(datasets.Dataset, "shuffle", unexpected_shuffle)
    assert list(source_sampler([source], {"earning": {}})[0]["id"]) == list(range(10))
    sampled = source_sampler([source], {"earning": {"num_sample": 4}})[0]
    assert list(sampled["id"]) == random.Random(42).sample(range(10), 4)


def test_loader_applies_per_source_shuffle(dataset_class, cached_sources, monkeypatch):
    calls = []
    original_shuffle = datasets.Dataset.shuffle

    def track_shuffle(ds, *args, **kwargs):
        calls.append((len(ds), args, kwargs))
        return original_shuffle(ds, *args, **kwargs)

    monkeypatch.setattr(datasets.Dataset, "shuffle", track_shuffle)
    config = OmegaConf.create(
        {
            "use_interleave": False,
            "num_proc": None,
            "data_source": {"earning": {"num_epoch": 2, "shuffle": True}, "openml": {}},
        }
    )
    dataset = dataset_class(cached_sources, None, config)
    expected = datasets.concatenate_datasets(
        [datasets.Dataset.load_from_disk(cached_sources[i]["cache_path"]) for i in [0, 2, 0, 2]]
    )
    assert Counter(dataset.ds["id"][:10]) == Counter(expected["id"])
    assert list(dataset.ds["id"])[10:] == ["other-0", "other-1"]
    assert list(dataset.ds["data_source"]) == ["earning"] * 10 + ["openml"] * 2
    assert calls == [(10, (), {})]


def test_empty_datasets_are_skipped_with_warning(source_sampler, caplog):
    ds = datasets.Dataset.from_dict({"id": ["a"], "data_source": ["earning"]})
    result = source_sampler([ds.select([]), ds], {"earning": {"num_epoch": 2}})
    assert len(result) == 1
    assert list(result[0]["id"]) == ["a", "a"]
    assert "Skipping empty training dataset 0" in caplog.text
    with pytest.raises(ValueError, match="no non-empty training datasets"):
        source_sampler([ds.select([])], {"earning": {"num_sample": 2}})
    with pytest.raises(ValueError, match="No samples remain"):
        source_sampler([ds.select([])], {})


@pytest.mark.parametrize("source", [None, "", 1])
def test_source_labels_are_not_prechecked(source_sampler, source):
    ds = datasets.Dataset.from_dict({"id": ["a"], "data_source": [source]})
    grouped = source_sampler([ds], {source: {}})
    assert len(grouped) == 1
    assert grouped[0].to_dict() == ds.to_dict()


@pytest.mark.parametrize("config", [{"default": {}, "other": {}}, {"default": {"num_epoch": 2}, "other": {}}])
def test_absent_source_labels_group_under_default(source_sampler, config):
    inputs = [
        datasets.Dataset.from_dict({"id": ["a"]}),
        datasets.Dataset.from_dict({"id": ["other"], "data_source": ["other"]}),
        datasets.Dataset.from_dict({"id": ["b"], "data_source": ["default"]}),
        datasets.Dataset.from_dict({"id": ["c"]}),
    ]
    grouped = source_sampler(inputs, config)
    assert len(grouped) == 2
    assert list(grouped[0]["id"]) == ["a", "b", "c"] * config["default"].get("num_epoch", 1)
    assert grouped[1].to_dict() == inputs[1].to_dict()
    assert inputs[0].column_names == ["id"]


def test_source_is_inferred_only_from_first_sample(source_sampler):
    ds = datasets.Dataset.from_dict({"id": ["a", "b"], "data_source": ["earning", "not_inspected"]})
    assert list(source_sampler([ds], {"earning": {"num_epoch": 2}})[0]["id"]) == ["a", "b", "a", "b"]


@pytest.mark.parametrize(
    "config_name",
    sorted(
        path.stem
        for path in (Path(__file__).parents[3] / "recipe/phimm/config/v2609_asr").glob("*earning_ml*verb_hint*.yaml")
    ),
)
def test_earnings_recipes_share_source_and_reward_mapping(config_name):
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    with initialize_config_dir(config_dir=str(config_root / "v2609_asr"), version_base=None):
        config = compose(
            config_name=config_name,
            overrides=[f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"],
        )
    sources = _load_flatten_data_confs()(config.data.train_data)
    expected_sources = [
        "earnings_fy27",
        "earnings_fy27",
        "earnings_fy27",
        "openml",
    ]
    if "_earning_ml_ls_" in config_name:
        expected_sources.append("ls_rare_verb")
    assert [source.post_process.add_field.fields.data_source for source in sources] == expected_sources
    assert config.reward_function_by_data_source.earnings_fy27 == "openml"
    assert "earning_fy27_verb" not in config.reward_function_by_data_source
    assert "earnings_tts_clean_verb" not in config.reward_function_by_data_source


@pytest.mark.parametrize(
    "suffix,earnings_samples,openml_samples",
    [("_smp", 10746, 10746), ("_smp2", 21492, 10746), ("_smp3", 102400, 25600)],
)
def test_source_sampling_recipe_shares_earnings_source_and_preserves_base(
    dataset_class, dataset_namespace, monkeypatch, suffix, earnings_samples, openml_samples
):
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    recipe_name = "remax_2609v0_earning_ml_verb_hint_s1k_bs128_n8_r256_g32_flr"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2609_asr"), version_base=None):
        base = compose(config_name=recipe_name, overrides=[searchpath])
        sampled = compose(config_name=f"{recipe_name}{suffix}", overrides=[searchpath])
        overridden = compose(
            config_name=f"{recipe_name}{suffix}",
            overrides=[
                searchpath,
                "data.data_source.earnings_fy27.num_sample=15000",
                "data.data_source.openml.num_sample=12800",
            ],
        )

    assert OmegaConf.to_container(sampled.data.data_source, resolve=True) == {
        "earnings_fy27": {"num_sample": earnings_samples, **({"shuffle": True} if suffix == "_smp3" else {})},
        "openml": {"num_sample": openml_samples, **({"shuffle": True} if suffix == "_smp3" else {})},
    }
    flatten = _load_flatten_data_confs()
    base_sources = [OmegaConf.to_container(conf, resolve=True) for conf in flatten(base.data.train_data)]
    sampled_sources = [OmegaConf.to_container(conf, resolve=True) for conf in flatten(sampled.data.train_data)]
    assert sampled_sources == base_sources
    assert [conf["post_process"]["add_field"]["fields"]["data_source"] for conf in sampled_sources] == [
        "earnings_fy27",
        "earnings_fy27",
        "earnings_fy27",
        "openml",
    ]
    assert OmegaConf.to_container(sampled.reward_function_by_data_source) == {
        "earnings_fy27": "openml",
        "openml": "openml",
    }
    assert sampled.data.use_interleave is False
    assert sampled.data.shuffle is (suffix != "_smp3")
    assert OmegaConf.to_container(sampled.data.val_data, resolve=True) == OmegaConf.to_container(
        base.data.val_data, resolve=True
    )

    def load_source(**conf):
        source = conf["post_process"]["add_field"]["fields"]["data_source"]
        return datasets.Dataset.from_dict({"id": list(range(1000)), "data_source": [source] * 1000})

    monkeypatch.setitem(dataset_namespace, "create_audio_dataset", load_source)
    training = dataset_class(sampled.data.train_data, None, sampled.data)
    assert Counter(training.ds["data_source"]) == {
        "earnings_fy27": earnings_samples,
        "openml": openml_samples,
    }
    assert len(training) == earnings_samples + openml_samples
    assert list(training.ds["data_source"]) == [
        source for source, options in sampled.data.data_source.items() for _ in range(options.num_sample)
    ]
    del sampled.data.data_source
    if suffix == "_smp3":
        sampled.data.shuffle = base.data.shuffle
    assert OmegaConf.to_container(sampled, resolve=False) == OmegaConf.to_container(base, resolve=False)
    assert overridden.data.data_source.earnings_fy27.num_sample == 15000
    assert overridden.data.data_source.openml.num_sample == 12800


def test_earnings_v0_fixed_lr_recipe_uses_verb_hint_data():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    name = "remax_2609v0_earning_verb_hint_s1k_bs128_n8_r256_g32_flr"
    raw = OmegaConf.load(config_root / f"v2609_asr/{name}.yaml")
    assert set(raw.data) == {"train_batch_size"}
    assert "model" not in raw.actor_rollout_ref
    assert set(raw.actor_rollout_ref.actor.optim) == {"warmup_style", "lr_warmup_steps"}
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2609_asr"), version_base=None):
        config = compose(config_name=name, overrides=[searchpath])
        original = compose(
            config_name="remax_2609v0a_earning_s1k_bs128_n4_r256_g32", overrides=[searchpath]
        )
        base = compose(config_name="base", overrides=[searchpath])

        donor = compose(
            config_name="remax_2609v1_earning_ml_verb_hint_s1k_bs128_n8_r256_g32_flr_smp",
            overrides=[searchpath],
        )

    flatten = _load_flatten_data_confs()
    for key, donor_key in (("train_data", "_earning_train"), ("val_data", "_earning_val")):
        assert [
            OmegaConf.to_container(ds, resolve=True) for ds in flatten(config.data[key])
        ] == [
            OmegaConf.to_container(ds, resolve=True) for ds in flatten(donor[donor_key])
        ]
        assert all(ds.add_task_info.task == "lang_asr_verb" for ds in flatten(config.data[key]))
        assert all(ds.add_task_info.lang_hint is True for ds in flatten(config.data[key]))
    sources = [ds.post_process.add_field.fields.data_source for ds in flatten(config.data.train_data)]
    assert sources == ["earnings_fy27", "earnings_fy27", "earnings_fy27"]
    assert [ds.post_process.add_field.fields.data_source for ds in flatten(config.data.val_data)] == [
        "earning_chunk_verb"
    ]
    assert "_ml_train" not in config
    assert "_ml_val" not in config
    assert OmegaConf.to_container(config.reward_function_by_data_source) == {
        "earnings_fy27": "asr_measure",
    }
    assert config.data.get("data_source") is None
    assert config.actor_rollout_ref.model.path == base.actor_rollout_ref.model.path
    assert config.actor_rollout_ref.model.path != original.actor_rollout_ref.model.path
    assert config.actor_rollout_ref.model.lora_rank == 256
    assert config.actor_rollout_ref.rollout.n == 8
    assert config.actor_rollout_ref.actor.optim.lr == 5e-6
    assert config.actor_rollout_ref.actor.optim.warmup_style == "constant"
    assert config.actor_rollout_ref.actor.optim.lr_warmup_steps == 0
    assert config.trainer.nnodes * config.trainer.n_gpus_per_node == 32
    assert config.data.train_batch_size == 128
    assert config.trainer.total_epochs == 30
    assert config.trainer.total_training_steps == 1000

    original.actor_rollout_ref.model.path = base.actor_rollout_ref.model.path
    original.actor_rollout_ref.rollout.n = 8
    original.actor_rollout_ref.actor.optim.warmup_style = "constant"
    original.actor_rollout_ref.actor.optim.lr_warmup_steps = 0
    actual = OmegaConf.to_container(config, resolve=False)
    expected = OmegaConf.to_container(original, resolve=False)
    assert "_earning_train" not in actual
    assert "_earning_val" not in actual
    for key in ("train_data", "val_data"):
        actual["data"][key] = expected["data"][key]
    actual["reward_function_by_data_source"] = expected["reward_function_by_data_source"]
    assert actual == expected


def test_v2609_asr_base_lr_floor_inheritance_and_override():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    name = "remax_2609r2a_ml_verb_hint_s100_bs128_n32_r256_g32_avg4_t2"
    override_name = "remax_2609r2_earning_ml_verb_hint_s948_bs128_n16_r256_g32_ds3_sfl_edit_smts_t15"
    raw = OmegaConf.load(config_root / f"v2609_asr/{name}.yaml")
    assert "actor" not in raw.actor_rollout_ref
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2609_asr"), version_base=None):
        base = compose(config_name="base", overrides=[searchpath])
        inherited = compose(config_name=name, overrides=[searchpath])
        overridden = compose(config_name=override_name, overrides=[searchpath])

    assert base.actor_rollout_ref.actor.optim.min_lr_ratio == 0.01
    assert inherited.actor_rollout_ref.actor.optim == base.actor_rollout_ref.actor.optim
    assert overridden.actor_rollout_ref.actor.optim.min_lr_ratio == 0.2


def test_smp3_without_fixed_lr_inherits_base_optimizer():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    stem = "remax_2609v0_earning_ml_verb_hint_s1k_bs128_n8_r256_g32"
    raw = OmegaConf.load(config_root / f"v2609_asr/{stem}_smp3.yaml")
    assert "actor" not in raw.actor_rollout_ref
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2609_asr"), version_base=None):
        config = compose(config_name=f"{stem}_smp3", overrides=[searchpath])
        fixed = compose(config_name=f"{stem}_flr_smp3", overrides=[searchpath])
        base = compose(config_name="base", overrides=[searchpath])

    assert config.actor_rollout_ref.actor.optim == base.actor_rollout_ref.actor.optim
    assert config.actor_rollout_ref.actor.optim.lr == 5e-6
    assert config.actor_rollout_ref.actor.optim.warmup_style == "cosine"
    assert config.actor_rollout_ref.actor.optim.min_lr_ratio == 0.01
    assert list(config.data.data_source) == ["openml", "earnings_fy27"]
    assert config.data.data_source.openml == {"num_sample": 25600, "shuffle": True}
    assert config.data.data_source.earnings_fy27 == {"num_sample": 102400, "shuffle": True}
    assert config.data.shuffle is False
    fixed.actor_rollout_ref.actor.optim = base.actor_rollout_ref.actor.optim
    assert OmegaConf.to_container(config, resolve=False) == OmegaConf.to_container(fixed, resolve=False)


@pytest.mark.parametrize("name,original_name,steps", [
    (
        "remax_2609v0_ml_hint_s200_bs128_n8_flr",
        "remax_2609v0_ml_hint_s200_bs128_n8_r256_g16",
        200,
    ),
])
def test_fixed_lr_recipes_inherit_base_defaults(name, original_name, steps):
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    raw = OmegaConf.load(config_root / f"v2609_asr/{name}.yaml")
    assert "model" not in raw.actor_rollout_ref
    assert set(raw.actor_rollout_ref.rollout) == {"n"}
    assert set(raw.actor_rollout_ref.actor.optim) == {"warmup_style", "lr_warmup_steps"}
    assert "nnodes" not in raw.trainer
    assert not {"r256", "g16", "gt"} & set(name.split("_"))
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2609_asr"), version_base=None):
        config = compose(config_name=name, overrides=[searchpath])
        original = compose(config_name=original_name, overrides=[searchpath])
        base = compose(config_name="base", overrides=[searchpath])

    assert config.actor_rollout_ref.model.lora_rank == base.actor_rollout_ref.model.lora_rank == 256
    assert config.actor_rollout_ref.model.path == base.actor_rollout_ref.model.path
    assert config.actor_rollout_ref.rollout.gt_rollout is True
    assert config.actor_rollout_ref.rollout.n == 8
    assert config.actor_rollout_ref.actor.optim.lr == base.actor_rollout_ref.actor.optim.lr == 5e-6
    assert config.actor_rollout_ref.actor.optim.warmup_style == "constant"
    assert config.actor_rollout_ref.actor.optim.lr_warmup_steps == 0
    assert config.trainer.nnodes * config.trainer.n_gpus_per_node == 16
    assert config.data.train_batch_size == 128
    assert config.trainer.total_epochs == 30
    assert config.trainer.total_training_steps == steps

    original.actor_rollout_ref.model.lora_rank = 256
    original.actor_rollout_ref.model.path = original.actor_rollout_ref.model.path.rstrip("/")
    original.actor_rollout_ref.rollout.n = 8
    original.actor_rollout_ref.actor.optim.warmup_style = "constant"
    original.actor_rollout_ref.actor.optim.lr_warmup_steps = 0
    assert OmegaConf.to_container(config, resolve=False) == OmegaConf.to_container(original, resolve=False)


def test_v1_source_sampling_recipe_changes_only_model():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    recipe_suffix = "earning_ml_verb_hint_s1k_bs128_n8_r256_g32"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2609_asr"), version_base=None):
        v0 = compose(config_name=f"remax_2609v0_{recipe_suffix}_flr_smp", overrides=[searchpath])
        v1 = compose(config_name=f"remax_2609v1_{recipe_suffix}_flr_smp", overrides=[searchpath])
        existing_v1 = compose(config_name=f"remax_2609v1_{recipe_suffix}_nshf_flr", overrides=[searchpath])

    assert v1.actor_rollout_ref.model.path == existing_v1.actor_rollout_ref.model.path
    assert v1.actor_rollout_ref.model.path != v0.actor_rollout_ref.model.path
    v1.actor_rollout_ref.model.path = v0.actor_rollout_ref.model.path
    assert OmegaConf.to_container(v1, resolve=False) == OmegaConf.to_container(v0, resolve=False)


def test_ls_rare_sampling_recipe_adds_verbatim_hint_data(dataset_class, dataset_namespace, monkeypatch):
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    recipe_name = "remax_2609v0_earning_ml_verb_hint_s1k_bs128_n8_r256_g32_flr_smp"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2609_asr"), version_base=None):
        base = compose(config_name=recipe_name, overrides=[searchpath])
        config = compose(config_name=recipe_name.replace("_ml_", "_ml_ls_"), overrides=[searchpath])

    flatten = _load_flatten_data_confs()
    train = flatten(config.data.train_data)
    val = flatten(config.data.val_data)
    base_train = flatten(base.data.train_data)
    base_val = flatten(base.data.val_data)
    assert train[:-1] == base_train
    assert val[:-2] == base_val
    assert [item.post_process.add_field.fields.data_source for item in val[-2:]] == ["ls_clean", "ls_other"]
    original_train = OmegaConf.load(config_root / "data/train_data/ls_rare_nonempty_verb.yaml")
    original_val = OmegaConf.load(config_root / "data/val_data/ls_kw_verb.yaml")
    for added, original in zip([train[-1], *val[-2:]], [original_train, *original_val], strict=True):
        assert added.add_task_info == {
            "task": "lang_asr_verb", "language": "English", "prefix_prob": 0.0, "lang_hint": True,
        }
        expected = OmegaConf.to_container(original, resolve=True)
        expected["add_task_info"]["lang_hint"] = True
        assert OmegaConf.to_container(added, resolve=True) == expected

    assert config.reward_function_by_data_source.ls_rare_verb == "openml"
    assert config.reward_functions.openml.reward_kwargs.measures.word_kw.beta == 1.0
    assert config.reward_functions.openml.reward_kwargs.version == 2609
    for source in ("ls_clean", "ls_other"):
        assert base.val_reward.reward_function_by_data_source[source] == "openasr_en"
        assert config.val_reward.reward_function_by_data_source[source] == "openasr_en"
    assert config.val_reward.reward_functions.openasr_en.reward_kwargs.version == 2609
    assert config.data.data_source.ls_rare_verb == {"num_sample": 10746}

    def load_source(**conf):
        source = conf["post_process"]["add_field"]["fields"]["data_source"]
        size = 256213 if source == "ls_rare_verb" else 3
        return datasets.Dataset.from_dict({"id": list(range(size)), "data_source": [source] * size})

    monkeypatch.setitem(dataset_namespace, "create_audio_dataset", load_source)
    training = dataset_class(config.data.train_data, None, config.data)
    assert Counter(training.ds["data_source"]) == {"earnings_fy27": 10746, "openml": 10746, "ls_rare_verb": 10746}
    validation = dataset_class(config.data.val_data, None, config.data, is_train=False)
    assert Counter(validation.ds["data_source"]) == {
        item.post_process.add_field.fields.data_source: 3 for item in val
    }

    config.data.train_data = base.data.train_data
    config.data.val_data = base.data.val_data
    remaining = OmegaConf.to_container(config, resolve=False)
    del remaining["_ls_train"]
    del remaining["_ls_val"]
    del remaining["reward_functions"]["openml"]["reward_kwargs"]["measures"]["word_kw"]
    del remaining["reward_function_by_data_source"]["ls_rare_verb"]
    del remaining["data"]["data_source"]["ls_rare_verb"]
    assert remaining == OmegaConf.to_container(base, resolve=False)


def test_load_audio_with_retries_skips_unreadable_training_sample():
    load_audio_with_retries = _load_audio_retry_helper()
    rows = [{"audio_path": "bad.wav"}, {"audio_path": "good.wav"}]

    def load_audio(row, max_dur):
        if row["audio_path"] == "bad.wav":
            raise OSError("unreadable")
        return max_dur, row["audio_path"]

    row, audio = load_audio_with_retries(rows, 0, 40, 1, load_audio, (OSError,))

    assert row == rows[1]
    assert audio == (40, "good.wav")


def test_load_audio_with_retries_spreads_fallbacks_across_dataset():
    load_audio_with_retries = _load_audio_retry_helper()
    rows = [{"audio_path": f"{index}.wav"} for index in range(20)]
    attempted = []

    def load_audio(row, max_dur):
        attempted.append(row["audio_path"])
        if row["audio_path"] == "0.wav":
            raise OSError("unreadable")
        return row["audio_path"]

    row, audio = load_audio_with_retries(rows, 0, 40, 1, load_audio, (OSError,))

    assert attempted == ["0.wav", "10.wav"]
    assert row == rows[10]
    assert audio == "10.wav"


def test_load_audio_with_retries_avoids_stringifying_broken_audio_error():
    load_audio_with_retries = _load_audio_retry_helper()

    class BrokenAudioError(Exception):
        def __str__(self):
            raise TypeError("broken exception formatting")

    def load_audio(row, max_dur):
        raise BrokenAudioError()

    try:
        load_audio_with_retries(
            [{"audio_path": "bad.wav"}],
            0,
            40,
            0,
            load_audio,
            (BrokenAudioError,),
        )
    except RuntimeError as exc:
        assert str(exc) == "Unable to load audio after 1 attempt(s): ['bad.wav']"
    else:
        raise AssertionError("expected contextual RuntimeError")


def test_training_audio_retries_remote_cache_failures():
    module_path = Path(__file__).parents[3] / "recipe/phimm/data/rl_dataset.py"
    module = ast.parse(module_path.read_text())
    getitem = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "__getitem__"
    )
    retry_call = next(
        node
        for node in ast.walk(getitem)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_load_audio_with_retries"
    )

    recoverable_errors = retry_call.args[5]
    assert isinstance(recoverable_errors, ast.Tuple)
    assert "RemoteAudioCacheError" in {
        element.id for element in recoverable_errors.elts if isinstance(element, ast.Name)
    }


def test_name_en13m_training_tolerates_consecutive_missing_audio():
    config_path = (
        Path(__file__).parents[3]
        / "recipe/phimm/config/v2609_entity/remax_2609r2_name_en13m_s1k_bs128_n4_r256_g32.yaml"
    )
    config = OmegaConf.load(config_path)

    assert config.data.max_audio_load_retries == 64
    assert config.actor_rollout_ref.model.path == "/root/models/qwen35_audio_2609r2"

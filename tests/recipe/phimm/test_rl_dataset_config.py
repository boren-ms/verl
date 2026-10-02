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
from omegaconf import OmegaConf


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
    for name in ("base", "eval_base"):
        assert OmegaConf.load(root / f"v2609_asr/{name}.yaml").data.val_batch_size == 256


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
    return dataset_namespace["group_weighted_datasource"]


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
        ([0, 1, 2], ["earning", "openml"]),
        ([1, 0, 2], ["openml", "earning"]),
    ],
)
def test_datasources_stay_serial_in_first_appearance_order(dataset_class, cached_sources, dataset_order, source_order):
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
def test_interleaving_uses_weighted_datasource_groups(dataset_class, cached_sources, interleave_options):
    config = OmegaConf.create(
        {
            "use_interleave": True,
            "num_proc": None,
            "interleave_ds": interleave_options,
            "data_source": {
                "earning": {"num_epoch": 2},
                "openml": {"num_sample": 3},
            },
        }
    )
    dataset = dataset_class(cached_sources, None, config)
    earning = datasets.concatenate_datasets(
        [datasets.Dataset.load_from_disk(cached_sources[i]["cache_path"]) for i in [0, 2, 0, 2]]
    )
    openml = datasets.Dataset.load_from_disk(cached_sources[1]["cache_path"]).select([0, 1, 0])
    expected = datasets.interleave_datasets(
        [earning, openml],
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


@pytest.mark.parametrize("extra_config", [{}, {"data_source": {}}, {"data_source": None}])
def test_training_interleaving_groups_only_with_source_config(dataset_class, cached_sources, extra_config):
    dataset = dataset_class(cached_sources, None, OmegaConf.create({"num_proc": None, **extra_config}))
    if extra_config.get("data_source") is None:
        assert list(dataset.ds["id"]) == ["first-0", "other-0", "second-0", "first-1", "other-1", "second-1"]
    else:
        assert list(dataset.ds["id"]) == ["first-0", "other-0", "first-1", "other-1"]


@pytest.mark.parametrize("extra_config", [{}, {"data_source": {}}, {"data_source": None}])
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

    monkeypatch.setitem(dataset_namespace, "group_weighted_datasource", unexpected_grouping)
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
    grouped = source_sampler(inputs, OmegaConf.create({"earning": options}))
    assert isinstance(grouped, list)
    assert all(isinstance(ds, datasets.Dataset) for ds in grouped)
    assert [len(ds) for ds in grouped] == ([len(expected), 1] if expected else [1])
    assert [ds[0]["data_source"] for ds in grouped] == (["earning", "unbudgeted"] if expected else ["unbudgeted"])
    result = datasets.concatenate_datasets(grouped)
    assert list(result["id"]) == expected + ["other"]
    assert result.features == inputs[0].features
    assert [len(ds) for ds in inputs] == [2, 1, 2]


def test_source_config_must_be_a_mapping(source_sampler):
    ds = datasets.Dataset.from_dict({"id": ["a"], "data_source": ["earning"]})
    with pytest.raises(ValueError, match="must map"):
        source_sampler([ds], OmegaConf.create([]))


@pytest.mark.parametrize("options", [{"num_epoch": -1}, {"num_sample": -1}])
def test_negative_result_size_raises_when_sampling(source_sampler, options):
    ds = datasets.Dataset.from_dict({"id": ["a"], "data_source": ["earning"]})
    with pytest.raises(ValueError, match="negative sampled size"):
        source_sampler([ds], {"earning": options})


@pytest.mark.parametrize("missing_options", [{"num_sample": 100}, {"num_sample": -1}, "unused"])
def test_missing_configured_sources_only_warn(source_sampler, caplog, missing_options):
    ds = datasets.Dataset.from_dict({"id": ["a"], "data_source": ["earning"]})
    result = source_sampler([ds], {"missing": missing_options, "earning": {"num_epoch": 2}})
    assert len(result) == 1
    assert list(result[0]["id"]) == ["a", "a"]
    assert "no non-empty training datasets: ['missing']" in caplog.text


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
    config = OmegaConf.create({"earning": {**budget, "shuffle": shuffle}})
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
    assert list(source_sampler([source])[0]["id"]) == list(range(10))
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
            "data_source": {"earning": {"num_epoch": 2, "shuffle": True}},
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
    with pytest.raises(ValueError, match="No samples remain"):
        source_sampler([ds.select([])], {"earning": {"num_sample": 2}})


@pytest.mark.parametrize("source", [None, "", 1])
def test_source_labels_are_not_prechecked(source_sampler, source):
    ds = datasets.Dataset.from_dict({"id": ["a"], "data_source": [source]})
    grouped = source_sampler([ds])
    assert len(grouped) == 1
    assert grouped[0].to_dict() == ds.to_dict()


@pytest.mark.parametrize("config", [None, {"default": {"num_epoch": 2}}])
def test_absent_source_labels_group_under_default(source_sampler, config):
    inputs = [
        datasets.Dataset.from_dict({"id": ["a"]}),
        datasets.Dataset.from_dict({"id": ["other"], "data_source": ["other"]}),
        datasets.Dataset.from_dict({"id": ["b"], "data_source": ["default"]}),
        datasets.Dataset.from_dict({"id": ["c"]}),
    ]
    grouped = source_sampler(inputs, config)
    assert len(grouped) == 2
    assert list(grouped[0]["id"]) == ["a", "b", "c"] * (2 if config else 1)
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


@pytest.mark.parametrize("suffix,earnings_samples", [("_smp", 10746), ("_smp2", 21492)])
def test_source_sampling_recipe_shares_earnings_source_and_preserves_base(
    dataset_class, dataset_namespace, monkeypatch, suffix, earnings_samples
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
        "earnings_fy27": {"num_sample": earnings_samples},
        "openml": {"num_sample": 10746},
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
    assert sampled.data.shuffle is True
    assert OmegaConf.to_container(sampled.data.val_data, resolve=True) == OmegaConf.to_container(
        base.data.val_data, resolve=True
    )

    def load_source(**conf):
        source = conf["post_process"]["add_field"]["fields"]["data_source"]
        return datasets.Dataset.from_dict({"id": [0, 1], "data_source": [source] * 2})

    monkeypatch.setitem(dataset_namespace, "create_audio_dataset", load_source)
    training = dataset_class(sampled.data.train_data, None, sampled.data)
    assert Counter(training.ds["data_source"]) == {
        "earnings_fy27": earnings_samples,
        "openml": 10746,
    }
    assert len(training) == earnings_samples + 10746
    assert list(training.ds["data_source"]) == ["earnings_fy27"] * earnings_samples + ["openml"] * 10746
    del sampled.data.data_source
    assert OmegaConf.to_container(sampled, resolve=False) == OmegaConf.to_container(base, resolve=False)
    assert overridden.data.data_source.earnings_fy27.num_sample == 15000
    assert overridden.data.data_source.openml.num_sample == 12800


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
    assert config.reward_functions.openml.reward_kwargs.measures.keyword.beta == 1.0
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
    del remaining["reward_functions"]["openml"]["reward_kwargs"]["measures"]["keyword"]
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

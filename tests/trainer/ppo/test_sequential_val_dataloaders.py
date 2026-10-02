import ast
import sys
from collections.abc import Mapping
from itertools import chain
from pathlib import Path
from types import ModuleType

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader


def _load_trainer_helpers():
    """Exercise loader code without importing the GPU/model training stack."""
    path = Path(__file__).parents[3] / "verl/trainer/ppo/ray_trainer.py"
    tree = ast.parse(path.read_text())
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_validation_dataset_batches", "_extend_validation_reward_extra_infos", "get_collate_fn"}
    ]
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    trainer.body = [
        node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_create_dataloader"
    ]
    module = ModuleType("validation_loader_helpers")
    module.__dict__.update(
        OmegaConf=OmegaConf, open_dict=open_dict, Mapping=Mapping, StatefulDataLoader=StatefulDataLoader,
    )
    body = ast.parse("from __future__ import annotations").body + functions + [trainer]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


ray_trainer = _load_trainer_helpers()


@pytest.fixture
def main_ppo(monkeypatch):
    module = ModuleType("verl.trainer.main_ppo")
    module.create_rl_dataset = None
    module.create_rl_sampler = None
    monkeypatch.setitem(sys.modules, "verl.trainer.main_ppo", module)
    return module


class _NamedDataset(Dataset):
    def __init__(self, name, size):
        self.name = name
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.name, index


class _RecordingDataLoader:
    def __init__(self, dataset, batch_size, persistent_workers, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.persistent_workers = persistent_workers

    def __len__(self):
        return 1

    def __iter__(self):
        yield self.dataset.name


def test_validation_configs_create_ordered_full_dataset_loaders(monkeypatch, main_ppo):
    config = OmegaConf.create(
        {
            "data": {
                "val_data": [
                    {"name": "mixlang", "size": 2},
                    {"name": "earnings", "size": 3},
                ],
                "dataloader_num_workers": 1,
                "persistent_workers": True,
                "train_batch_size": 1,
                "val_batch_size": -1,
            },
            "trainer": {"total_epochs": 1, "total_training_steps": None},
        }
    )
    dataset_calls = []

    def create_dataset(data, data_config, tokenizer, processor, is_train):
        dataset_calls.append(data)
        item = data[0]
        return _NamedDataset(item.name, item.size)

    monkeypatch.setattr(main_ppo, "create_rl_dataset", create_dataset)
    monkeypatch.setattr(ray_trainer, "StatefulDataLoader", _RecordingDataLoader)

    trainer = ray_trainer.RayPPOTrainer.__new__(ray_trainer.RayPPOTrainer)
    trainer.config = config
    trainer.tokenizer = None
    trainer.processor = None
    trainer._create_dataloader(
        train_dataset=_NamedDataset("train", 1),
        val_dataset=None,
        collate_fn=lambda batch: batch,
        train_sampler=object(),
    )

    assert [len(data) for data in dataset_calls] == [1, 1]
    assert [loader.batch_size for loader in trainer.val_dataloaders] == [2, 3]
    assert [loader.persistent_workers for loader in trainer.val_dataloaders] == [False, False]
    assert list(chain.from_iterable(trainer.val_dataloaders)) == ["mixlang", "earnings"]


@pytest.mark.parametrize("as_omegaconf", [False, True])
def test_validation_batch_specs_flatten_groups_and_preserve_configs(as_omegaconf):
    data = [
        [{"name": "short"}, [{"name": "long", "val_batch_size": -1}]],
        {"name": "small", "val_batch_size": 2},
        {"name": "inherited", "val_batch_size": None},
        "validation.parquet",
    ]
    if as_omegaconf:
        data = OmegaConf.create(data)
    specs = list(ray_trainer._validation_dataset_batches(data, 256))
    assert [size for _, size in specs] == [256, -1, 2, 256, 256]
    assert [item["name"] for item, _ in specs[:-1]] == ["short", "long", "small", "inherited"]
    assert specs[-1][0] == "validation.parquet"
    assert all("val_batch_size" not in item for item, _ in specs[:-1])
    assert data[0][1][0]["val_batch_size"] == -1


def test_validation_batch_specs_resolve_parent_interpolations():
    config = OmegaConf.create({
        "batch": 7, "path": "test.jsonl",
        "val": [{"jsonl_paths": "${path}", "val_batch_size": "${batch}"}],
    })
    [(item, size)] = ray_trainer._validation_dataset_batches(config.val, 256)
    assert item.jsonl_paths == "test.jsonl"
    assert size == 7
    assert OmegaConf.to_container(config, resolve=False)["val"][0]["val_batch_size"] == "${batch}"


def test_positive_override_takes_precedence_over_global_full_batch():
    [(item, size)] = ray_trainer._validation_dataset_batches({"name": "short", "val_batch_size": 64}, -1)
    assert item == {"name": "short"}
    assert size == 64


@pytest.mark.parametrize("recipe", [
    "remax_2609v0_earning_ml_ls_verb_hint_s1k_bs128_n8_r256_g32_flr_smp",
    "eval_2609_openall_mix",
    "eval_2609_openml_verb",
])
def test_2609_composed_recipes_select_full_batches_only_for_parent_audio(recipe):
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    with initialize_config_dir(config_dir=str(config_root / "v2609_asr"), version_base=None):
        config = compose(
            config_name=recipe,
            overrides=[f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"],
        )
    assert config.data.val_batch_size == 256
    specs = list(ray_trainer._validation_dataset_batches(config.data.val_data, config.data.val_batch_size))
    assert len(specs) > 1
    for item, batch_size in specs:
        extra_keys = item.get("post_process", {}).get("verl_format", {}).get("extra_keys", [])
        assert batch_size == (-1 if "parent_audio_path" in extra_keys else 256)
        assert "val_batch_size" not in item


@pytest.mark.parametrize("value", [0, -2, True, False, 2.5, "64"])
def test_validation_batch_specs_reject_invalid_overrides(value):
    with pytest.raises(ValueError, match="Dataset val_batch_size"):
        list(ray_trainer._validation_dataset_batches({"val_batch_size": value}, 256))


def _build_loaders(main_ppo, val_data=None, val_files=None, val_dataset=None, default=3):
    calls = []

    def create_dataset(data, data_config, tokenizer, processor, is_train):
        calls.append(data)
        item = data[0]
        if isinstance(item, str):
            return _NamedDataset(item, 5)
        assert "val_batch_size" not in item
        return _NamedDataset(item["name"], item["size"])

    main_ppo.create_rl_dataset = create_dataset
    config = OmegaConf.create({
        "data": {
            "val_data": val_data, "val_files": val_files, "val_batch_size": default,
            "dataloader_num_workers": 0, "persistent_workers": True,
            "train_batch_size": 2, "gen_batch_size": 2,
        },
        "trainer": {"total_epochs": 1, "total_training_steps": None},
    })
    trainer = ray_trainer.RayPPOTrainer.__new__(ray_trainer.RayPPOTrainer)
    trainer.config, trainer.tokenizer, trainer.processor = config, None, None
    trainer._create_dataloader(
        train_dataset=_NamedDataset("train", 4), val_dataset=val_dataset,
        collate_fn=lambda batch: batch, train_sampler=range(4),
    )
    return trainer, calls


def test_validation_loaders_apply_nested_overrides_without_dropping_rows(main_ppo):
    data = [
        [{"name": "short", "size": 8}, {"name": "long", "size": 7, "val_batch_size": -1}],
        [{"name": "small", "size": 5, "val_batch_size": 2}],
    ]
    trainer, calls = _build_loaders(main_ppo, val_data=data)
    assert len(calls) == 3
    assert [loader.batch_size for loader in trainer.val_dataloaders] == [3, 7, 2]
    batches = list(chain.from_iterable(trainer.val_dataloaders))
    assert [len(batch) for batch in batches] == [3, 3, 2, 7, 2, 2, 1]
    assert list(chain.from_iterable(batches)) == [
        (name, index) for name, size in [("short", 8), ("long", 7), ("small", 5)] for index in range(size)
    ]
    assert trainer.train_dataloader.batch_size == 2
    assert trainer.config.data.val_data[0][1].val_batch_size == -1


@pytest.mark.parametrize("default,expected", [(None, 2), (0, 2), (-1, 5), (3, 3)])
def test_supplied_validation_dataset_preserves_global_batch_behavior(main_ppo, default, expected):
    trainer, calls = _build_loaders(main_ppo, val_dataset=_NamedDataset("provided", 5), default=default)
    assert not calls
    assert trainer.val_dataloaders[0].batch_size == expected
    assert sum(len(batch) for batch in trainer.val_dataloaders[0]) == 5


@pytest.mark.parametrize("paths", ["first.parquet", ["first.parquet", "second.parquet"]])
def test_validation_file_paths_remain_supported(main_ppo, paths):
    trainer, calls = _build_loaders(main_ppo, val_files=paths)
    expected_paths = [paths] if isinstance(paths, str) else paths
    assert [call[0] for call in calls] == expected_paths
    assert [loader.batch_size for loader in trainer.val_dataloaders] == [3] * len(expected_paths)


def test_validation_reward_extra_infos_align_heterogeneous_batches():
    accumulated = {"reward": [0.9, 0.8]}

    ray_trainer._extend_validation_reward_extra_infos(
        accumulated,
        {"dter": [0.1, 0.2]},
        previous_sample_count=2,
        batch_size=2,
    )
    accumulated["reward"].extend([0.7, 0.6])
    ray_trainer._extend_validation_reward_extra_infos(
        accumulated,
        {"wer": [0.3, 0.4]},
        previous_sample_count=4,
        batch_size=2,
    )

    assert accumulated["reward"] == [0.9, 0.8, 0.7, 0.6]
    assert accumulated["dter"] == [None, None, 0.1, 0.2, None, None]
    assert accumulated["wer"] == [None, None, None, None, 0.3, 0.4]


def test_validation_reward_extra_infos_reject_misaligned_batch():
    with pytest.raises(ValueError, match="batch_size"):
        ray_trainer._extend_validation_reward_extra_infos(
            {},
            {"dter": [0.1]},
            previous_sample_count=0,
            batch_size=2,
        )
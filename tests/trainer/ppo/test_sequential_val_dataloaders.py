from itertools import chain

import pytest
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from verl.trainer.ppo import ray_trainer


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


def test_validation_configs_create_ordered_full_dataset_loaders(monkeypatch):
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

    monkeypatch.setattr("verl.trainer.main_ppo.create_rl_dataset", create_dataset)
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
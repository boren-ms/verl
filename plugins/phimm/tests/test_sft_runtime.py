"""Exercise native trainer integration in the installed Qwen training runtime."""

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from recipe.phimm.sft.trainer_ray import ASRSFTTrainerRay


def test_ray_sft_retains_partial_validation_and_base3a_order():
    trainer = ASRSFTTrainerRay.__new__(ASRSFTTrainerRay)
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 32, "pad_mode": "no_padding", "num_workers": 0, "shuffle": False},
            "trainer": {
                "nnodes": 2,
                "n_gpus_per_node": 8,
                "total_training_steps": None,
                "total_epochs": 2,
                "save_freq": 10,
                "test_freq": 20,
            },
        }
    )
    trainer.optimizer_config = SimpleNamespace()
    sample = {"input_ids": torch.tensor([1, 2, 3]), "loss_mask": torch.tensor([0, 1, 1])}
    trainer.train_dataset = [sample] * 64
    trainer.val_dataset = [sample] * 19
    trainer._build_dataloader()
    assert trainer.total_training_steps == 4
    assert list(trainer.train_sampler) == list(range(64))
    batches = list(trainer.val_dataloader)
    assert len(batches) == 1
    assert batches[0]["input_ids"].shape[0] == 32
    assert batches[0]["loss_mask"].values().sum() == 64


def test_sft_caches_model_and_tokenizer_paths_before_worker_loading(monkeypatch):
    from recipe.phimm.sft.main_ray import cache_model_paths

    calls = []

    def copy(path):
        calls.append(path)
        return "/node-cache/" + path.rsplit("/", 1)[-1]

    monkeypatch.setattr("verl.utils.fs.copy_to_local", copy)
    result = cache_model_paths(["az://account/model", None, "az://account/tokenizer", "az://account/model"])
    assert calls == ["az://account/model", "az://account/tokenizer"]
    assert result == {"az://account/model": "/node-cache/model", "az://account/tokenizer": "/node-cache/tokenizer"}

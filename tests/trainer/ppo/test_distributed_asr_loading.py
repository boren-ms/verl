"""Exercise the actual trainer's opt-in path without loading GPU model code."""

from types import SimpleNamespace

from omegaconf import OmegaConf
from torch.utils.data import SequentialSampler

from .test_sequential_val_dataloaders import main_ppo as main_ppo, ray_trainer
from recipe.phimm.data.training_loader import DistributedTrainingLoader
from scripts.validate_asr_generation_loading import AudioValidationDataset, identity_collate


def test_trainer_uses_fresh_batch_size_and_resolves_only_actor_rollout_nodes(monkeypatch, main_ppo):
    from recipe.phimm.data import generation_loader

    ds = AudioValidationDataset(17)
    ds.is_training = True
    ds.max_audio_load_retries = 2
    config = OmegaConf.create(
        {
            "data": {
                "train_batch_size": 8,
                "gen_batch_size": 8,
                "val_batch_size": 3,
                "dataloader_num_workers": 0,
                "distributed_training_loading": {
                    "enabled": True,
                    "workers_per_node": 2,
                    "prefetch_batches": 3,
                    "timeout_s": 60,
                },
            },
            "algorithm": {
                "adv_estimator": "remax",
                "bad_case_replay": {"enable": True, "replay_size": 4, "buffer_size": 100},
            },
            "trainer": {"total_epochs": 2, "total_training_steps": None},
        }
    )
    trainer = ray_trainer.RayPPOTrainer.__new__(ray_trainer.RayPPOTrainer)
    trainer.config = config
    trainer.tokenizer = None
    trainer.processor = None
    calls = []
    pool = object()
    trainer.resource_pool_manager = SimpleNamespace(get_resource_pool=lambda role: calls.append(role) or pool)
    monkeypatch.setattr(ray_trainer, "AdvantageEstimator", SimpleNamespace(REMAX="remax"), raising=False)
    monkeypatch.setattr(ray_trainer, "Role", SimpleNamespace(ActorRollout="actor_rollout"), raising=False)
    monkeypatch.setattr(
        generation_loader, "rollout_node_ids", lambda value: ["node1", "node2"] if value is pool else []
    )
    trainer._create_dataloader(ds, ds, identity_collate, SequentialSampler(ds))
    loader = trainer.train_dataloader
    assert isinstance(loader, DistributedTrainingLoader)
    assert calls == []  # GPU placement does not exist when dataloaders are built.
    assert loader.node_ids() == ["node1", "node2"]
    assert calls == ["actor_rollout"]
    assert loader.batch_size == trainer.bad_case_replay.fresh_batch_size == 4
    assert loader.workers_per_node == 2 and loader.prefetch_batches == 3
    assert trainer.total_training_steps == len(loader) * 2 == 8
    assert trainer.val_dataloaders[0].batch_size == 3
    loader.close()

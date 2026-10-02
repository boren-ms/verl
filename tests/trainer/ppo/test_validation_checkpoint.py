import ast
import json
import uuid
from collections import defaultdict
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

import blobfile as bf
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.metric_utils import process_validation_metrics, update_var2metric2val
from verl.trainer.ppo.validation_checkpoint import ValidationCheckpoint


class Samples(Dataset):
    def __init__(self, source):
        self.source = source
        self.reads = []

    def __len__(self):
        return 3

    def __getitem__(self, index):
        self.reads.append(index)
        return {
            "input_ids": torch.tensor([index + 1, 2]),
            "attention_mask": torch.ones(2, dtype=torch.long),
            "position_ids": torch.arange(2),
            "reward_model": {"ground_truth": "reference"},
            "data_source": self.source,
            "extra_info": {"id": f"{self.source}-{index}"},
        }


def collate(rows):
    return {
        key: torch.stack([row[key] for row in rows]) if isinstance(rows[0][key], torch.Tensor)
        else np.array([row[key] for row in rows], dtype=object)
        for key in rows[0]
    }


class Tokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __len__(self):
        return 10

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids.tolist()))


@pytest.fixture
def trainer(tmp_path):
    path = Path(__file__).parents[3] / "verl/trainer/ppo/ray_trainer.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {
        "_validate", "_get_gen_batch", "_dump_generations",
    }]
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "_extend_validation_reward_extra_infos"]
    namespace = dict(globals())
    namespace.update(
        uuid=uuid, defaultdict=defaultdict, chain=chain, bf=bf,
        pad_dataproto_to_divisor=pad_dataproto_to_divisor, unpad_dataproto=unpad_dataproto,
        process_validation_metrics=process_validation_metrics, update_var2metric2val=update_var2metric2val,
    )
    exec(compile(ast.Module(body=functions + [cls], type_ignores=[]), str(path), "exec"), namespace)
    instance = namespace["RayPPOTrainer"]()
    instance.config = OmegaConf.create({
        "trainer": {"val_only": True, "validation_resume": True, "validation_data_dir": str(tmp_path)},
        "data": {"validation_shuffle": False},
        "actor_rollout_ref": {"model": {"path": "model"}, "rollout": {"val_kwargs": {"n": 2, "do_sample": False}}},
        "reward_model": {"enable": False},
    })
    instance.global_steps = 0
    instance.tokenizer = Tokenizer()
    instance.dump_fn = None
    instance.async_rollout_mode = False
    instance._maybe_log_val_generations = lambda **kwargs: None
    instance.val_dataloaders = [
        DataLoader(Samples(source), batch_size=2, collate_fn=collate) for source in ["first", "second"]
    ]
    instance.calls = 0
    instance.fail_at = None

    def generate(batch):
        instance.calls += 1
        if instance.calls == instance.fail_at:
            raise RuntimeError("interrupted")
        return DataProto.from_dict(tensors={"responses": torch.ones((len(batch), 2), dtype=torch.long)})

    instance.actor_rollout_wg = SimpleNamespace(world_size=4, generate_sequences=generate)

    def reward(batch, **kwargs):
        key = "wer" if batch.non_tensor_batch["data_source"][0] == "first" else "dter"
        return {
            "reward_tensor": torch.ones((len(batch), 2)),
            "reward_extra_info": {key: [0.25] * len(batch), "n_err": [1] * len(batch), "n_ref": [4] * len(batch)},
        }

    instance.val_reward_fn = reward
    return instance


def test_dataset_resume_preserves_metrics_outputs_and_skips_completed_audio(trainer, tmp_path):
    trainer.fail_at = 4  # Dataset one complete; dataset two interrupted in its second batch.
    with pytest.raises(RuntimeError, match="interrupted"):
        trainer._validate()
    checkpoint = json.loads((tmp_path / "_resume/0.json").read_text())
    assert checkpoint["completed_datasets"] == 1
    assert len(checkpoint["state"]["scores"]) == 6  # n=2 includes the short final batch.
    for loader in trainer.val_dataloaders:
        loader.dataset.reads.clear()
    trainer.calls, trainer.fail_at = 0, None
    resumed_metrics = trainer._validate()
    assert trainer.calls == 2
    assert trainer.val_dataloaders[0].dataset.reads == []
    assert trainer.val_dataloaders[1].dataset.reads == [0, 1, 2]
    resumed_outputs = [(tmp_path / source / "0.jsonl").read_text() for source in ["first", "second"]]
    assert all(len(output.splitlines()) == 6 for output in resumed_outputs)
    trainer.calls = 0
    assert trainer._validate() == resumed_metrics
    assert trainer.calls == 0
    trainer.config.trainer.validation_resume = False
    assert trainer._validate() == resumed_metrics
    assert trainer.calls == 4
    assert not (tmp_path / "_resume/0.json").exists()
    assert resumed_outputs == [(tmp_path / source / "0.jsonl").read_text() for source in ["first", "second"]]


@pytest.mark.parametrize("setting,value", [
    ("actor_rollout_ref.model.path", "different"),
    ("actor_rollout_ref.rollout.val_kwargs.n", 1),
    ("data.validation_shuffle", True),
])
def test_incompatible_resume_fails_before_decoding(trainer, setting, value):
    trainer._validate()
    trainer.calls = 0
    OmegaConf.update(trainer.config, setting, value)
    with pytest.raises(ValueError, match="Incompatible|validation_shuffle"):
        trainer._validate()
    assert trainer.calls == 0


def test_training_validation_never_reuses_evaluation_state(trainer):
    trainer._validate()
    trainer.config.trainer.val_only = False
    trainer.calls = 0
    trainer._validate()
    assert trainer.calls == 4


def test_resume_requires_output_directory(trainer):
    trainer.config.trainer.validation_data_dir = None
    with pytest.raises(ValueError, match="requires trainer.validation_data_dir"):
        trainer._validate()


def test_full_dataset_batches_resume_without_splitting_groups(trainer):
    trainer.val_dataloaders = [
        DataLoader(loader.dataset, batch_size=len(loader.dataset), collate_fn=collate)
        for loader in trainer.val_dataloaders
    ]
    trainer.fail_at = 2
    with pytest.raises(RuntimeError, match="interrupted"):
        trainer._validate()
    trainer.calls, trainer.fail_at = 0, None
    trainer._validate()
    assert trainer.calls == 1


def test_changed_dataset_order_and_batch_size_are_rejected(trainer):
    trainer._validate()
    trainer.val_dataloaders[0] = DataLoader(Samples("first"), batch_size=3, collate_fn=collate)
    with pytest.raises(ValueError, match="Incompatible"):
        trainer._validate()


def test_corrupt_checkpoint_is_not_silently_ignored(trainer, tmp_path):
    trainer._validate()
    (tmp_path / "_resume/0.json").write_text('{"version":')
    with pytest.raises(json.JSONDecodeError):
        trainer._validate()


def test_interrupted_checkpoint_write_keeps_previous_dataset(trainer, monkeypatch, tmp_path):
    trainer._validate()
    path = tmp_path / "_resume/0.json"
    previous = path.read_bytes()
    checkpoint = ValidationCheckpoint(trainer.config, trainer.val_dataloaders, 0)

    def fail_replace(*args):
        raise OSError("disk failure")

    monkeypatch.setattr("verl.trainer.ppo.validation_checkpoint.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        checkpoint.save(1, {})
    assert path.read_bytes() == previous
    assert not list(path.parent.glob("*.tmp"))

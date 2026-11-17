"""Verify SFT Blob commit ordering and remote dataloader resume without Azure."""

import copy
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
import torch
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from recipe.phimm.sft import checkpoint


def handler(tmp_path, *, resume_mode="auto", resume_from_path=None):
    engine = Mock()
    engine.save_checkpoint.side_effect = lambda **kwargs: Path(kwargs["local_path"]).mkdir(parents=True, exist_ok=True)
    loader = MagicMock()
    loader.__len__.return_value = 3
    loader.state_dict.return_value = {"step": 10}
    (tmp_path / "effective_config.yaml").write_text("model: verified\n")
    result = checkpoint.ASRSFTCheckpointHandler(
        engine=engine,
        train_dataloader=loader,
        remote_dir="az://account/data/sft",
        default_local_dir=str(tmp_path),
        resume_mode=resume_mode,
        resume_from_path=resume_from_path,
        mode=checkpoint.OrchestrationMode.RAY,
        lora_train_meta={"r": 320, "lora_alpha": 640, "task_type": "CAUSAL_LM"},
    )
    return result, engine, loader


def test_sft_publishes_worker_shards_and_state_before_remote_tracker(tmp_path, monkeypatch):
    instance, engine, _ = handler(tmp_path)
    uploads = []

    def upload(local, remote):
        engine.save_checkpoint.assert_called_once()
        assert Path(local).is_file()
        uploads.append(remote)

    monkeypatch.setattr(checkpoint, "upload_path", upload)
    instance.save_checkpoint(10)
    engine.save_checkpoint.assert_called_once_with(
        local_path=str(tmp_path / "global_step_10"),
        hdfs_path="az://account/data/sft/global_step_10",
        global_step=10,
        max_ckpt_to_keep=None,
    )
    assert uploads == [
        "az://account/data/sft/global_step_10/data_0.pt",
        "az://account/data/sft/global_step_10/lora_train_meta.json",
        "az://account/data/sft/effective_config.yaml",
        "az://account/data/sft/latest_checkpointed_iteration.txt",
    ]


def test_failed_shard_save_never_commits_remote_tracker(tmp_path, monkeypatch):
    instance, engine, _ = handler(tmp_path)
    engine.save_checkpoint.side_effect = RuntimeError("worker upload failed")
    upload = Mock()
    monkeypatch.setattr(checkpoint, "upload_path", upload)
    with pytest.raises(RuntimeError, match="worker upload failed"):
        instance.save_checkpoint(10)
    upload.assert_not_called()


def test_remote_explicit_resume_restores_model_and_dataloader(tmp_path, monkeypatch):
    remote = "az://account/data/sft/global_step_10"
    instance, engine, loader = handler(tmp_path, resume_mode="resume_path", resume_from_path=remote)
    state = tmp_path / "data_0.pt"
    torch.save({"step": 10}, state)
    monkeypatch.setattr(checkpoint.bf, "exists", lambda path: path == remote + "/data_0.pt")
    monkeypatch.setattr(checkpoint, "copy_to_local", lambda path: str(state))
    assert instance.load_checkpoint() == 10
    engine.load_checkpoint.assert_called_once_with(remote)
    loader.load_state_dict.assert_called_once_with({"step": 10})


def test_missing_remote_resume_state_fails_before_loading_model(tmp_path, monkeypatch):
    instance, engine, _ = handler(
        tmp_path, resume_mode="resume_path", resume_from_path="az://account/data/sft/global_step_10"
    )
    monkeypatch.setattr(checkpoint.bf, "exists", lambda path: False)
    with pytest.raises(ValueError, match="dataloader state"):
        instance.load_checkpoint()
    engine.load_checkpoint.assert_not_called()


def stateful_loader(num_workers, *, samples=869, shuffle=False):
    dataset = list(range(samples))
    sampler = DistributedSampler(dataset, shuffle=shuffle, num_replicas=1, rank=0, drop_last=True)
    loader = StatefulDataLoader(dataset, batch_size=2, sampler=sampler, num_workers=num_workers, drop_last=True)
    return sampler, loader


def restore_actual_loader(tmp_path, monkeypatch, state, loader, step, remote):
    path = tmp_path / f"global_step_{step}"
    path.mkdir(exist_ok=True)
    state_path = path / "data_0.pt"
    torch.save(state, state_path)
    source = f"az://account/data/sft/global_step_{step}" if remote else str(path)
    instance, engine, _ = handler(tmp_path, resume_mode="resume_path", resume_from_path=source)
    instance.train_dataloader = loader
    monkeypatch.setattr(checkpoint.bf, "exists", lambda path: path == source + "/data_0.pt")
    monkeypatch.setattr(checkpoint, "copy_to_local", lambda path: str(state_path))
    before = hashlib.sha256(state_path.read_bytes()).hexdigest()
    assert instance.load_checkpoint() == step
    assert hashlib.sha256(state_path.read_bytes()).hexdigest() == before
    engine.load_checkpoint.assert_called_once_with(source)
    return instance


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("num_workers", [0, 1])
@pytest.mark.parametrize("shuffle", [False, True])
def test_actual_epoch_boundary868_resumes869_through1736(tmp_path, monkeypatch, capsys, remote, num_workers, shuffle):
    sampler, loader = stateful_loader(num_workers, shuffle=shuffle)
    assert len(loader) == 434
    sampler.set_epoch(0)
    list(loader)
    sampler.set_epoch(1)
    iterator = iter(loader)
    for _ in range(434):
        next(iterator)
    saved = copy.deepcopy(loader.state_dict())
    assert saved["_iterator_finished"] is False
    new_sampler, resumed = stateful_loader(num_workers, shuffle=shuffle)
    instance = restore_actual_loader(tmp_path, monkeypatch, saved, resumed, 868, remote)
    global_step = instance.resume_global_step
    observed = []
    new_steps = []
    for epoch in range(global_step // len(resumed), 4):
        new_sampler.set_epoch(epoch)
        batches = [batch.tolist() for batch in resumed]
        expected_sampler, expected = stateful_loader(0, shuffle=shuffle)
        expected_sampler.set_epoch(epoch)
        assert batches == [batch.tolist() for batch in expected]
        observed.append(len(batches))
        for _ in batches:
            global_step += 1
            new_steps.append(global_step)
    assert observed == [434, 434]
    assert global_step == 1736
    assert new_steps == list(range(869, 1737))
    assert saved["_iterator_finished"] is False
    log = capsys.readouterr().out
    assert '"resume_step": 868' in log and '"first_new_optimizer_step": 869' in log
    assert '"saved_yielded": 434' in log and '"normalized_iterator_finished": true' in log


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("num_workers", [0, 1])
def test_actual_mid_epoch_resume_preserves_exact_remaining_batches(tmp_path, monkeypatch, capsys, remote, num_workers):
    sampler, loader = stateful_loader(num_workers, samples=11, shuffle=True)
    sampler.set_epoch(1)
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    saved = copy.deepcopy(loader.state_dict())
    expected_remaining = [batch.tolist() for batch in iterator]
    new_sampler, resumed = stateful_loader(num_workers, samples=11, shuffle=True)
    restore_actual_loader(tmp_path, monkeypatch, saved, resumed, 7, remote)
    new_sampler.set_epoch(1)
    assert [batch.tolist() for batch in resumed] == expected_remaining
    assert len(expected_remaining) == 3
    assert "ASR_SFT_RESUME_BOUNDARY=" not in capsys.readouterr().out


@pytest.mark.parametrize("state", [{"_num_yielded": 2}, {"_snapshot":{"_snapshot_step":2},"_steps_since_snapshot":0}, {}])
def test_epoch_boundary_rejects_unverified_or_incomplete_state(tmp_path, state):
    instance, _, loader = handler(tmp_path)
    instance.resume_global_step = 6
    loader.__len__.return_value = 3
    with pytest.raises(ValueError,match="epoch.boundary|epoch boundary"):
        instance._normalize_epoch_boundary_state(state)

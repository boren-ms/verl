"""Resuming exactly at an epoch boundary continues with the next epoch."""

import torch
from recipe.phimm.agentic.resume import restore_dataloader
from torch.utils.data import RandomSampler
from torchdata.stateful_dataloader import StatefulDataLoader


def _loader():
    data = list(range(10))
    return StatefulDataLoader(
        data, batch_size=2, sampler=RandomSampler(data, generator=torch.Generator().manual_seed(1))
    )


def _run(steps):
    """Save after the given number of batches; return state and the uninterrupted continuation."""
    loader = _loader()
    iterator = iter(loader)
    for _ in range(steps):
        next(iterator)
    state = loader.state_dict()
    rest = [batch.tolist() for batch in iterator]
    next_epoch = [batch.tolist() for batch in loader]
    return state, rest, next_epoch


def test_epoch_boundary_resume_matches_uninterrupted_next_epoch():
    state, rest, next_epoch = _run(5)
    assert rest == []
    resumed = _loader()
    assert restore_dataloader(resumed, state, global_steps=5) is True
    assert [batch.tolist() for batch in resumed] == next_epoch


def test_mid_epoch_resume_keeps_remaining_batches():
    state, rest, _ = _run(2)
    resumed = _loader()
    assert restore_dataloader(resumed, state, global_steps=2) is False
    assert [batch.tolist() for batch in resumed] == rest

"""Small CPU checks of training order, legacy state and speculative prefetch."""

import ast
import copy
import logging
from pathlib import Path

import pytest
import ray
import torch
from ray.cluster_utils import Cluster
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from recipe.phimm.data.training_loader import (
    DistributedTrainingLoader,
    SparseRows,
    retry_indices,
    restore_training_loader_state,
)
from scripts.validate_asr_generation_loading import AudioValidationDataset, assert_equal, identity_collate


@pytest.fixture(scope="module")
def nodes():
    cluster = Cluster()
    try:
        for rank in range(2):
            cluster.add_node(
                num_cpus=2, include_dashboard=False if rank == 0 else None, object_store_memory=80 * 1024 * 1024
            )
        ray.init(address=cluster.address, logging_level="ERROR")
        yield sorted(node["NodeID"] for node in ray.nodes() if node["Alive"])
    finally:
        ray.shutdown()
        cluster.shutdown()


def training_dataset(rows=17):
    ds = AudioValidationDataset(rows)
    ds.is_training = True
    ds.max_audio_load_retries = 2
    return ds


def sampler(ds, shuffle=True):
    return RandomSampler(ds, generator=torch.Generator().manual_seed(19)) if shuffle else SequentialSampler(ds)


def distributed(ds, nodes, *, shuffle=True, num_workers=0, batch_size=4, drop_last=True):
    return DistributedTrainingLoader(
        ds,
        batch_size,
        sampler(ds, shuffle),
        identity_collate,
        nodes,
        workers_per_node=1,
        prefetch_batches=3,
        timeout_s=120,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers else None,
        drop_last=drop_last,
    )


def local(ds, *, shuffle=True, num_workers=0, batch_size=4, drop_last=True):
    return StatefulDataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler(ds, shuffle),
        num_workers=num_workers,
        prefetch_factor=2 if num_workers else None,
        collate_fn=identity_collate,
        drop_last=drop_last,
    )


def ids(batches):
    return [[row["id"] for row in batch] for batch in batches]


def test_random_order_features_two_epochs_and_drop_last(nodes):
    ds = training_dataset()
    original = local(ds)
    loader = distributed(ds, nodes)
    try:
        expected = [list(original), list(original)]
        actual = [list(loader), list(loader)]
        assert_equal(actual, expected)
        assert all(len(epoch) == 4 for epoch in actual)
        assert ids(actual[0]) != ids(actual[1])
        assert {loc["node_id"] for loc in loader.locations} == set(nodes)
        assert not loader.workers
    finally:
        loader.close()


@pytest.mark.parametrize("num_workers", [0, 2])
def test_legacy_checkpoint_resume_with_original_worker_layout(nodes, num_workers):
    ds = training_dataset()
    original = local(ds, num_workers=num_workers)
    iterator = iter(original)
    next(iterator)
    snapshot = copy.deepcopy(original.state_dict())
    expected = list(iterator)
    loader = distributed(ds, nodes, num_workers=num_workers)
    try:
        loader.load_state_dict(snapshot)
        assert_equal(list(loader), expected)
    finally:
        loader.close()


def test_prefetch_checkpoint_restarts_at_consumed_batch_and_can_restore_local(nodes):
    ds = training_dataset(29)
    loader = distributed(ds, nodes)
    iterator = iter(loader)
    first = next(iterator)
    snapshot = loader.state_dict()
    # Three future batches are speculative; none may count as consumed.
    iterator.close()
    original = local(ds)
    restore_training_loader_state(original, snapshot)
    expected = list(original)
    try:
        assert_equal(list(loader), expected)
        resumed = distributed(ds, nodes)
        try:
            resumed.load_state_dict(snapshot)
            assert_equal(list(resumed), expected)
        finally:
            resumed.close()
        baseline = list(local(ds))
        assert ids([first, *expected]) == ids(baseline)
    finally:
        loader.close()


def test_partial_batch_when_drop_last_is_disabled(nodes):
    ds = training_dataset()
    loader = distributed(ds, nodes, shuffle=False, drop_last=False)
    try:
        batches = list(loader)
        assert [len(batch) for batch in batches] == [4, 4, 4, 4, 1]
        assert_equal(batches, list(local(ds, shuffle=False, drop_last=False)))
    finally:
        loader.close()


def test_training_retry_uses_original_global_candidates():
    # Execute the production retry function against a complete and sparse table.
    source = Path(__file__).parents[3] / "recipe/phimm/data/rl_dataset.py"
    node = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_audio_with_retries"
    )
    namespace = {"logger": logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    rows = [{"audio_path": f"{i}.wav"} for i in range(101)]
    chosen = retry_indices(93, len(rows), 8)
    sparse = SparseRows(len(rows), {index: rows[index] for index in chosen})

    def load(row, max_dur):
        if row["audio_path"] != f"{chosen[3]}.wav":
            raise FileNotFoundError(row["audio_path"])
        return "waveform"

    fn = namespace["_load_audio_with_retries"]
    assert fn(rows, 93, None, 8, load, (OSError,)) == fn(sparse, 93, None, 8, load, (OSError,))
    assert len(sparse) == 101
    with pytest.raises(KeyError):
        sparse[0]


def test_custom_sampler_and_mutating_dataset_are_rejected():
    ds = training_dataset()
    with pytest.raises(ValueError, match="samplers only"):
        DistributedTrainingLoader(ds, 4, [0, 1], identity_collate, ["node"])
    ds.on_batch_end = lambda **kwargs: None
    with pytest.raises(ValueError, match="on_batch_end"):
        DistributedTrainingLoader(ds, 4, sampler(ds), identity_collate, ["node"])


def test_second_epoch_checkpoint_keeps_random_generator_state(nodes):
    ds = training_dataset()
    loader = distributed(ds, nodes)
    list(loader)
    iterator = iter(loader)
    first = next(iterator)
    snapshot = loader.state_dict()
    expected = list(iterator)
    restored_local = local(ds)
    restore_training_loader_state(restored_local, snapshot)
    assert_equal(list(restored_local), expected)
    restored = distributed(ds, nodes)
    try:
        restored.load_state_dict(snapshot)
        assert_equal(list(restored), expected)
        baseline = local(ds)
        list(baseline)
        assert ids([first, *expected]) == ids(list(baseline))
    finally:
        restored.close()
        loader.close()


def test_failed_prefetch_does_not_commit_or_skip_inputs(nodes):
    ds = training_dataset()
    saved = ds.ds[4]["wav"]
    ds.ds[4]["wav"] = b"bad audio"
    loader = distributed(ds, nodes, shuffle=False)
    iterator = iter(loader)
    assert ids([next(iterator)]) == [[0, 1, 2, 3]]
    with pytest.raises(ray.exceptions.RayTaskError):
        next(iterator)
    assert not loader.workers
    ds.ds[4]["wav"] = saved
    try:
        expected = list(local(ds, shuffle=False))[1:]
        assert_equal(list(loader), expected)
    finally:
        loader.close()

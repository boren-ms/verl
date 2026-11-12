"""Small real-Ray checks; no CUDA, model weights, or network audio required."""

import time

import pytest
import ray
from ray.cluster_utils import Cluster

from recipe.phimm.data.generation_loader import (
    DistributedGenerationLoader,
    GenerationLoadWorker,
    rollout_node_ids,
    portable_audio_row,
)
from scripts.validate_asr_generation_loading import (
    AudioValidationDataset,
    assert_actors_stopped,
    identity_collate,
    validate,
)


@pytest.fixture(scope="module")
def nodes():
    cluster = Cluster()
    try:
        for index in range(2):
            cluster.add_node(
                num_cpus=2, include_dashboard=False if index == 0 else None, object_store_memory=80 * 1024 * 1024
            )
        ray.init(address=cluster.address, logging_level="ERROR")
        yield sorted(node["NodeID"] for node in ray.nodes() if node["Alive"])
    finally:
        ray.shutdown()
        cluster.shutdown()


def test_two_nodes_waveforms_features_resume_and_failure(nodes):
    report = validate(nodes, rows=37, batch_size=8)
    assert report["cases"][0]["batch_sizes"] == [8, 8, 8, 8, 5]
    assert report["cases"][1]["batch_sizes"] == [8, 8, 5]


def test_explicit_close_releases_prefetched_actors(nodes):
    loader = DistributedGenerationLoader(
        AudioValidationDataset(9), 4, nodes, identity_collate, workers_per_node=1, timeout_s=120
    )
    iterator = iter(loader)
    assert [row["id"] for row in next(iterator)] == [0, 1, 2, 3]
    handles = list(loader.workers)
    iterator.close()
    assert loader.closed and not loader.workers
    assert_actors_stopped(handles)


def test_slow_load_times_out_and_cleans_up(nodes):
    class SlowWorker(GenerationLoadWorker):
        def load(self, rows):
            time.sleep(0.5)
            return super().load(rows)

    loader = DistributedGenerationLoader(
        AudioValidationDataset(9), 4, nodes, identity_collate, workers_per_node=1, timeout_s=120, worker_cls=SlowWorker
    )
    loader.timeout_s = 0.01
    with pytest.raises(ray.exceptions.GetTimeoutError):
        list(loader)
    assert loader.closed and not loader.workers


def test_only_allocated_placement_group_nodes_are_selected(nodes):
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    pg = placement_group([{"CPU": 1}], strategy="STRICT_PACK")
    try:
        ray.get(pg.ready(), timeout=30)
        selected = rollout_node_ids(type("Pool", (), {"get_placement_groups": lambda self: [pg]})())
        assert len(selected) == 1 and selected[0] in nodes
        node_id = ray.get(
            ray.remote(lambda: ray.get_runtime_context().get_node_id())
            .options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg),
            )
            .remote(),
            timeout=30,
        )
        assert selected == [node_id]
    finally:
        remove_placement_group(pg)


@pytest.mark.parametrize(
    "options",
    [
        {"workers_per_node": 0},
        {"prefetch_batches": 0},
        {"batch_size": 0},
        {"timeout_s": 0},
        {"start_index": -1},
        {"start_index": 10},
        {"node_ids": []},
        {"node_ids": ["duplicate", "duplicate"]},
    ],
)
def test_invalid_options_fail_before_creating_actors(options):
    kwargs = {"batch_size": 4, "node_ids": ["test"], "collate_fn": identity_collate, **options}
    with pytest.raises(ValueError):
        DistributedGenerationLoader(AudioValidationDataset(9), **kwargs)


def test_training_dataset_is_rejected():
    dataset = AudioValidationDataset(9)
    dataset.is_training = True
    with pytest.raises(ValueError, match="no sample substitution"):
        DistributedGenerationLoader(dataset, 4, ["test"], identity_collate)


def test_cache_paths_are_portable_without_changing_selectors_or_source_row(monkeypatch):
    monkeypatch.setenv("VERL_AUDIO_CACHE_ROOT", "/tmp/private-cache")
    row = {
        "audio_path": "/tmp/private-cache/speech/a.audio!33:77#0:1;/tmp/private-cache/speech/b.wav",
        "audio_chunk": "/tmp/private-cache/speech/a.audio:9:2",
        "audio_file": "shared/file.wav",
        "text": "reference",
        "extra_info": {"id": 4},
    }
    portable = portable_audio_row(row)
    assert (
        portable["audio_path"]
        == "az://orngwus2cresco/data/speech/a.audio!33:77#0:1;az://orngwus2cresco/data/speech/b.wav"
    )
    assert portable["audio_chunk"] == "az://orngwus2cresco/data/speech/a.audio:9:2"
    assert portable["audio_file"] == row["audio_file"]
    assert portable["extra_info"] == row["extra_info"]
    assert row["audio_path"].startswith("/tmp/private-cache/")

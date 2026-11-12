"""Small CPU-only ASR training loader validation; no optimizer or GPU inference."""

import argparse
import copy
import json
from pathlib import Path

import ray
import torch
from torch.utils.data import RandomSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from recipe.phimm.data.training_loader import DistributedTrainingLoader
from scripts.validate_asr_generation_loading import (
    AudioValidationDataset,
    assert_actors_stopped,
    assert_equal,
    identity_collate,
    real_dataset,
)


def validate(dataset, nodes, batch_size=4, num_workers=1):
    dataset.is_training = True
    if not hasattr(dataset, "max_audio_load_retries"):
        dataset.max_audio_load_retries = 2

    def sampler():
        return RandomSampler(dataset, generator=torch.Generator().manual_seed(19))

    local_options = dict(num_workers=num_workers, prefetch_factor=2 if num_workers else None, drop_last=True)
    original = StatefulDataLoader(
        dataset, batch_size=batch_size, sampler=sampler(), collate_fn=identity_collate, **local_options
    )
    epoch1, epoch2 = list(original), list(original)

    def distributed():
        return DistributedTrainingLoader(
            dataset,
            batch_size,
            sampler(),
            identity_collate,
            nodes,
            workers_per_node=1,
            prefetch_batches=3,
            timeout_s=120,
            **local_options,
        )

    first = distributed()
    iterator = iter(first)
    try:
        first_batch = next(iterator)
        state = copy.deepcopy(first.state_dict())
        torch.save(state, "training-loader-data.pt")
        assert_equal(first_batch, epoch1[0])
        handles = list(first.workers)
        locations = list(first.locations)
    finally:
        iterator.close()
        first.close()
    assert_actors_stopped(handles)

    resumed = distributed()
    try:
        resumed.load_state_dict(torch.load("training-loader-data.pt", weights_only=False))
        actual = []
        seen = set()
        for batch in resumed:
            actual.append(batch)
            seen.update(stat["node_id"] for stat in resumed.stats)
        assert_equal(actual, epoch1[1:])
        assert_equal(list(resumed), epoch2)
        assert seen == set(nodes)
        assert not resumed.workers
    finally:
        resumed.close()

    # An existing local checkpoint must restore without changing its layout.
    legacy = StatefulDataLoader(
        dataset, batch_size=batch_size, sampler=sampler(), collate_fn=identity_collate, **local_options
    )
    iterator = iter(legacy)
    next(iterator)
    legacy_state = legacy.state_dict()
    expected = list(iterator)
    migrated = distributed()
    try:
        migrated.load_state_dict(legacy_state)
        assert_equal(list(migrated), expected)
    finally:
        migrated.close()
    return {
        "input_rows": len(dataset),
        "batch_size": batch_size,
        "batches_per_epoch": len(original),
        "drop_last": True,
        "locations": locations,
        "nodes_seen": sorted(seen),
        "two_epochs_match_local": True,
        "prefetched_resume_match": True,
        "legacy_local_checkpoint_match": True,
        "actor_cleanup": True,
        "scope": "Training sampler and actual ASR loading only; no optimizer or GPU inference",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address")
    parser.add_argument("--node-ids", nargs="+")
    parser.add_argument("--rows", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--input-jsonl")
    parser.add_argument("--processor-path")
    parser.add_argument("--report", type=Path, default=Path("tmp/distributed-training-loading/validation.json"))
    args = parser.parse_args()
    if args.address and not args.node_ids:
        parser.error("--address requires the exact test --node-ids")
    if bool(args.input_jsonl) != bool(args.processor_path):
        parser.error("--input-jsonl and --processor-path must be used together")
    if args.batch_size <= 0 or args.rows < args.batch_size * 3 or args.num_workers < 0:
        parser.error("Use a positive batch size, at least three batches and nonnegative num-workers")
    cluster = None
    try:
        if args.address:
            ray.init(address=args.address, logging_level="ERROR")
        else:
            from ray.cluster_utils import Cluster

            cluster = Cluster()
            for rank in range(2):
                cluster.add_node(
                    num_cpus=2, include_dashboard=False if rank == 0 else None, object_store_memory=80 * 1024 * 1024
                )
            ray.init(address=cluster.address, logging_level="ERROR")
        nodes = args.node_ids or sorted(node["NodeID"] for node in ray.nodes() if node["Alive"])
        if len(nodes) < 2:
            raise ValueError("Two nodes are required")
        torch.set_num_threads(1)
        dataset = (
            real_dataset(args.input_jsonl, args.processor_path, args.rows)
            if args.input_jsonl
            else AudioValidationDataset(args.rows)
        )
        report = validate(dataset, nodes, args.batch_size, args.num_workers)
        report["input_jsonl"] = args.input_jsonl
        report["topology"] = "physical cluster" if args.address else "two logical nodes on one physical host"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        ray.shutdown()
        if cluster is not None:
            cluster.shutdown()


if __name__ == "__main__":
    main()

"""Bounded, ordered CPU preprocessing for standalone ASR generation.

Only small metadata slices cross to loaders. Audio reads, cache population and
RLHFDataset.__getitem__ (including processor features) run on rollout nodes.
The full Arrow dataset stays in the coordinator; its local mmap paths are never
sent to another host. No GPUs or placement-group bundles are consumed.
"""

import copy
import math
import os
import time
from collections import deque

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def portable_audio_row(row):
    """Undo coordinator cache paths before sending metadata to another host."""
    row = dict(row)
    root = os.path.expanduser(os.getenv("VERL_AUDIO_CACHE_ROOT", "~/data")).rstrip("/") + "/"
    for key in ("audio_path", "audio_chunk", "audio_file"):
        if isinstance(row.get(key), str):
            parts = row[key].split(";")
            row[key] = ";".join(
                "az://orngwus2cresco/data/" + part.strip()[len(root) :] if part.strip().startswith(root) else part
                for part in parts
            )
    return row


class GenerationLoadWorker:
    def __init__(self, dataset):
        import torch
        from recipe.phimm.utils.audio import set_chunk_load_mode

        torch.set_num_threads(1)
        set_chunk_load_mode(dataset.config.get("chunk_load_mode") or "indexed")
        self.dataset = dataset
        self.node_id = ray.get_runtime_context().get_node_id()

    def ready(self):
        return {"node_id": self.node_id, "pid": os.getpid()}

    def load(self, rows):
        started = time.monotonic()
        self.dataset.ds = rows
        try:
            samples = [self.dataset[i] for i in range(len(rows))]
        finally:
            self.dataset.ds = None
        return samples, {**self.ready(), "rows": len(samples), "load_s": time.monotonic() - started}


def rollout_node_ids(resource_pool):
    """Use exactly this job's allocated nodes, excluding unrelated Ray nodes."""
    nodes = set()
    for pg in resource_pool.get_placement_groups():
        table = ray.util.placement_group_table(pg)
        nodes.update(table["bundles_to_node_id"].values())
    if not nodes or "" in nodes:
        raise RuntimeError("Cannot determine rollout nodes from placement groups")
    return sorted(nodes)


class DistributedGenerationLoader:
    """One-shot iterable with hard node affinity and a bounded batch window.

    Each global batch is split across all CPU actors. Results are collected in
    source order, even when actors finish out of order. start_index refers to
    the original concatenated dataset and therefore preserves JSONL resume.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        node_ids,
        collate_fn,
        *,
        start_index=0,
        workers_per_node=4,
        prefetch_batches=2,
        timeout_s=300,
        worker_cls=GenerationLoadWorker,
    ):
        if batch_size <= 0 or workers_per_node <= 0 or prefetch_batches <= 0 or timeout_s <= 0:
            raise ValueError("batch size, workers_per_node, prefetch_batches and timeout_s must be positive")
        if not node_ids or len(set(node_ids)) != len(node_ids):
            raise ValueError("node_ids must be nonempty and unique")
        if not 0 <= start_index <= len(dataset):
            raise ValueError("start_index is outside the generation dataset")
        if dataset.is_training:
            raise ValueError("Distributed generation requires an evaluation dataset (no sample substitution)")
        self.dataset = dataset
        self.batch_size = batch_size
        self.start_index = start_index
        self.collate_fn = collate_fn
        self.prefetch_batches = prefetch_batches
        self.timeout_s = timeout_s
        self.workers = []
        self.closed = False
        self.stats = []
        self.started = False
        # Do not pickle Arrow tables: their backing files live on the main node.
        template = copy.copy(dataset)
        template.ds = None
        template_ref = ray.put(template)
        actor_cls = ray.remote(num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0)(worker_cls)
        try:
            for _ in range(workers_per_node):
                for node_id in node_ids:
                    self.workers.append(
                        actor_cls.options(
                            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
                        ).remote(template_ref)
                    )
            self.locations = ray.get([worker.ready.remote() for worker in self.workers], timeout=timeout_s)
            if [loc["node_id"] for loc in self.locations] != node_ids * workers_per_node:
                raise RuntimeError("Generation loader node affinity verification failed")
            print(f"Distributed generation loaders: {self.locations}", flush=True)
        except BaseException:
            self.close()
            raise

    def __len__(self):
        return math.ceil((len(self.dataset) - self.start_index) / self.batch_size)

    def _submit(self, start):
        end = min(start + self.batch_size, len(self.dataset))
        worker_count = min(end - start, len(self.workers))
        refs = []
        for index, worker in enumerate(self.workers[:worker_count]):
            shard_start = start + (end - start) * index // worker_count
            shard_end = start + (end - start) * (index + 1) // worker_count
            rows = [portable_audio_row(self.dataset.ds[i]) for i in range(shard_start, shard_end)]
            refs.append(worker.load.remote(rows))
        return refs

    def __iter__(self):
        if self.started or self.closed:
            raise RuntimeError("DistributedGenerationLoader is a one-shot iterable")
        self.started = True
        pending = deque()
        starts = iter(range(self.start_index, len(self.dataset), self.batch_size))
        try:
            for _ in range(self.prefetch_batches):
                start = next(starts, None)
                if start is None:
                    break
                pending.append(self._submit(start))
            while pending and not self.closed:
                parts = ray.get(pending.popleft(), timeout=self.timeout_s)
                stats = [stat for _, stat in parts]
                self.stats = stats  # Keep only the latest batch, not a corpus-sized history.
                rows_by_node = {}
                for stat in stats:
                    rows_by_node[stat["node_id"]] = rows_by_node.get(stat["node_id"], 0) + stat["rows"]
                print(
                    f"Generation loading rows_by_node={rows_by_node} "
                    f"max_worker_load_s={max(stat['load_s'] for stat in stats):.3f}",
                    flush=True,
                )
                batch = self.collate_fn([sample for samples, _ in parts for sample in samples])
                # Release fetched shards before scheduling another batch.
                del parts
                start = next(starts, None)
                if start is not None and not self.closed:
                    pending.append(self._submit(start))
                yield batch
        finally:
            self.close()

    def close(self):
        if not self.closed:
            self.closed = True
            for worker in self.workers:
                ray.kill(worker, no_restart=True)
            self.workers.clear()

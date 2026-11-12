"""Stateful index sampling with node-pinned ASR audio preprocessing."""

import copy
import time
from collections import deque

import ray
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from recipe.phimm.data.generation_loader import GenerationLoadWorker, portable_audio_row


SAMPLER_GENERATOR_KEY = "_asr_training_sampler_epoch_generator"


def restore_training_loader_state(loader, state):
    """Restore our extra RNG snapshot when switching back to a local loader."""
    generator = getattr(loader.sampler, "generator", None)
    if SAMPLER_GENERATOR_KEY in state and generator is not None:
        generator.set_state(state[SAMPLER_GENERATOR_KEY].clone())
    loader.load_state_dict(state)


class IndexDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return int(index)


def index_collate(indices):
    return indices


class SparseRows:
    """Keep global length/index semantics without shipping the whole Arrow table."""

    def __init__(self, size, rows):
        self.size = size
        self.rows = rows

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # HF Dataset returns a fresh row; retries must not share mutated dicts.
        return copy.deepcopy(self.rows[int(index)])


def retry_indices(index, size, retries):
    stride = max(1, size // (retries + 1))
    return [(int(index) + offset * stride) % size for offset in range(retries + 1)]


class TrainingLoadWorker(GenerationLoadWorker):
    def load(self, indices, rows, size):
        started = time.monotonic()
        self.dataset.ds = SparseRows(size, rows)
        try:
            samples = [self.dataset[index] for index in indices]
        finally:
            self.dataset.ds = None
        return samples, {**self.ready(), "rows": len(samples), "load_s": time.monotonic() - started}


class DistributedTrainingLoader:
    """Keep torchdata sampling/state; distribute only deterministic ASR __getitem__.

    Index-loader worker count/options match the original DataLoader, including
    legacy checkpoint layout. The committed snapshot advances only when a
    processed batch is delivered to the trainer, never for speculative work.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        sampler,
        collate_fn,
        node_ids,
        *,
        workers_per_node=4,
        prefetch_batches=2,
        timeout_s=300,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory=False,
        drop_last=True,
        worker_cls=TrainingLoadWorker,
    ):
        if batch_size <= 0 or workers_per_node <= 0 or prefetch_batches <= 0 or timeout_s <= 0:
            raise ValueError("Training batch size, worker count, prefetch and timeout must be positive")
        if not isinstance(sampler, (RandomSampler, SequentialSampler)):
            raise ValueError("Distributed ASR training currently supports random/sequential samplers only")
        if isinstance(sampler, RandomSampler) and sampler.generator is None:
            raise ValueError("Distributed random sampling requires an explicit torch.Generator")
        if hasattr(dataset, "on_batch_end"):
            raise ValueError("Distributed ASR training does not support datasets changed by on_batch_end")
        if not hasattr(dataset, "ds") or not hasattr(dataset, "is_training"):
            raise ValueError("Distributed ASR training requires the phimm RLHFDataset interface")
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.node_ids = node_ids  # Callable is resolved after GPU placement groups exist.
        self.workers_per_node = workers_per_node
        self.prefetch_batches = prefetch_batches
        self.timeout_s = timeout_s
        self.worker_cls = worker_cls
        self.index_loader = StatefulDataLoader(
            IndexDataset(len(dataset)),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=index_collate,
        )
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.workers = []
        self.locations = []
        self.stats = []
        self.last_input_wait_s = 0.0
        self._committed_state = None
        self._active = False
        self._epoch_generator_state = None
        self._resume_pending = False
        self.closed = False

    @property
    def sampler(self):
        return self.index_loader.sampler

    def __len__(self):
        return len(self.index_loader)

    def _snapshot(self):
        generator = getattr(self.sampler, "generator", None)
        if self._epoch_generator_state is None and generator is not None:
            self._epoch_generator_state = generator.get_state().clone()
        state = copy.deepcopy(self.index_loader.state_dict())
        if self._epoch_generator_state is not None:
            state[SAMPLER_GENERATOR_KEY] = self._epoch_generator_state.clone()
        return state

    def state_dict(self):
        return copy.deepcopy(self._committed_state) if self._committed_state is not None else self._snapshot()

    def _restore(self, state):
        state = copy.deepcopy(state)
        epoch_generator = state.pop(SAMPLER_GENERATOR_KEY, None)
        generator = getattr(self.sampler, "generator", None)
        if epoch_generator is not None and generator is not None:
            generator.set_state(epoch_generator)
        if epoch_generator is None and generator is not None:
            epoch_generator = generator.get_state().clone()
        self._epoch_generator_state = epoch_generator
        self.index_loader.load_state_dict(state)
        self._resume_pending = True

    def load_state_dict(self, state):
        if self._active:
            raise RuntimeError("Close the active training iterator before restoring its state")
        self._restore(state)
        self._committed_state = copy.deepcopy(state)
        if self._epoch_generator_state is not None:
            self._committed_state[SAMPLER_GENERATOR_KEY] = self._epoch_generator_state.clone()

    def _start_workers(self):
        nodes = self.node_ids() if callable(self.node_ids) else self.node_ids
        if not nodes or len(set(nodes)) != len(nodes):
            raise ValueError("Training loader node IDs must be nonempty and unique")
        template = copy.copy(self.dataset)
        template.ds = None
        template_ref = ray.put(template)
        actor_cls = ray.remote(num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0)(self.worker_cls)
        for _ in range(self.workers_per_node):
            for node in nodes:
                self.workers.append(
                    actor_cls.options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node, soft=False),
                    ).remote(template_ref)
                )
        self.locations = ray.get([worker.ready.remote() for worker in self.workers], timeout=self.timeout_s)
        if [location["node_id"] for location in self.locations] != list(nodes) * self.workers_per_node:
            raise RuntimeError("Training loader node affinity verification failed")
        print(f"Distributed training loaders: {self.locations}", flush=True)

    def _submit(self, indices):
        size = len(self.dataset)
        retries = self.dataset.max_audio_load_retries if self.dataset.is_training else 0
        count = min(len(indices), len(self.workers))
        refs = []
        for rank, worker in enumerate(self.workers[:count]):
            shard = indices[len(indices) * rank // count : len(indices) * (rank + 1) // count]
            candidates = dict.fromkeys(
                candidate for index in shard for candidate in retry_indices(index, size, retries)
            )
            rows = {index: portable_audio_row(self.dataset.ds[index]) for index in candidates}
            refs.append(worker.load.remote(shard, rows, size))
        return refs

    def __iter__(self):
        if self.closed or self._active:
            raise RuntimeError("Training loader is closed or already has an active iterator")
        self._active = True
        exhausted = False
        try:
            self._start_workers()
            if not self._resume_pending:
                generator = getattr(self.sampler, "generator", None)
                self._epoch_generator_state = generator.get_state().clone() if generator is not None else None
            self._resume_pending = False
            if self._committed_state is None:
                self._committed_state = self._snapshot()
            source = iter(self.index_loader)
            pending = deque()

            def submit_next():
                try:
                    indices = next(source)
                except StopIteration:
                    return False
                snapshot = self._snapshot()
                pending.append((self._submit(indices), snapshot))
                return True

            for _ in range(self.prefetch_batches):
                if not submit_next():
                    break
            while pending and not self.closed:
                waiting_at = time.monotonic()
                refs, snapshot = pending.popleft()
                parts = ray.get(refs, timeout=self.timeout_s)
                self.stats = [stat for _, stat in parts]
                batch = self.collate_fn([sample for samples, _ in parts for sample in samples])
                del parts, refs
                submit_next()
                self._committed_state = snapshot
                self.last_input_wait_s = time.monotonic() - waiting_at
                rows_by_node = {}
                for stat in self.stats:
                    rows_by_node[stat["node_id"]] = rows_by_node.get(stat["node_id"], 0) + stat["rows"]
                print(
                    f"Training loading rows_by_node={rows_by_node} input_wait_s={self.last_input_wait_s:.3f}",
                    flush=True,
                )
                yield batch
            if not self.closed:
                exhausted = True
                self._committed_state = self._snapshot()
        finally:
            self._stop_workers()
            if not exhausted and self._committed_state is not None:
                self._restore(self._committed_state)
            self._active = False

    def _stop_workers(self):
        for worker in self.workers:
            ray.kill(worker, no_restart=True)
        self.workers.clear()

    def close(self):
        self.closed = True
        self._stop_workers()

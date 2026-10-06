"""Dataset-level checkpoints for eval-only validation."""

import hashlib
import json
import os
import uuid

import blobfile as bf
import numpy as np
from omegaconf import OmegaConf


def _json_default(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    raise TypeError(f"Cannot serialize validation value of type {type(value).__name__}")


class ValidationCheckpoint:
    def __init__(self, config, loaders, step, model_checkpoint=None):
        self.enabled = config.trainer.get("validation_resume", False)
        output_dir = config.trainer.get("validation_data_dir")
        if self.enabled and not output_dir:
            raise ValueError("trainer.validation_resume requires trainer.validation_data_dir")
        self.path = bf.join(output_dir, "_resume", f"{step}.json") if output_dir else None
        self.completed = 0
        self.in_progress_dataset = None
        self.completed_batches = 0
        self.loaders = loaders
        self.total = len(loaders)
        self.save_freq = config.trainer.get("validation_resume_save_freq", 0)
        if type(self.save_freq) is not int or self.save_freq < 0:
            raise ValueError("trainer.validation_resume_save_freq must be a non-negative integer")
        if not self.enabled:
            # A forced fresh run must not leave an older checkpoint reusable.
            if self.path and bf.exists(self.path):
                bf.remove(self.path)
            return
        if config.data.get("validation_shuffle", False):
            raise ValueError("trainer.validation_resume requires data.validation_shuffle=false")
        data_config = OmegaConf.to_container(config.data, resolve=True)
        for key in ("dataloader_num_workers", "prefetch_factor", "persistent_workers", "pin_memory"):
            data_config.pop(key, None)
        identity = {
            "data": data_config,
            "model": OmegaConf.to_container(config.actor_rollout_ref.model, resolve=True),
            "rollout": OmegaConf.to_container(config.actor_rollout_ref.rollout, resolve=True),
            "reward": {
                key: OmegaConf.to_container(config[key], resolve=True)
                for key in (
                    "val_reward", "reward_model", "custom_reward_function",
                    "reward_functions", "reward_function_by_data_source",
                )
                if key in config
            },
            "checkpoint": model_checkpoint,
            "datasets": [
                {
                    "size": len(loader.dataset),
                    "batch_size": loader.batch_size,
                }
                for loader in loaders
            ],
        }
        self.signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def load(self, empty_state):
        if not self.enabled or not bf.exists(self.path):
            return empty_state
        with bf.BlobFile(self.path, "r") as stream:
            checkpoint = json.load(stream)
        if checkpoint.get("version") != 1 or checkpoint.get("signature") != self.signature:
            raise ValueError(
                f"Incompatible evaluation checkpoint: {self.path} "
                f"(stored signature {checkpoint.get('signature')!r}, expected {self.signature!r}). "
                "Use a new validation_data_dir "
                "or trainer.validation_resume=false to evaluate from scratch."
            )
        completed = checkpoint.get("completed_datasets")
        if type(completed) is not int or not 0 <= completed <= self.total:
            raise ValueError(f"Invalid completed dataset count in {self.path}")
        in_progress = checkpoint.get("in_progress")
        if in_progress is not None:
            if not isinstance(in_progress, dict) or set(in_progress) != {"dataset", "completed_batches"}:
                raise ValueError(f"Invalid in-progress validation state in {self.path}")
            dataset = in_progress["dataset"]
            completed_batches = in_progress["completed_batches"]
            if (
                type(dataset) is not int
                or type(completed_batches) is not int
                or dataset != completed
                or not 0 <= dataset < self.total
                or not 0 < completed_batches <= len(self.loaders[dataset])
            ):
                raise ValueError(f"Invalid in-progress validation state in {self.path}")
            self.in_progress_dataset = dataset
            self.completed_batches = completed_batches
        state = checkpoint.get("state")
        if not isinstance(state, dict) or state.keys() != empty_state.keys():
            raise ValueError(f"Invalid validation state in {self.path}")
        self.completed = completed
        progress = (
            f" and {self.completed_batches} batches of dataset {self.in_progress_dataset + 1}"
            if self.in_progress_dataset is not None else ""
        )
        print(f"Resuming evaluation: skipping {completed}/{self.total} completed datasets{progress} from {self.path}")
        return state

    def batches(self, loaders, state):
        for index, loader in enumerate(loaders):
            if index < self.completed:
                continue
            completed_batches = self.completed_batches if index == self.in_progress_dataset else 0
            for batch_index, batch in enumerate(loader):
                if batch_index < completed_batches:
                    continue
                yield batch
                processed_batches = batch_index + 1
                if self.enabled and self.save_freq and processed_batches % self.save_freq == 0:
                    self.save(index, state, in_progress_dataset=index, completed_batches=processed_batches)
            if self.enabled:
                self.save(index + 1, state)

    def save(self, completed, state, in_progress_dataset=None, completed_batches=0):
        in_progress = None
        if in_progress_dataset is not None:
            in_progress = {
                "dataset": in_progress_dataset,
                "completed_batches": completed_batches,
            }
        payload = json.dumps(
            {
                "version": 1, "signature": self.signature,
                "completed_datasets": completed, "in_progress": in_progress, "state": state,
            },
            default=_json_default,
            ensure_ascii=False,
        )
        bf.makedirs(bf.dirname(self.path))
        if "://" in self.path:
            # Non-streaming BlobFile uploads the complete blob on close.
            with bf.BlobFile(self.path, "w", streaming=False) as stream:
                stream.write(payload)
        else:
            temporary = f"{self.path}.{uuid.uuid4().hex}.tmp"
            try:
                with open(temporary, "w", encoding="utf-8") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.remove(temporary)
        progress = (
            f", {completed_batches} batches of dataset {in_progress_dataset + 1}"
            if in_progress_dataset is not None else ""
        )
        print(f"Saved evaluation progress: {completed}/{self.total} datasets{progress} to {self.path}")

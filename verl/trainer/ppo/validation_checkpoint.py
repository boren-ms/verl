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
        self.total = len(loaders)
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
        state = checkpoint.get("state")
        if not isinstance(state, dict) or state.keys() != empty_state.keys():
            raise ValueError(f"Invalid validation state in {self.path}")
        self.completed = completed
        print(f"Resuming evaluation: skipping {completed}/{self.total} completed datasets from {self.path}")
        return state

    def batches(self, loaders, state):
        for index, loader in enumerate(loaders):
            if index < self.completed:
                continue
            yield from loader
            if self.enabled:
                self.save(index + 1, state)

    def save(self, completed, state):
        payload = json.dumps(
            {
                "version": 1, "signature": self.signature,
                "completed_datasets": completed, "state": state,
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
        print(f"Saved evaluation progress: {completed}/{self.total} datasets to {self.path}")

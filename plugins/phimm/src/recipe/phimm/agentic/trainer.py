"""ASR corpus metrics and durable artifacts on the V1 trainers."""

import json
from collections import defaultdict
from pathlib import Path

import blobfile as bf
import numpy as np
import torch
import transfer_queue as tq
from recipe.phimm.agentic.scoring import _compute_reward_metrics, aggregate_error_rates

from verl.trainer.ppo.checkpoint_callback import CheckpointCallback
from verl.trainer.ppo.v1.trainer_base import _tq_supports_checkpoint
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils.azure_io import upload_path
from verl.utils.fs import copy_to_local


class ASRCheckpointCallback(CheckpointCallback):
    def on_save(self, trainer, global_step, checkpoint_dir, async_save=False, **kwargs):
        if async_save:
            raise ValueError("ASR durable checkpoint publication requires synchronous saves")
        remote = self.config.trainer.default_hdfs_dir
        if not remote:
            return
        upload_path(Path(checkpoint_dir) / "data.pt", f"{remote}/global_step_{global_step}/data.pt")
        queue_path = Path(checkpoint_dir) / "transfer_queue"
        if trainer.trainer_mode != "sync" and _tq_supports_checkpoint():
            upload_path(queue_path, f"{remote}/global_step_{global_step}/transfer_queue")
        # All worker uploads completed before this hook. Publish the tracker last.
        upload_path(
            Path(checkpoint_dir).parent / "latest_checkpointed_iteration.txt",
            f"{remote}/latest_checkpointed_iteration.txt",
        )


class ASRTrainerMixin:
    """Share ASR metrics and artifact handling without overriding V1 scheduling."""

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["extra_fields"])
        reward_infos = defaultdict(list)
        # TransferQueue may return a tensordict LinkedList, which has no tolist().
        for tag, record in zip(batch.tags, list(data["extra_fields"]), strict=True):
            if tag.get("is_padding", False):
                continue
            for key, value in record.get("reward_extra_info", {}).items():
                reward_infos[key].append(value)
        metrics.update(_compute_reward_metrics(reward_infos))

    def _init_dataloader(self):
        super()._init_dataloader()
        if not self.config.trainer.val_only:
            expected_rows = self.config.asr.get("expected_training_rows")
            expected_steps = self.config.asr.get("expected_training_steps")
            if expected_rows is not None and len(self.train_dataset) != expected_rows:
                raise ValueError(f"ASR training row count changed: {len(self.train_dataset)}")
            if expected_steps is not None and self.total_training_steps != expected_steps:
                raise ValueError(f"ASR training target changed: {self.total_training_steps}")

    def _load_checkpoint(self):
        remote = self.config.trainer.default_hdfs_dir
        mode = self.config.trainer.resume_mode
        if mode == "disable" or not remote or not remote.startswith("az://"):
            return super()._load_checkpoint()
        marker = f"{remote}/latest_checkpointed_iteration.txt"
        if mode == "auto" and not bf.exists(marker):
            # Unique experiment root: an absent remote marker means no committed step.
            return super()._load_checkpoint()
        if mode == "resume_path":
            checkpoint = self.config.trainer.resume_from_path
            if not checkpoint.startswith("az://"):
                return super()._load_checkpoint()
            step = int(checkpoint.rstrip("/").rsplit("global_step_", 1)[-1])
        else:
            with bf.BlobFile(marker, "r") as stream:
                step = int(stream.read().strip())
            checkpoint = f"{remote}/global_step_{step}"
        self.global_steps = step
        self.actor_rollout_wg.load_checkpoint(local_path=f"{checkpoint}/actor", del_local_after_load=False)
        if self.use_critic:
            raise ValueError("This ReMax ASR adapter does not use a critic")
        state = torch.load(copy_to_local(f"{checkpoint}/data.pt"), weights_only=False)
        self.train_dataloader.load_state_dict(state)
        print(
            "ASR_RESUME_STATE="
            + json.dumps(
                {
                    "step": self.global_steps,
                    "checkpoint": checkpoint,
                    "dataloader_state_restored": True,
                    "snapshot_step": state.get("_snapshot", {}).get("_snapshot_step"),
                }
            ),
            flush=True,
        )
        if self.trainer_mode != "sync" and _tq_supports_checkpoint():
            queue_path = f"{checkpoint}/transfer_queue"
            if bf.exists(queue_path):
                # TransferQueue storage actors must see the restored snapshot on every node.
                cache_dir = str(Path(self.config.trainer.default_local_dir) / "remote_queue_cache")
                tq.load_checkpoint(copy_to_local(queue_path, cache_dir=cache_dir))

    def _write_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, global_steps):
        super()._write_generations(inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, global_steps)
        remote = self.config.asr.validation_remote_dir
        if remote:
            upload_path(Path(dump_path) / f"{global_steps}.jsonl", f"{remote}/{global_steps}.jsonl")
        # Keep the canonical per-dataset result paths used by ASR comparison tools.
        groups = {}
        filename = Path(dump_path) / f"{global_steps}.jsonl"
        with filename.open() as stream:
            for line in stream:
                row = json.loads(line)
                groups.setdefault(row["asr_data_source"], []).append(row)
        for source, rows in groups.items():
            target = Path(dump_path) / source / f"{global_steps}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if remote:
                upload_path(target, f"{remote}/{source}/{global_steps}.jsonl")

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        numeric = {
            key: values
            for key, values in reward_extra_infos_dict.items()
            if values and all(isinstance(value, (int, float)) for value in values)
        }
        metrics = super()._val_metrics_update(data_sources, sample_uids, numeric, sample_turns)
        metrics.update(aggregate_error_rates(data_sources, reward_extra_infos_dict))
        metrics = {key: value.item() if isinstance(value, np.generic) else value for key, value in metrics.items()}
        destination = Path(self.config.trainer.validation_data_dir) / f"{self.global_steps}.metrics.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"step": self.global_steps, "metrics": metrics}, indent=2))
        if self.config.asr.validation_remote_dir:
            upload_path(destination, f"{self.config.asr.validation_remote_dir}/{destination.name}")
        return metrics


class ASRTrainerSync(ASRTrainerMixin, PPOTrainerSync):
    """ASR artifacts with synchronous V1 scheduling."""


class ASRTrainerColocateAsync(ASRTrainerMixin, PPOTrainerColocateAsync):
    """ASR artifacts with colocated async V1 scheduling and partial rollout."""

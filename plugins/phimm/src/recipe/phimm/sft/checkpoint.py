"""Publish native SFT shards and dataloader state durably to Azure Blob."""

import json
from pathlib import Path

import blobfile as bf
import torch.distributed as dist

from verl.utils.azure_io import upload_path
from verl.utils.checkpoint import CheckpointHandler, OrchestrationMode
from verl.utils.fs import copy_to_local


class BlobCheckpointEngine:
    """Pass Blob destinations through to the existing FSDP shard publisher."""

    def __init__(self, engine, remote_dir):
        self.engine = engine
        self.remote_dir = remote_dir

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def save_checkpoint(self, *, local_path, global_step, max_ckpt_to_keep):
        return self.engine.save_checkpoint(
            local_path=local_path,
            hdfs_path=f"{self.remote_dir}/global_step_{global_step}",
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )


class ASRSFTCheckpointHandler(CheckpointHandler):
    """Commit the remote tracker only after worker shards and driver state exist."""

    def __init__(self, *, remote_dir=None, **kwargs):
        self.remote_dir = remote_dir.rstrip("/") if remote_dir else None
        if self.remote_dir and not self.remote_dir.startswith("az://"):
            raise ValueError("ASR SFT remote checkpoints require an az:// destination")
        if self.remote_dir:
            kwargs["engine"] = BlobCheckpointEngine(kwargs["engine"], self.remote_dir)
        super().__init__(default_hdfs_dir=None, **kwargs)
        if self.mode == OrchestrationMode.RAY:
            self.is_mp_src_rank_with_outputs = True

    def save_checkpoint(self, step):
        super().save_checkpoint(step)
        if not self.remote_dir:
            return
        directory = Path(self.default_local_dir) / f"global_step_{step}"
        remote = f"{self.remote_dir}/global_step_{step}"
        if self.is_mp_src_rank_with_outputs:
            upload_path(directory / f"data_{self.dp_rank}.pt", f"{remote}/data_{self.dp_rank}.pt")
        if self.mode == OrchestrationMode.SPMD:
            dist.barrier()
        if self.rank == 0:
            for name in ("lora_train_meta.json",):
                path = directory / name
                if path.exists():
                    upload_path(path, f"{remote}/{name}")
            config = Path(self.default_local_dir) / "effective_config.yaml"
            upload_path(config, f"{self.remote_dir}/effective_config.yaml")
            upload_path(
                Path(self.default_local_dir) / "latest_checkpointed_iteration.txt",
                f"{self.remote_dir}/latest_checkpointed_iteration.txt",
            )
            print("ASR_SFT_CHECKPOINT=" + json.dumps({"step": step, "path": remote}), flush=True)
        if self.mode == OrchestrationMode.SPMD:
            dist.barrier()

    def _determine_resume_path(self):
        explicit = self.resume_from_path
        if self.resume_mode in ("auto", "resume_path") and explicit and explicit.startswith("az://"):
            if "/global_step_" not in explicit or not bf.exists(f"{explicit.rstrip('/')}/data_{self.dp_rank}.pt"):
                raise ValueError("Remote resume path must identify a committed SFT checkpoint with dataloader state")
            return explicit.rstrip("/")
        if self.resume_mode == "resume_path":
            return super()._determine_resume_path()
        if self.resume_mode == "auto" and self.remote_dir and explicit is None:
            marker = f"{self.remote_dir}/latest_checkpointed_iteration.txt"
            if bf.exists(marker):
                with bf.BlobFile(marker, "r") as stream:
                    step = int(stream.read().strip())
                return f"{self.remote_dir}/global_step_{step}"
        return super()._determine_resume_path()

    def _load_dataloader_state(self, checkpoint_path):
        import torch

        source = f"{checkpoint_path.rstrip('/')}/data_{self.dp_rank}.pt"
        if source.startswith("az://"):
            local = copy_to_local(source)
        else:
            local = source
        if not Path(local).is_file():
            raise ValueError(f"ASR SFT resume requires committed dataloader state: {source}")
        state = torch.load(local, map_location="cpu", weights_only=False)
        state = self._normalize_epoch_boundary_state(state)
        self.train_dataloader.load_state_dict(state)
        print(f"ASR SFT restored dataloader state: {source}", flush=True)

    def _normalize_epoch_boundary_state(self, state):
        """Finish a fully consumed iterator before entering the next epoch.

        Final checkpoints are saved after the last batch, before the loader
        receives StopIteration. Without this marker, fit()'s next epoch would
        consume an empty restored iterator and silently lose one whole epoch.
        Mid-epoch snapshots retain their complete original state.
        """
        step = self.resume_global_step
        steps_per_epoch = len(self.train_dataloader)
        if steps_per_epoch <= 0:
            raise ValueError("ASR SFT resume requires a nonempty training dataloader")
        if step <= 0 or step % steps_per_epoch:
            return state
        if "_num_yielded" in state:
            yielded = state["_num_yielded"]
        else:
            snapshot = state.get("_snapshot") or {}
            snapshot_step = snapshot.get("_snapshot_step")
            since_snapshot = state.get("_steps_since_snapshot")
            if not isinstance(snapshot_step, int) or not isinstance(since_snapshot, int):
                raise ValueError("Cannot verify saved dataloader yield count at an SFT epoch boundary")
            yielded = snapshot_step + since_snapshot
        if yielded != steps_per_epoch:
            raise ValueError(
                f"SFT epoch-boundary state mismatch: resume_step={step}, "
                f"saved_yielded={yielded}, steps_per_epoch={steps_per_epoch}"
            )
        normalized = {**state, "_iterator_finished": True}
        print("ASR_SFT_RESUME_BOUNDARY=" + json.dumps({
            "resume_step": step, "first_new_optimizer_step": step + 1,
            "steps_per_epoch": steps_per_epoch, "saved_yielded": yielded,
            "saved_iterator_finished": state.get("_iterator_finished"),
            "normalized_iterator_finished": True,
        }), flush=True)
        return normalized


def build_checkpoint_handler(trainer, *, engine, mode, lora_meta):
    """Use identical checkpoint settings for torchrun and Ray orchestration."""
    cfg = trainer.config.trainer
    return ASRSFTCheckpointHandler(
        engine=engine,
        train_dataloader=trainer.train_dataloader,
        remote_dir=cfg.default_hdfs_dir,
        default_local_dir=cfg.default_local_dir,
        max_ckpt_to_keep=cfg.max_ckpt_to_keep,
        resume_mode=cfg.resume_mode,
        resume_from_path=cfg.resume_from_path,
        mode=mode,
        lora_train_meta=lora_meta,
    )

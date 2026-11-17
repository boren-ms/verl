"""Reuse ASR generation workers across read-only SFT checkpoint evaluations."""

import datetime
import hashlib
import json
import math
import time
from pathlib import Path

import blobfile as bf
from omegaconf import OmegaConf
from recipe.phimm.agentic.trainer import ASRTrainerSync
from recipe.phimm.sft.validation_state import checkpoint_snapshot, checkpoint_steps, source_revision

from verl.utils.skip import SkipManager
from verl.utils.tracking import Tracking, ValidationGenerationsLogger


class SFTCheckpointValidationTrainer(ASRTrainerSync):
    """Load only source model shards; never restore a training optimizer/loader."""

    def _init_dataloader(self):
        super()._init_dataloader()
        expected = self.config.asr.sft_validation.expected_validation_groups
        if len(self.val_dataset) != expected:
            raise ValueError(f"SFT validation coverage changed: {len(self.val_dataset)} != {expected}")

    def _load_checkpoint(self):
        # The first source checkpoint is loaded explicitly in fit, after setup.
        return

    def on_init_end(self):
        # Do not publish the unmodified base weights before loading SFT weights.
        return

    def fit(self, agent_loop_manager):
        self.agent_loop_manager = agent_loop_manager
        SkipManager.init(self.config)
        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        settings = self.config.asr.sft_validation
        steps = checkpoint_steps(settings.steps)
        world_size = self.config.trainer.nnodes * self.config.trainer.n_gpus_per_node
        remote = self.config.asr.validation_remote_dir.rstrip("/")
        config_raw = OmegaConf.to_yaml(self.config, resolve=True)
        identity = {
            "evaluation_config_sha256": hashlib.sha256(config_raw.encode()).hexdigest(),
            "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        revision = source_revision()
        try:
            for step in steps:
                marker = f"{remote}/{step}.complete.json"
                if bf.exists(marker):
                    with bf.BlobFile(marker, "r") as stream:
                        previous = json.load(stream)
                    if any(previous.get(key) != value for key, value in identity.items()):
                        raise ValueError(f"Existing SFT validation provenance differs at step {step}")
                    if previous["checkpoint"] != f"{settings.checkpoint_root.rstrip('/')}/global_step_{step}":
                        raise ValueError(f"Existing SFT validation source differs at step {step}")
                    for artifact in previous["artifacts"]:
                        current = bf.stat(f"{remote}/{artifact['name']}")
                        if current.size != artifact["bytes"] or current.mtime != artifact["mtime"]:
                            raise ValueError(f"Completed SFT validation artifact changed: {artifact['name']}")
                    print(f"SFT_VALIDATION_ALREADY_COMPLETE={step}", flush=True)
                    continue
                snapshot = checkpoint_snapshot(
                    settings.checkpoint_root, step, self.config.actor_rollout_ref.model, world_size
                )
                announced = False
                while snapshot is None:
                    if not settings.wait_for_checkpoints:
                        raise FileNotFoundError(f"SFT checkpoint {step} is not committed")
                    if not announced:
                        print(f"SFT_VALIDATION_WAITING={step}", flush=True)
                        announced = True
                    time.sleep(settings.poll_seconds)
                    snapshot = checkpoint_snapshot(
                        settings.checkpoint_root, step, self.config.actor_rollout_ref.model, world_size
                    )
                self.global_steps = step
                self.checkpoint_manager.sleep_replicas()
                self.actor_rollout_wg.load_checkpoint(local_path=snapshot["checkpoint"], del_local_after_load=False)
                print("SFT_VALIDATION_LOADED=" + json.dumps(snapshot), flush=True)
                self.checkpoint_manager.update_weights(step)
                self.on_validate_begin()
                metrics = self._validate()
                self.on_validate_end()
                expected_keys = {f"val-core/{source}/p_err/mean@1" for source in settings.expected_sources}
                if not expected_keys.issubset(metrics):
                    raise ValueError(f"Incomplete SFT validation metrics at {step}: {expected_keys - metrics.keys()}")
                if any(not math.isfinite(metrics[key]) or metrics[key] < 0 for key in expected_keys):
                    raise ValueError(f"Invalid SFT validation error rate at step {step}")
                # Generation writes are asynchronous; commit completion only after all uploads.
                for future in self._dump_futures:
                    future.result()
                self._dump_futures.clear()
                artifacts = []
                for name in [f"{step}.metrics.json"] + [
                    f"{source}/{step}.jsonl" for source in settings.expected_sources
                ]:
                    info = bf.stat(f"{remote}/{name}")
                    if info.size <= 0:
                        raise ValueError(f"Empty SFT validation artifact: {name}")
                    artifacts.append({"name": name, "bytes": info.size, "mtime": info.mtime})
                self.logger.log(data=metrics, step=step)
                record = {
                    **snapshot,
                    **identity,
                    "metrics": {key: metrics[key] for key in sorted(expected_keys)},
                    "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "revision": revision,
                    "optimizer_updates_performed": 0,
                    "artifacts": artifacts,
                }
                with bf.BlobFile(marker, "w") as stream:
                    json.dump(record, stream, indent=2)
                print("SFT_VALIDATION_COMPLETE=" + json.dumps(record), flush=True)
                self.checkpoint_manager.sleep_replicas()
        finally:
            self._shutdown_dump_executor()

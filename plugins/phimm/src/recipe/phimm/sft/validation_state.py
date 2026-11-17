"""Read-only readiness checks for native SFT generation validation."""

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import blobfile as bf
from omegaconf import OmegaConf


def source_revision():
    """Resolve the owning checkout even when Ray/Hydra changes the working directory."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[6], text=True
    ).strip()


def checkpoint_snapshot(root, step, model_config, world_size):
    """Return committed model provenance, or None while the producer is behind."""
    root = root.rstrip("/")
    marker = f"{root}/latest_checkpointed_iteration.txt"
    if not bf.exists(marker):
        return None
    with bf.BlobFile(marker, "r") as stream:
        if int(stream.read().strip()) < step:
            return None
    with bf.BlobFile(f"{root}/effective_config.yaml", "rb") as stream:
        raw = stream.read()
    producer = OmegaConf.create(raw.decode())
    expected = (
        str(model_config.path).rstrip("/"),
        model_config.lora_rank,
        model_config.lora_alpha,
        world_size,
    )
    actual = (
        str(producer.model.path).rstrip("/"),
        producer.model.lora_rank,
        producer.model.lora_alpha,
        producer.trainer.nnodes * producer.trainer.n_gpus_per_node,
    )
    if actual != expected:
        raise ValueError(f"SFT checkpoint producer mismatch: {actual!r} != {expected!r}")
    checkpoint = f"{root}/global_step_{step}"
    with bf.BlobFile(f"{checkpoint}/lora_train_meta.json", "r") as stream:
        metadata = json.load(stream)
    if (metadata["r"], metadata["lora_alpha"]) != expected[1:3]:
        raise ValueError("SFT checkpoint LoRA metadata mismatch")

    def inspect(rank):
        name = f"model_world_size_{world_size}_rank_{rank}.pt"
        info = bf.stat(f"{checkpoint}/{name}")
        if info.size <= 0:
            raise ValueError(f"Empty committed SFT model shard: {name}")
        return {"name": name, "bytes": info.size, "mtime": info.mtime}

    with ThreadPoolExecutor(max_workers=min(world_size, 8)) as executor:
        shards = list(executor.map(inspect, range(world_size)))
    return {
        "checkpoint": checkpoint,
        "source_optimizer_step": step,
        "base_model": actual[0],
        "lora_rank": actual[1],
        "lora_alpha": actual[2],
        "world_size": world_size,
        "model_shards": shards,
        "producer_config_sha256": hashlib.sha256(raw).hexdigest(),
    }


def checkpoint_steps(values):
    """Reject ambiguous weight-version ordering before allocating workers."""
    steps = list(values)
    if not steps or any(type(step) is not int or step <= 0 for step in steps):
        raise ValueError("SFT validation steps must be positive integers")
    if steps != sorted(set(steps)):
        raise ValueError("SFT validation steps must be strictly increasing and unique")
    return steps

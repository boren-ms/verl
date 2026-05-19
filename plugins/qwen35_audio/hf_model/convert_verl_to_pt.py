#!/usr/bin/env python3
"""Convert a verl FSDP checkpoint to a DeepSpeed-style pytorch checkpoint.

Reads ``model_world_size_{W}_rank_{R}.pt`` shards (saved by verl's
FSDPCheckpointManager) from a checkpoint directory, merges all shards using
verl's FSDPModelMerger logic, and saves the resulting state dict as a single
file in the same layout as ``mp_rank_00_model_states.pt``:

    {"module": OrderedDict({key: bfloat16_tensor, ...})}

Supports ``az://`` source paths (downloads shards locally first via blobfile).

Example::

    python convert_verl_to_pt.py \
        --input az://orngwus2cresco/data/boren/outputs/verl_repeat/\
remax_qwen_bad_repeat_bracket_all_e1/global_step_113 \
        --output /tmp/remax_qwen_bad_repeat_bracket_all_e1_step113/\
mp_rank_00_model_states.pt \
        --local-cache /tmp/verl_ckpt_cache
"""

import argparse
import os
import re
import shutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm

try:
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover
    from torch.distributed._tensor import DTensor


SHARD_RE = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")


# -----------------------------------------------------------------------------
# Locating / downloading shards
# -----------------------------------------------------------------------------


def _is_remote(path: str) -> bool:
    return "://" in path


def _resolve_actor_dir(input_dir: str) -> str:
    """Allow passing either ``.../global_step_N`` or ``.../global_step_N/actor``."""
    actor = input_dir.rstrip("/") + "/actor"
    if _is_remote(input_dir):
        import blobfile as bf
        # peek at one expected file
        if bf.exists(actor + "/model_world_size_8_rank_0.pt") or any(
            SHARD_RE.search(p) for p in bf.listdir(actor)
        ):
            return actor
        return input_dir.rstrip("/")
    actor_p = Path(actor)
    if actor_p.is_dir() and any(SHARD_RE.search(p.name) for p in actor_p.iterdir()):
        return str(actor_p)
    return input_dir.rstrip("/")


def _list_shards(actor_dir: str) -> list[tuple[int, int, str]]:
    """Return list of (world_size, rank, full_path) for model shards."""
    out: list[tuple[int, int, str]] = []
    if _is_remote(actor_dir):
        import blobfile as bf
        entries = list(bf.listdir(actor_dir))
        for name in entries:
            m = SHARD_RE.search(name)
            if m:
                out.append((int(m.group(1)), int(m.group(2)),
                            actor_dir.rstrip("/") + "/" + name))
    else:
        for p in Path(actor_dir).iterdir():
            m = SHARD_RE.search(p.name)
            if m:
                out.append((int(m.group(1)), int(m.group(2)), str(p)))
    out.sort(key=lambda x: x[1])
    return out


def _download_shards(shards: list[tuple[int, int, str]], cache_dir: str,
                     workers: int = 8) -> list[tuple[int, int, str]]:
    """Download remote shards to ``cache_dir``. Returns updated list with local paths."""
    import blobfile as bf

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    new_list: list[tuple[int, int, str]] = []

    def _one(ws: int, rank: int, src: str) -> tuple[int, int, str]:
        dst = os.path.join(cache_dir, os.path.basename(src))
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            # try to validate size against remote
            try:
                if bf.stat(src).size == os.path.getsize(dst):
                    return ws, rank, dst
            except Exception:
                pass
        tmp = dst + ".part"
        with bf.BlobFile(src, "rb") as r, open(tmp, "wb") as w:
            shutil.copyfileobj(r, w, length=8 * 1024 * 1024)
        os.replace(tmp, dst)
        return ws, rank, dst

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one, ws, r, s) for ws, r, s in shards]
        for f in tqdm(futures, desc=f"Downloading {len(shards)} shards", total=len(shards)):
            new_list.append(f.result())

    new_list.sort(key=lambda x: x[1])
    return new_list


# -----------------------------------------------------------------------------
# FSDP merge (adapted from verl/model_merger/fsdp_model_merger.py)
# -----------------------------------------------------------------------------


def _merge_by_placement(tensors: list[torch.Tensor], placement) -> torch.Tensor:
    if placement.is_replicate():
        return tensors[0]
    if placement.is_partial():
        raise NotImplementedError("Partial placement is not supported")
    if placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    raise NotImplementedError(f"Unsupported placement: {placement}")


def _load_and_merge(shard_paths: list[str], workers: int = 8) -> "OrderedDict[str, torch.Tensor]":
    total = len(shard_paths)
    shard_state_dicts: list = [None] * total

    def _load(rank: int):
        shard_state_dicts[rank] = torch.load(
            shard_paths[rank], map_location="cpu", weights_only=False
        )

    with ThreadPoolExecutor(max_workers=min(workers, total)) as ex:
        futures = [ex.submit(_load, r) for r in range(total)]
        for f in tqdm(futures, desc=f"Loading {total} FSDP shards", total=total):
            f.result()

    # Sanity: all shards have same keys
    keys = list(shard_state_dicts[0].keys())
    for r in range(1, total):
        if set(shard_state_dicts[r].keys()) != set(keys):
            raise RuntimeError(f"Shard {r} has different keys than rank 0")

    # Detect mesh from first DTensor (if any)
    pivot = shard_state_dicts[0][sorted(keys)[0]]
    if isinstance(pivot, DTensor):
        mesh_dim_names = pivot.device_mesh.mesh_dim_names
    else:
        mesh_dim_names = ("fsdp",)
    if mesh_dim_names not in (("fsdp",), ("ddp", "fsdp")):
        raise NotImplementedError(f"Unsupported mesh_dim_names {mesh_dim_names}")

    merged: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key in tqdm(keys, desc="Merging tensors"):
        tensors: list[torch.Tensor] = []
        placements = None
        for r in range(total):
            t = shard_state_dicts[r].pop(key)
            if isinstance(t, DTensor):
                tensors.append(t._local_tensor.to(torch.bfloat16))
                ps = tuple(t.placements)
                if mesh_dim_names[0] in ("dp", "ddp"):
                    ps = ps[1:]
                if placements is None:
                    placements = ps
                elif placements != ps:
                    raise RuntimeError(f"Placement mismatch for {key}")
            else:
                tensors.append(t.to(torch.bfloat16))

        if placements is not None:
            if len(placements) != 1:
                raise NotImplementedError("FSDP + TP not supported")
            merged[key] = _merge_by_placement(tensors, placements[0])
        else:
            # plain FSDP flatten -> cat on dim 0
            merged[key] = torch.cat(tensors, dim=0).contiguous() if len(tensors) > 1 else tensors[0]

    return merged


# -----------------------------------------------------------------------------
# Key remapping to match reference DeepSpeed/PEFT layout
# -----------------------------------------------------------------------------


_LORA_RE = re.compile(r"^(?P<prefix>.+)\.lora_(?P<ab>[AB])\.(?P<adapter>[^.]+)\.(?P<tail>weight|bias)$")


def _remap_keys(state_dict: "OrderedDict[str, torch.Tensor]",
                target_adapter_name: str = "default",
                strip_prefix: str = "base_model.model.") -> "OrderedDict[str, torch.Tensor]":
    """Rewrite verl/PEFT keys to match reference checkpoint layout.

    Transformations (applied in order):
      1. Strip ``strip_prefix`` (default ``base_model.model.``) from every key.
      2. For every LoRA module (any key matching ``<prefix>.lora_[AB].<adapter>.<tail>``):
           - rename adapter -> ``target_adapter_name`` (default ``default``).
           - rewrite the base parameter ``<prefix>.<tail>`` to ``<prefix>.base_layer.<tail>``.
    """
    # Pass 1: strip outer LoRA wrapper prefix.
    stripped: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k, v in state_dict.items():
        if strip_prefix and k.startswith(strip_prefix):
            stripped[k[len(strip_prefix):]] = v
        else:
            stripped[k] = v

    # Pass 2: discover which module prefixes are LoRA-wrapped, and rename adapter.
    lora_prefixes: set[str] = set()
    renamed: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k, v in stripped.items():
        m = _LORA_RE.match(k)
        if m:
            lora_prefixes.add(m.group("prefix"))
            new_k = f"{m.group('prefix')}.lora_{m.group('ab')}.{target_adapter_name}.{m.group('tail')}"
            renamed[new_k] = v
        else:
            renamed[k] = v

    # Pass 3: insert .base_layer before .weight/.bias for keys whose module prefix is LoRA-wrapped.
    final: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k, v in renamed.items():
        # only the *direct* base param needs `.base_layer.` inserted; lora_A/lora_B keys are already handled above.
        if "lora_A." in k or "lora_B." in k:
            final[k] = v
            continue
        new_k = k
        for tail in ("weight", "bias"):
            suffix = "." + tail
            if k.endswith(suffix):
                prefix = k[: -len(suffix)]
                if prefix in lora_prefixes:
                    new_k = f"{prefix}.base_layer.{tail}"
                break
        final[new_k] = v

    return final


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="verl checkpoint dir (global_step_N or .../actor). Supports az:// paths.")
    ap.add_argument("--output", required=True,
                    help="Destination .pt file (local). Will be created.")
    ap.add_argument("--local-cache", default=None,
                    help="Local cache dir for downloaded shards (required if --input is remote).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lora-adapter-name", default="default",
                    help="Rename PEFT adapter to this name in the output (default: 'default').")
    ap.add_argument("--strip-prefix", default="base_model.model.",
                    help="Strip this prefix from every key (default: 'base_model.model.').")
    ap.add_argument("--no-remap", action="store_true",
                    help="Skip key remapping (keep raw verl/PEFT key names).")
    args = ap.parse_args()

    actor_dir = _resolve_actor_dir(args.input)
    print(f"[1/4] Actor checkpoint dir: {actor_dir}")

    shards = _list_shards(actor_dir)
    if not shards:
        raise FileNotFoundError(f"No model_world_size_*_rank_*.pt under {actor_dir}")
    world_sizes = {ws for ws, _, _ in shards}
    if len(world_sizes) != 1:
        raise RuntimeError(f"Mixed world sizes found: {world_sizes}")
    world_size = world_sizes.pop()
    ranks = sorted(r for _, r, _ in shards)
    if ranks != list(range(world_size)):
        raise RuntimeError(f"Missing ranks; have {ranks} expected {list(range(world_size))}")
    print(f"       Found {len(shards)} shards, world_size={world_size}")

    if _is_remote(actor_dir):
        if not args.local_cache:
            raise SystemExit("--local-cache is required for remote --input")
        print(f"[2/4] Downloading shards to {args.local_cache}")
        shards = _download_shards(shards, args.local_cache, workers=args.workers)
    else:
        print("[2/4] Skipping download (local input)")

    shard_paths = [p for _, _, p in shards]
    print("[3/4] Merging FSDP shards")
    state_dict = _load_and_merge(shard_paths, workers=args.workers)

    if not args.no_remap:
        before = len(state_dict)
        state_dict = _remap_keys(
            state_dict,
            target_adapter_name=args.lora_adapter_name,
            strip_prefix=args.strip_prefix,
        )
        print(f"       Remapped keys: {before} -> {len(state_dict)} "
              f"(strip_prefix={args.strip_prefix!r}, adapter={args.lora_adapter_name!r})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[4/4] Saving {len(state_dict)} tensors -> {out_path}")
    torch.save({"module": state_dict}, out_path)

    total_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
    print(f"       Done. {len(state_dict)} tensors, {total_bytes / 1e9:.2f} GB (bfloat16)")
    print("       Sample keys:")
    for k in list(state_dict.keys())[:5]:
        t = state_dict[k]
        print(f"         {k}  {tuple(t.shape)}  {t.dtype}")


if __name__ == "__main__":
    main()

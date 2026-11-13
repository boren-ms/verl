#!/usr/bin/env python3
from argparse import ArgumentParser
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
import re

import torch


LORA_KEY_PATTERN = re.compile(
    r"^(?P<prefix>.+)\.lora_(?P<side>A|B)(?:\.(?P<adapter>[^.]+))?\.weight$"
)
SHARD_FILE_PATTERN = re.compile(r"model_world_size_(?P<world>\d+)_rank_(?P<rank>\d+)\.pt$")


def parse_shard_paths(sources: list[Path]) -> list[Path]:
    indexed = []
    for source in sources:
        match = SHARD_FILE_PATTERN.search(source.name)
        if not match:
            raise ValueError(f"Expected model_world_size_<N>_rank_<R>.pt: {source}")
        indexed.append((int(match.group("world")), int(match.group("rank")), source))

    world_sizes = {world_size for world_size, _, _ in indexed}
    if len(world_sizes) != 1:
        raise ValueError(f"Mixed world sizes in model shards: {sorted(world_sizes)}")
    world_size = world_sizes.pop()
    ranks = [rank for _, rank, _ in indexed]
    if sorted(ranks) != list(range(world_size)):
        raise ValueError(f"Expected ranks 0..{world_size - 1}, got {sorted(ranks)}")
    return [source for _, _, source in sorted(indexed)]


def load_state(source: Path) -> Mapping[str, object]:
    state = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise TypeError(f"Expected a state dictionary in {source}, got {type(state).__name__}")
    return state


def merge_dtensors(key: str, values: list[object]) -> torch.Tensor:
    placements = tuple(values[0].placements)
    mesh_dim_names = values[0].device_mesh.mesh_dim_names
    if mesh_dim_names and mesh_dim_names[0] in ("dp", "ddp"):
        placements = placements[1:]
    if len(placements) != 1:
        raise NotImplementedError(f"Unsupported DTensor placements for {key}: {placements}")
    if any(tuple(value.placements)[-1:] != tuple(values[0].placements)[-1:] for value in values):
        raise ValueError(f"Inconsistent DTensor placements for {key}")

    local_tensors = [value._local_tensor.detach().cpu() for value in values]
    placement = placements[0]
    if placement.is_shard():
        return torch.cat(local_tensors, dim=placement.dim).contiguous()
    if placement.is_replicate():
        return local_tensors[0].contiguous()
    raise NotImplementedError(f"Unsupported DTensor placement for {key}: {placement}")


def merge_sharded_tensors(key: str, values: list[object]) -> torch.Tensor:
    size = tuple(values[0].size())
    dtype = values[0].dtype
    output = torch.empty(size, dtype=dtype)
    covered = torch.zeros(size, dtype=torch.bool)

    for value in values:
        if tuple(value.size()) != size or value.dtype != dtype:
            raise ValueError(f"Inconsistent ShardedTensor metadata for {key}")
        for shard in value.local_shards():
            offsets = shard.metadata.shard_offsets
            sizes = shard.metadata.shard_sizes
            slices = tuple(slice(offset, offset + length) for offset, length in zip(offsets, sizes, strict=True))
            tensor = shard.tensor.detach().cpu()
            if covered[slices].any() and not torch.equal(output[slices], tensor):
                raise ValueError(f"Conflicting ShardedTensor values for {key} at {offsets}")
            output[slices] = tensor
            covered[slices] = True

    if not covered.all():
        raise ValueError(f"Incomplete ShardedTensor coverage for {key}")
    return output.contiguous()


def merge_values(key: str, values: list[object]) -> torch.Tensor:
    type_names = {type(value).__name__ for value in values}
    if type_names == {"DTensor"}:
        return merge_dtensors(key, values)
    if type_names == {"ShardedTensor"}:
        return merge_sharded_tensors(key, values)
    if not all(torch.is_tensor(value) for value in values):
        raise TypeError(f"Unsupported LoRA values for {key}: {sorted(type_names)}")
    tensors = [value.detach().cpu() for value in values]
    if len(tensors) == 1:
        return tensors[0].contiguous()
    return torch.cat(tensors, dim=0).contiguous()


def extract_lora_weights(sources: list[Path], output: Path) -> tuple[int, int]:
    shard_paths = parse_shard_paths(sources)
    states = [load_state(source) for source in shard_paths]
    key_sets = [{key for key in state if LORA_KEY_PATTERN.match(str(key))} for state in states]
    if not key_sets[0]:
        keys = sorted(str(key) for key in states[0])[:20]
        raise ValueError(f"No lora_A/lora_B weights found; rank-0 top-level keys: {keys}")
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("LoRA key sets differ across model rank files")

    weights = OrderedDict()
    pairs: dict[tuple[str, str], set[str]] = {}
    for key in sorted(key_sets[0]):
        match = LORA_KEY_PATTERN.match(key)
        assert match is not None
        weights[key] = merge_values(key, [state[key] for state in states])
        pair_id = (match.group("prefix"), match.group("adapter") or "")
        pairs.setdefault(pair_id, set()).add(match.group("side"))

    incomplete = sorted(pair_id for pair_id, sides in pairs.items() if sides != {"A", "B"})
    if incomplete:
        raise ValueError(f"Incomplete LoRA A/B pairs: {incomplete[:10]}")

    torch.save(weights, output)
    return len(weights), len(pairs)


def parse_args():
    parser = ArgumentParser(description="Extract LoRA A/B tensors from all verl FSDP model rank files.")
    parser.add_argument("sources", nargs="+", type=Path, help="model_world_size_<N>_rank_<R>.pt files")
    parser.add_argument("--output", required=True, type=Path, help="Path for lora_weights.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tensor_count, pair_count = extract_lora_weights(args.sources, args.output)
    print(f"{args.output}: {tensor_count} tensors, {pair_count} complete A/B pairs")


if __name__ == "__main__":
    main()

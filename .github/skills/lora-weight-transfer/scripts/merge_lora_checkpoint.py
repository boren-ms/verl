#!/usr/bin/env python3
from argparse import ArgumentParser
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
import hashlib
import os
import re

import torch


LORA_KEY_PATTERN = re.compile(
    r"^(?P<prefix>.+)\.lora_(?P<side>A|B)(?:\.(?P<adapter>[^.]+))?\.weight$"
)
STRIP_PREFIX = "base_model.model."


def load_tensor_dict(path: Path) -> tuple[Mapping[str, torch.Tensor], bool]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    wrapped = isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("module"), Mapping)
    state = checkpoint["module"] if wrapped else checkpoint
    if not isinstance(state, Mapping) or not state or not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError(f"{path} does not contain a tensor state dictionary")
    return state, wrapped


def normalize_prefix(prefix: str) -> str:
    return prefix.removeprefix(STRIP_PREFIX)


def normalize_baseline_keys(baseline: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    normalized = OrderedDict()
    for key, tensor in baseline.items():
        normalized_key = key.replace(".base_layer.", ".")
        if normalized_key in normalized:
            raise ValueError(f"Baseline key collision after removing .base_layer: {normalized_key}")
        normalized[normalized_key] = tensor
    return normalized


def collect_pairs(lora_state: Mapping[str, torch.Tensor]):
    pairs: dict[tuple[str, str], dict[str, tuple[str, torch.Tensor]]] = {}
    for key, tensor in lora_state.items():
        match = LORA_KEY_PATTERN.match(key)
        if not match:
            raise ValueError(f"Non-LoRA key in adapter artifact: {key}")
        pair_id = (normalize_prefix(match.group("prefix")), match.group("adapter") or "")
        pairs.setdefault(pair_id, {})[match.group("side")] = (key, tensor)

    incomplete = sorted(pair_id for pair_id, sides in pairs.items() if set(sides) != {"A", "B"})
    if incomplete:
        raise ValueError(f"Incomplete LoRA A/B pairs: {incomplete[:10]}")
    return pairs


def merge_lora(
    baseline: Mapping[str, torch.Tensor],
    lora_state: Mapping[str, torch.Tensor],
    scaling: float,
) -> tuple[OrderedDict[str, torch.Tensor], list[str]]:
    merged = normalize_baseline_keys(baseline)
    pairs = collect_pairs(lora_state)
    updated_keys = []

    for (prefix, adapter), sides in sorted(pairs.items()):
        _, a = sides["A"]
        _, b = sides["B"]
        base_key = f"{prefix}.weight"
        if base_key not in merged:
            raise ValueError(f"No baseline weight for LoRA prefix {prefix} adapter={adapter}")

        base = merged[base_key]
        expected_shape = (b.shape[0], a.shape[1])
        if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0] or tuple(base.shape) != expected_shape:
            raise ValueError(
                f"Shape mismatch for {prefix}: base={tuple(base.shape)}, A={tuple(a.shape)}, B={tuple(b.shape)}"
            )

        delta = torch.mm(b.to(torch.float32), a.to(torch.float32)).mul_(scaling)
        merged[base_key] = base.to(torch.float32).add_(delta).to(base.dtype)
        updated_keys.append(base_key)

    return merged, updated_keys


def write_checkpoint(state: OrderedDict[str, torch.Tensor], output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    torch.save({"module": state}, temporary)
    os.replace(temporary, output)

    digest = hashlib.md5()
    with output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    checksum = digest.hexdigest()
    output.with_suffix(".md5").write_text(f"{checksum}  {output.name}\n", encoding="ascii")
    return checksum


def parse_args():
    parser = ArgumentParser(description="Merge extracted verl LoRA A/B weights into a full baseline checkpoint.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--lora", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lora-alpha", type=float, default=640.0)
    parser.add_argument("--lora-rank", type=float, default=320.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.resolve() in {args.baseline.resolve(), args.lora.resolve()}:
        raise ValueError("Output must differ from baseline and LoRA input paths")
    if args.lora_rank <= 0:
        raise ValueError("LoRA rank must be positive")

    baseline, wrapped = load_tensor_dict(args.baseline)
    lora_state, lora_wrapped = load_tensor_dict(args.lora)
    if lora_wrapped:
        raise ValueError("LoRA artifact must be a bare tensor dictionary")

    scaling = args.lora_alpha / args.lora_rank
    merged, updated_keys = merge_lora(baseline, lora_state, scaling)
    checksum = write_checkpoint(merged, args.output)
    print(
        f"{args.output}: {len(merged)} tensors, {len(updated_keys)} LoRA pairs merged, "
        f"scaling={scaling:g}, wrapped=True, baseline_wrapped={wrapped}, md5={checksum}"
    )


if __name__ == "__main__":
    main()

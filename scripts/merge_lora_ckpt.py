"""Merge a verl phimm LoRA+FSDP2 actor checkpoint into a full HF model.

Gathers DTensor shards via verl's FSDP merger internals, folds the LoRA
adapter (W += (alpha/r) * B @ A) into the base weights, strips the PEFT
prefixes, instantiates the base Qwen3_5Audio model from config, loads the
merged weights, and saves a self-contained HF model directory.
"""

import argparse
import glob
import os
import shutil

import json

import torch

from transformers import AutoConfig, AutoModelForCausalLM

try:
    from torch.distributed.tensor import DTensor
except ImportError:  # older torch
    from torch.distributed._tensor import DTensor


def gather_state_dict(ckpt_dir: str, cfg_dir: str) -> dict:
    with open(os.path.join(ckpt_dir, "fsdp_config.json")) as f:
        world_size = json.load(f)["world_size"]
    print(f"world_size={world_size}; loading shards...")
    shards = [
        torch.load(
            os.path.join(ckpt_dir, f"model_world_size_{world_size}_rank_{r}.pt"),
            map_location="cpu",
            weights_only=False,
        )
        for r in range(world_size)
    ]
    sd = {}
    for key in shards[0].keys():
        tensors = [s[key] for s in shards]
        t0 = tensors[0]
        if isinstance(t0, DTensor):
            placement = t0.placements[0]
            locals_ = [t._local_tensor for t in tensors]
            if placement.is_shard():
                sd[key] = torch.cat(locals_, dim=placement.dim).contiguous()
            else:  # replicate / partial -> take rank 0
                sd[key] = locals_[0]
        else:
            sd[key] = t0
    return sd


def fold_lora(sd: dict, alpha: float, r: float) -> dict:
    scale = alpha / r
    base_keys = [k for k in sd if k.endswith(".base_layer.weight")]
    folded = 0
    for bk in base_keys:
        prefix = bk[: -len(".base_layer.weight")]
        a_key = b_key = None
        for k in sd:
            if k.startswith(prefix + ".lora_A."):
                a_key = k
            elif k.startswith(prefix + ".lora_B."):
                b_key = k
        if a_key is None or b_key is None:
            print(f"WARN: no lora pair for {prefix}")
            continue
        W = sd[bk].float()
        A = sd[a_key].float()
        B = sd[b_key].float()
        W = W + scale * (B @ A)
        sd[bk] = W.to(torch.bfloat16)
        folded += 1
    print(f"Folded {folded} LoRA modules (scale={scale})")
    # strip prefixes and drop lora tensors
    new = {}
    for k, v in sd.items():
        if ".lora_A." in k or ".lora_B." in k:
            continue
        nk = (
            k.replace("base_model.model.", "")
            .replace(".base_layer.weight", ".weight")
            .replace(".base_layer.bias", ".bias")
        )
        new[nk] = v
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="actor checkpoint dir (has model_world_size_*.pt)")
    ap.add_argument("--cfg", required=True, help="base HF dir with config + custom *.py + tokenizer/processor")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=640)
    ap.add_argument("--rank", type=float, default=320)
    args = ap.parse_args()

    print("Gathering FSDP shards...")
    sd = gather_state_dict(args.ckpt, args.cfg)
    print(f"Merged state dict: {len(sd)} keys")

    print("Folding LoRA...")
    sd = fold_lora(sd, args.alpha, args.rank)
    print(f"Final state dict: {len(sd)} keys")

    print("Building base model from config...")
    import importlib
    import sys

    pkg_dir = os.path.abspath(args.cfg)
    parent = os.path.dirname(pkg_dir)
    pkg = os.path.basename(pkg_dir)
    open(os.path.join(pkg_dir, "__init__.py"), "a").close()
    if parent not in sys.path:
        sys.path.insert(0, parent)
    cfg_mod = importlib.import_module(f"{pkg}.configuration_qwen3_5_audio")
    model_mod = importlib.import_module(f"{pkg}.modeling_qwen3_5_audio")
    with open(os.path.join(pkg_dir, "config.json")) as f:
        cfg_json = json.load(f)
    config = cfg_mod.Qwen3_5AudioConfig.from_dict(cfg_json)
    model = model_mod.Qwen3_5AudioForCausalLM(config)
    model = model.to(torch.bfloat16)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  sample missing:", missing[:10])
    if unexpected:
        print("  sample unexpected:", unexpected[:10])

    os.makedirs(args.out, exist_ok=True)
    print(f"Saving merged model to {args.out} ...")
    model.save_pretrained(args.out, safe_serialization=True)

    # copy custom code + tokenizer/processor sidecars
    for f in glob.glob(os.path.join(args.cfg, "*.py")) + [
        os.path.join(args.cfg, n)
        for n in (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "preprocessor_config.json",
            "processing_qwen3_5_audio.py",
            "chat_template.jinja",
            "special_tokens_map.json",
            "generation_config.json",
        )
    ]:
        if os.path.isfile(f):
            shutil.copy2(f, os.path.join(args.out, os.path.basename(f)))
    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diff / recover checkpoints between a baseline and a fully-merged "new"
checkpoint that share the same architecture but may differ in key naming and
top-level wrapping. PyTorch checkpoints and single-file safetensors are supported.

Layouts
-------
baseline (e.g. amlt fastllm merged ckpt):
    payload      : {"module": OrderedDict[...]}   (DeepSpeed-style)
    key naming   : `model.layers.{i}.<mod>.base_layer.weight`
    contains     : 1274 tensors, no `lora_A` / `lora_B` (already merged)

new (e.g. verl RL global_step_* lora_merged ckpt):
    payload      : {"module": OrderedDict[...]}
    key naming   : `model.layers.{i}.<mod>.weight`            (flat, no .base_layer.)

The two key conventions are reconciled by stripping `.base_layer.` from
baseline keys when comparing.

Subcommands (shared args: --new, --baseline, --delta, [--tol])
-----------
    diff    reads  --new + --baseline   ->  writes --delta
    apply   reads  --baseline + --delta ->  writes --new

`delta.pt` layout (deliberately mirrors baseline):
    plain `collections.OrderedDict`
    key naming = baseline's (`.base_layer.weight` preserved)
    values     = new[k]  (NOT the numerical difference) for tensors that changed
    only the changed tensors are stored (40 in practice)

`apply` recovers a checkpoint identical to NEW.pt (same flat key naming and
{"module": ...} wrapping) so it can be dropped into the new ckpt directory.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import OrderedDict

import torch


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _unwrap(sd):
    """Return the inner state-dict if `sd` is wrapped under a standard key."""
    for k in ("module", "state_dict", "model", "model_state_dict"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sub = sd[k]
            if any(torch.is_tensor(v) for v in sub.values()):
                return sub
    return sd


def _normalize_key(k: str) -> str:
    """Strip the LoRA `.base_layer.` infix so baseline keys align with new keys."""
    return k.replace(".base_layer.", ".") if ".base_layer." in k else k


def _load(path):
    print(f"[load] {path}", flush=True)
    t0 = time.time()
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        sd = load_file(path, device="cpu")
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    print(f"[load] done in {time.time() - t0:.1f}s", flush=True)
    return sd


def _save(state_dict, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".safetensors"):
        from safetensors.torch import save_file

        save_file(dict(state_dict), path)
    else:
        torch.save(state_dict, path)


def _tensor_items(sd):
    return [(k, v) for k, v in sd.items() if torch.is_tensor(v)]


# --------------------------------------------------------------------------- #
# diff                                                                        #
# --------------------------------------------------------------------------- #
def cmd_diff(args):
    out_path = args.delta
    new_raw = _load(args.new)
    base_raw = _load(args.baseline)

    new_sd = _unwrap(new_raw)
    base_sd = _unwrap(base_raw)

    # build normalized -> (orig_key, tensor) maps
    new_norm = {_normalize_key(k): (k, v) for k, v in _tensor_items(new_sd)}
    base_norm = {_normalize_key(k): (k, v) for k, v in _tensor_items(base_sd)}

    shared = sorted(set(new_norm) & set(base_norm))
    only_new = sorted(set(new_norm) - set(base_norm))
    only_base = sorted(set(base_norm) - set(new_norm))

    delta: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    n_changed = n_unchanged = n_shape_mm = 0
    total_l1 = 0.0
    total_changed_elem = 0
    stats = []  # for sorted top-20 print

    for nk in shared:
        new_orig, new_t = new_norm[nk]
        base_orig, base_t = base_norm[nk]
        if new_t.shape != base_t.shape:
            n_shape_mm += 1
            print(f"[shape-mismatch] {nk}: {tuple(new_t.shape)} vs {tuple(base_t.shape)}")
            continue
        d = new_t.detach().to(torch.float32) - base_t.detach().to(torch.float32)
        max_abs = d.abs().max().item() if d.numel() else 0.0
        if max_abs <= args.tol:
            n_unchanged += 1
            continue
        # store NEW tensor under BASELINE key naming
        delta[base_orig] = new_t.detach().contiguous()
        l1 = d.abs().sum().item()
        n_changed += 1
        total_l1 += l1
        total_changed_elem += new_t.numel()
        stats.append((max_abs, base_orig, tuple(new_t.shape)))

    print(f"[save] {out_path} ({len(delta)} tensors)", flush=True)
    _save(delta, out_path)
    sz_gib = os.path.getsize(out_path) / (1024 ** 3)

    bar = "=" * 78
    print(bar)
    print(f"shared keys              : {len(shared)}")
    print(f"changed (>tol)           : {n_changed}")
    print(f"unchanged                : {n_unchanged}")
    print(f"shape mismatch (skipped) : {n_shape_mm}")
    print(f"only in new              : {len(only_new)}")
    print(f"only in baseline         : {len(only_base)}")
    print(f"total |delta|_1          : {total_l1:.4e}")
    print(f"changed elements         : {total_changed_elem:,}")
    print(f"output size              : {sz_gib:.2f} GiB")
    print(bar)
    print("top-20 changed tensors by max_abs:")
    for max_abs, k, shape in sorted(stats, reverse=True)[:20]:
        print(f"  {max_abs:.4e}  shape={shape}  {k}")


# --------------------------------------------------------------------------- #
# apply                                                                       #
# --------------------------------------------------------------------------- #
def cmd_apply(args):
    out_path = args.new
    base_raw = _load(args.baseline)
    delta = _load(args.delta)
    assert isinstance(delta, dict), f"unexpected delta payload type: {type(delta)}"

    wrapped = isinstance(base_raw, dict) and "module" in base_raw and isinstance(base_raw["module"], dict)
    base_sd = base_raw["module"] if wrapped else base_raw

    base_norm = {_normalize_key(k): (k, v) for k, v in base_sd.items()}
    missing = [k for k in delta if _normalize_key(k) not in base_norm]
    shape_mm = [
        (k, tuple(base_norm[_normalize_key(k)][1].shape), tuple(v.shape))
        for k, v in delta.items()
        if _normalize_key(k) in base_norm and base_norm[_normalize_key(k)][1].shape != v.shape
    ]
    assert not missing, f"delta keys missing in baseline: {missing[:5]}"
    assert not shape_mm, f"shape mismatch: {shape_mm[:5]}"

    n_overwrite = len(delta)
    for delta_key, value in delta.items():
        base_key, _ = base_norm[_normalize_key(delta_key)]
        base_sd[base_key] = value

    # strip `.base_layer.` to match new.pt's flat layout
    flat: "OrderedDict[str, torch.Tensor]" = OrderedDict(
        (_normalize_key(k), v) for k, v in base_sd.items()
    )

    print(f"[save] {out_path} ({len(flat)} tensors, {n_overwrite} overwritten)", flush=True)
    output = flat if out_path.endswith(".safetensors") else {"module": flat}
    _save(output, out_path)
    sz_gib = os.path.getsize(out_path) / (1024 ** 3)
    print(f"[save] done. size={sz_gib:.2f} GiB")

    keys = list(flat.keys())
    print("first 4 keys:", keys[:4])
    print("last 4 keys :", keys[-4:])


# --------------------------------------------------------------------------- #
# cli                                                                         #
# --------------------------------------------------------------------------- #
def _add_shared_args(sp):
    sp.add_argument("--new",      required=True, help="fully-merged new ckpt (input for diff, output for apply)")
    sp.add_argument("--baseline", required=True, help="LoRA-merged baseline ckpt (input for both)")
    sp.add_argument("--delta",    required=True, help="delta ckpt (output for diff, input for apply)")
    sp.add_argument("--tol",      type=float, default=0.0,
                    help="diff: max_abs threshold; tensors with max|Δ| <= tol are treated as unchanged")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diff", help="compute new - baseline, write delta")
    _add_shared_args(d)
    d.set_defaults(func=cmd_diff)

    a = sub.add_parser("apply", help="baseline + delta -> recovered new")
    _add_shared_args(a)
    a.set_defaults(func=cmd_apply)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

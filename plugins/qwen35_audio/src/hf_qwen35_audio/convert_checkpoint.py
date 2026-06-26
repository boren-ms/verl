#!/usr/bin/env python3
"""Convert a LoRA-merged training checkpoint to a self-contained HF model.

Reads lora_merged_model_states.pt + config.json from a checkpoint directory,
remaps keys, saves as safetensors, copies model .py files / tokenizer / config,
and produces a ready-to-use HF model directory.

Usage (local paths):
    python convert_checkpoint.py \
        --input-dir /path/to/checkpoint/20000 \
        --output-dir /path/to/output/qwen_hf \
        --tokenizer-source /path/to/tokenizer_dir

Usage (blob, via wrapper):
    python convert_checkpoint.py \
        --input-dir /tmp/ckpt \
        --output-dir /tmp/qwen_hf \
        --tokenizer-source /tmp/tokenizer
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_PY_FILES = [
    "configuration_qwen3_5_audio.py",
    "modeling_qwen3_5_audio.py",
    "audio_embedding.py",
    "cascade_encoder.py",
    "processing_qwen3_5_audio.py",
    "verify_hf_model.py",
]

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
]

PREPROCESSOR_CONFIG = {
    "feature_extractor_type": "Qwen3_5AudioFeatureExtractor",
    "processor_class": "Qwen3_5AudioProcessor",
    "feature_size": 80,
    "sampling_rate": 16000,
    "compression_rate": 8,
    "padding_value": 0.0,
    "auto_map": {
        "AutoProcessor": "processing_qwen3_5_audio.Qwen3_5AudioProcessor",
        "AutoFeatureExtractor": "processing_qwen3_5_audio.Qwen3_5AudioFeatureExtractor",
    },
}


def remap_key(name: str) -> str:
    """Remap training checkpoint key to self-contained HF model key.

    Training checkpoint format:
        model.embed_tokens.*         -> model.embed_tokens.*       (same)
        model.embed_tokens_extend.*  -> model.embed_tokens_extend.* (same)
        model.layers.*               -> model.layers.*              (same)
        model.norm.*                 -> model.norm.*                (same)
        lm_head.*                    -> lm_head.*                   (same)

    Only need to strip LoRA base_layer prefix.
    """
    return name.replace("base_layer.", "")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LoRA-merged checkpoint to self-contained HF model"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing lora_merged_model_states.pt and config.json",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for the HF model",
    )
    parser.add_argument(
        "--tokenizer-source", type=str, default=None,
        help="Directory containing tokenizer files. If not provided, "
             "looks for them in input-dir.",
    )
    parser.add_argument(
        "--checkpoint-file", type=str, default="lora_merged_model_states.pt",
        help="Checkpoint filename (default: lora_merged_model_states.pt)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load and remap weights ---
    ckpt_path = input_dir / args.checkpoint_file
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    print(f"  {len(state_dict)} tensors loaded")

    remapped = {}
    skipped = 0
    for k, v in state_dict.items():
        if ".lora_A." in k or ".lora_B." in k:
            skipped += 1
            continue
        if k.startswith("mtp."):
            skipped += 1
            continue
        new_k = remap_key(k)
        if v.dtype == torch.float32:
            v = v.to(torch.bfloat16)
        remapped[new_k] = v

    if skipped:
        print(f"  Skipped {skipped} LoRA/MTP tensors")
    print(f"  {len(remapped)} tensors after remapping")

    # Show a few key samples
    sample_keys = sorted(remapped.keys())[:5]
    for k in sample_keys:
        print(f"    {k}  {tuple(remapped[k].shape)}  {remapped[k].dtype}")

    # Save as safetensors
    from safetensors.torch import save_file
    out_weights = output_dir / "model.safetensors"
    print(f"Saving weights: {out_weights}")
    save_file(remapped, str(out_weights))
    del state_dict, remapped
    print("  Done")

    # --- 2. Prepare config.json ---
    config_path = input_dir / "config.json"
    print(f"Reading config: {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    config["model_type"] = "qwen3_5_audio"
    config["architectures"] = ["Qwen3_5AudioForCausalLM"]
    config["auto_map"] = {
        "AutoConfig": "configuration_qwen3_5_audio.Qwen3_5AudioConfig",
        "AutoModelForCausalLM": "modeling_qwen3_5_audio.Qwen3_5AudioForCausalLM",
        "AutoProcessor": "processing_qwen3_5_audio.Qwen3_5AudioProcessor",
    }

    out_config = output_dir / "config.json"
    with open(out_config, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved {out_config}")

    # --- 3. Copy model Python files ---
    print("Copying model files:")
    for fname in MODEL_PY_FILES:
        src = SCRIPT_DIR / fname
        if not src.exists():
            print(f"  WARNING: {src} not found, skipping")
            continue
        shutil.copy2(src, output_dir / fname)
        print(f"  {fname}")

    # --- 4. Write preprocessor_config.json ---
    preproc = dict(PREPROCESSOR_CONFIG)
    if isinstance(config.get("embd_layer"), dict):
        preproc["compression_rate"] = config["embd_layer"].get("compression_rate", 8)
    preproc_path = output_dir / "preprocessor_config.json"
    with open(preproc_path, "w") as f:
        json.dump(preproc, f, indent=2)
    print(f"  preprocessor_config.json")

    # --- 5. Copy tokenizer files ---
    tok_source = Path(args.tokenizer_source) if args.tokenizer_source else input_dir
    print(f"Copying tokenizer from: {tok_source}")
    copied_tok = 0
    for fname in TOKENIZER_FILES:
        src = tok_source / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            copied_tok += 1
            print(f"  {fname}")
    if copied_tok == 0:
        print("  WARNING: No tokenizer files found!")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Self-contained HF model ready at: {output_dir}")
    print(f"Files:")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:40s} {size_mb:8.1f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

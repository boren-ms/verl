#!/usr/bin/env python3
"""Verify the self-contained HF Qwen3.5 Audio model using AutoProcessor.

Follows the Phi4-MultiModal sample_inference pattern:
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    inputs = processor(text=prompt, audios=[(wav, sr)], return_tensors="pt")
    outputs = model.generate(**inputs)
    text = processor.batch_decode(outputs, skip_special_tokens=True)

Both ``MODEL_DIR`` and ``AUDIO`` may be local paths or remote ``az://`` paths;
remote paths are downloaded to a local cache via ``blobfile`` before use.

Usage:
    MODEL_DIR=~/data/qwen35-audio-hf AUDIO=/tmp/1320-122617-0005.flac python verify_hf_model.py
    MODEL_DIR=az://bucket/path/to/model AUDIO=az://bucket/path/clip.flac python verify_hf_model.py
"""

import os
import time

import soundfile as sf
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

# Register the self-contained Qwen3.5-Audio classes with the Transformers Auto*
# registries so the model below loads from this installed package rather than the
# checkpoint's bundled ``trust_remote_code`` ``*.py`` copies.
try:
    from hf_qwen35_audio import register_hf_audio_model
except ImportError:  # running this file as a stand-alone script (not installed)
    from __init__ import register_hf_audio_model
register_hf_audio_model()

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "az://orngwus2cresco/data/speech/projects/phi-fastllm-2605/amlt-results/"
    "fast-llm-2605-qwen3-5-9b-s2-st-example-r2/90000/qwen_hf",
)
AUDIO = os.environ.get(
    "AUDIO",
    "az://orngwus2cresco/data/boren/data/LibriSpeech/train-clean-360/115/122944/"
    "115-122944-0000.flac",
)
INSTRUCTION = "Transcribe the audio clip into text."
MAX_NEW_TOKENS = 256

LOCAL_CACHE_ROOT = os.environ.get(
    "QWEN35_AUDIO_CACHE", os.path.expanduser("~/data/qwen35_audio_test")
)


def stage_file(remote: str) -> str:
    """Return a local path for ``remote``; download it if it's an ``az://`` path."""
    if not remote.startswith("az://"):
        return remote
    import blobfile as bf

    rel = remote[len("az://"):]
    local = os.path.join(LOCAL_CACHE_ROOT, "audio", rel.replace("/", "_"))
    if os.path.exists(local):
        return local
    os.makedirs(os.path.dirname(local), exist_ok=True)
    print(f"staging {remote} -> {local}")
    bf.copy(remote, local, overwrite=True)
    return local


def stage_model(model: str) -> str:
    """Return a local model dir; download it if ``model`` is an ``az://`` path."""
    if not model.startswith("az://"):
        return model
    import blobfile as bf

    rel = model[len("az://"):].rstrip("/")
    local = os.path.join(LOCAL_CACHE_ROOT, "models", rel.replace("/", "_"))
    remote = model.rstrip("/") + "/"
    if os.path.isdir(local) and os.listdir(local):
        print(f"model already staged at {local}")
        return local
    os.makedirs(local, exist_ok=True)
    print(f"staging model {remote} -> {local}")
    for entry in bf.scandir(remote):
        if entry.is_dir:
            continue
        dst = os.path.join(local, entry.name)
        if os.path.exists(dst):
            continue
        print(f"  {entry.name}")
        bf.copy(remote + entry.name, dst, overwrite=True)
    return local


def main():
    print(f"Model:  {MODEL_DIR}")
    print(f"Audio:  {AUDIO}")

    model_dir = stage_model(MODEL_DIR)
    audio_path = stage_file(AUDIO)

    device = torch.device("cuda:0")
    a100 = torch.cuda.get_device_properties(0).major >= 8
    dtype = torch.bfloat16 if a100 else torch.float16
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa" if a100 else "eager"

    # ---- Load model & processor via HF auto API ----
    # ``trust_remote_code=False``: resolve to the registered ``hf_model`` package
    # classes instead of the checkpoint's bundled ``*.py`` copies.
    t0 = time.time()
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=config, trust_remote_code=False,
        torch_dtype=dtype, attn_implementation=attn_impl,
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=False)
    print(f"Loaded in {time.time() - t0:.1f}s  ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")

    # ---- Load audio ----
    wav, sr = sf.read(audio_path)
    print(f"Audio: {len(wav)/sr:.2f}s @ {sr}Hz")

    # ---- Build prompt & process inputs ----
    prompt = f"<|im_start|>user\n<audio>\n{INSTRUCTION}<|im_end|>\n<|im_start|>assistant\n"

    inputs = processor(text=prompt, audios=[(wav, sr)], return_tensors="pt").to(device)

    # ---- Generate (audio features flow through **inputs directly) ----
    t0 = time.time()
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    # Remove input tokens
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    print(f"\nDecoded in {time.time() - t0:.2f}s")
    print(f"Transcription: {response.strip()}")
    print("\n[OK] Verification passed!")


if __name__ == "__main__":
    main()

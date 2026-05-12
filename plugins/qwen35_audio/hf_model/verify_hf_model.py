#!/usr/bin/env python3
"""Verify the self-contained HF Qwen3.5 Audio model using AutoProcessor.

Follows the Phi4-MultiModal sample_inference pattern:
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    inputs = processor(text=prompt, audios=[(wav, sr)], return_tensors="pt")
    outputs = model.generate(**inputs)
    text = processor.batch_decode(outputs, skip_special_tokens=True)

Usage:
    MODEL_DIR=~/data/qwen35-audio-hf AUDIO=/tmp/1320-122617-0005.flac python verify_hf_model.py
"""

import os
import time

import soundfile as sf
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.expanduser("~/data/qwen35-audio-hf"))
AUDIO = os.environ.get("AUDIO", "/tmp/1320-122617-0005.flac")
INSTRUCTION = "Transcribe the audio clip into text."
MAX_NEW_TOKENS = 256


def main():
    print(f"Model:  {MODEL_DIR}")
    print(f"Audio:  {AUDIO}")

    device = torch.device("cuda:0")
    a100 = torch.cuda.get_device_properties(0).major >= 8
    dtype = torch.bfloat16 if a100 else torch.float16
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa" if a100 else "eager"

    # ---- Load model & processor via HF auto API ----
    t0 = time.time()
    config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, config=config, trust_remote_code=True,
        torch_dtype=dtype, attn_implementation=attn_impl,
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    print(f"Loaded in {time.time() - t0:.1f}s  ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")

    # ---- Load audio ----
    wav, sr = sf.read(AUDIO)
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

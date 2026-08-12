#!/usr/bin/env python3
"""Verify the self-contained HF Qwen3.5 Audio model using AutoProcessor.

Follows the Phi4-MultiModal sample_inference pattern:
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    inputs = processor(text=prompt, audios=[(wav, sr)], return_tensors="pt")
    outputs = model.generate(**inputs)
    text = processor.batch_decode(outputs, skip_special_tokens=True)

Usage:
    python plugins/qwen35_audio/scripts/run_qwen35_audio_hf.py
"""

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

REMOTE_MODEL_PATH = (
    "az://orngwus2cresco/data/speech/projects/phi-fastllm-2607/amlt-results/"
    "fast-llm-2607-qwen3-5-9b-s2-data-v3.4-sr-afteraudio-lexical-fix/50000/qwen_hf/"
)
REMOTE_AUDIO_PATH = (
    "az://orngwus2cresco/data/boren/data/LibriSpeech/train-clean-360/115/"
    "122944/115-122944-0036.flac"
)
LOCAL_CACHE_ROOT = "/root/data/qwen35_audio_test"
DEFAULT_MODEL_PATH = REMOTE_MODEL_PATH
DEFAULT_AUDIO_PATH = REMOTE_AUDIO_PATH
INSTRUCTION = "Transcribe the audio clip into text."
MAX_NEW_TOKENS = 256
TARGET_SAMPLE_RATE = 16_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Verified remote reproduction paths for a new Brix node: "
            f"model={REMOTE_MODEL_PATH}, audio={REMOTE_AUDIO_PATH}"
        ),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("QWEN35_AUDIO_MODEL", os.getenv("MODEL_DIR", DEFAULT_MODEL_PATH)),
        help="Path to the converted Qwen3.5-Audio HuggingFace checkpoint.",
    )
    parser.add_argument(
        "--audio",
        default=os.getenv("QWEN35_AUDIO_SAMPLE", os.getenv("AUDIO", DEFAULT_AUDIO_PATH)),
        help="Path to an audio file readable by soundfile.",
    )
    parser.add_argument(
        "--local-cache-root",
        default=os.getenv("QWEN35_AUDIO_CACHE_ROOT", LOCAL_CACHE_ROOT),
        help="Local directory used to stage az:// model and audio inputs.",
    )
    parser.add_argument(
        "--skip-stage",
        action="store_true",
        help="Do not stage az:// inputs; pass paths through directly.",
    )
    parser.add_argument("--instruction", default=INSTRUCTION)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def is_az_path(path: str) -> bool:
    return path.startswith("az://")


def run_bbb(*args: str) -> None:
    if shutil.which("bbb") is None:
        raise RuntimeError("bbb is required to stage az:// paths on a fresh node")
    print("bbb " + " ".join(args))
    subprocess.run(["bbb", *args], check=True)


def stage_input(source: str, destination: Path, *, is_dir: bool) -> str:
    if not is_az_path(source):
        return source

    if destination.exists():
        print(f"using_staged_path={destination}")
        return str(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if is_dir:
        destination.mkdir(parents=True, exist_ok=True)
        run_bbb("sync", source, str(destination))
    else:
        run_bbb("cp", source, str(destination))
    return str(destination)


def stage_inputs(args: argparse.Namespace) -> tuple[str, str]:
    if args.skip_stage:
        return args.model, args.audio

    cache_root = Path(args.local_cache_root)
    model_path = stage_input(
        args.model,
        cache_root / "qwen35-audio-hf",
        is_dir=True,
    )
    audio_path = stage_input(
        args.audio,
        cache_root / "audio" / Path(args.audio.rstrip("/")).name,
        is_dir=False,
    )
    return model_path, audio_path


def load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(audio_path)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = F.resample(
            torch.from_numpy(waveform),
            sample_rate,
            TARGET_SAMPLE_RATE,
        ).numpy()
        sample_rate = TARGET_SAMPLE_RATE
    return waveform.astype(np.float32), sample_rate


def main():
    args = parse_args()
    model_path, audio_path = stage_inputs(args)

    print(f"model_source={args.model}")
    print(f"audio_source={args.audio}")
    print(f"model_path={model_path}")
    print(f"audio_path={audio_path}")

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
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype, attn_implementation=attn_impl,
    ).to(device).eval()
    print(f"Loaded in {time.time() - t0:.1f}s  ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")

    # ---- Load audio ----
    wav, sr = load_audio(audio_path)
    print(f"Audio: {len(wav)/sr:.2f}s @ {sr}Hz")

    # ---- Build prompt & process inputs ----
    prompt = f"<|im_start|>user\n<audio>\n{args.instruction}<|im_end|>\n<|im_start|>assistant\n"

    inputs = processor(text=prompt, audios=[(wav, sr)], return_tensors="pt").to(device)

    # ---- Generate (audio features flow through **inputs directly) ----
    t0 = time.time()
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=False,
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

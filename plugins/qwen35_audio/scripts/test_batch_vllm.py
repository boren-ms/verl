#!/usr/bin/env python3
"""Batch inference test: multiple LibriSpeech audio samples."""
import os, sys, time
import numpy as np
import soundfile as sf
import shutil, subprocess
from pathlib import Path

os.environ.setdefault("VLLM_PLUGINS", "qwen35_audio")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("QWEN35_AUDIO_DISABLE_CUDNN", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vllm_qwen35_audio.plugin import register
register()

from vllm import LLM, SamplingParams

MODEL = "/root/data/qwen35_audio_test/models/qwen_hf-3368974083"
AUDIO_AZ_ROOT = "az://orngwus2cresco/data/boren/data/LibriSpeech/train-clean-360/115/122944"
AUDIO_LOCAL_ROOT = "/root/data/qwen35_audio_test/audio"
AUDIO_FILES = [f"115-122944-{i:04d}.flac" for i in range(11)]  # 0000..0010
PROMPT = (
    "<|im_start|>user\n<audio>\n"
    "Detect the language and transcribe the audio into text.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def stage_audio(name: str) -> str:
    local = os.path.join(AUDIO_LOCAL_ROOT, name)
    if os.path.exists(local):
        return local
    os.makedirs(AUDIO_LOCAL_ROOT, exist_ok=True)
    print(f"staging {name}")
    subprocess.run(["bbb", "cp", f"{AUDIO_AZ_ROOT}/{name}", local], check=True)
    return local


def main():
    # Stage and load all audio files
    waveforms = []
    for name in AUDIO_FILES:
        fp = stage_audio(name)
        wav, sr = sf.read(fp)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        waveforms.append((name, wav, sr))
        print(f"  {name}: {len(wav)/sr:.2f}s @ {sr}Hz")

    print(f"\nbatch_size={len(waveforms)}")

    llm = LLM(
        model=MODEL, trust_remote_code=True,
        max_model_len=4096, max_num_seqs=len(waveforms) + 1, dtype="bfloat16",
        tensor_parallel_size=8, limit_mm_per_prompt={"audio": 1},
        gpu_memory_utilization=0.15,
    )
    print("model loaded")

    params = SamplingParams(temperature=0.0, max_tokens=128, stop_token_ids=[248044, 248046])

    requests = [
        {"prompt": PROMPT, "multi_modal_data": {"audio": [(wav, sr)]}}
        for _, wav, sr in waveforms
    ]

    t0 = time.time()
    outputs = llm.generate(requests, sampling_params=params)
    elapsed = time.time() - t0
    print(f"generate_seconds={elapsed:.1f}  batch_size={len(requests)}")

    for i, out in enumerate(outputs):
        name = waveforms[i][0]
        print(f"\n--- {name} ---")
        print(out.outputs[0].text.strip())

    print("\nBATCH_DONE")


if __name__ == "__main__":
    main()

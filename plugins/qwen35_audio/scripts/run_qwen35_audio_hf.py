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
import time

import torch

try:
    from .qwen35_audio_utils import add_input_arguments, load_audio, stage_inputs
except ImportError:
    from qwen35_audio_utils import add_input_arguments, load_audio, stage_inputs

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Verified remote reproduction paths for a new Brix node: "
            f"model={REMOTE_MODEL_PATH}, audio={REMOTE_AUDIO_PATH}"
        ),
    )
    add_input_arguments(
        parser,
        default_model_path=DEFAULT_MODEL_PATH,
        default_audio_path=DEFAULT_AUDIO_PATH,
        default_cache_root=LOCAL_CACHE_ROOT,
        model_env_names=("QWEN35_AUDIO_MODEL", "MODEL_DIR"),
        audio_env_names=("QWEN35_AUDIO_SAMPLE", "AUDIO"),
    )
    parser.add_argument("--instruction", default=INSTRUCTION)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path, audio_paths = stage_inputs(args)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    print(f"model_source={args.model}")
    print(f"model_path={model_path}")
    print(f"audio_count={len(audio_paths)}")

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

    # ---- Build prompt ----
    prompt = f"<|im_start|>user\n{args.instruction}<audio>\n<|im_end|>\n<|im_start|>assistant\n"

    for audio_source, audio_path in audio_paths:
        wav, sr = load_audio(audio_path)
        print("AUDIO_RESULT_START")
        print(f"audio_source={audio_source}")
        print(f"audio_path={audio_path}")
        print(f"Audio: {len(wav)/sr:.2f}s @ {sr}Hz")

        inputs = processor(text=prompt, audios=[(wav, sr)], return_tensors="pt").to(device)

        # Audio features flow through **inputs directly.
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

        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        print(f"Decoded in {time.time() - t0:.2f}s")
        print(f"Transcription: {response}")
        print("TRANSCRIPT_START")
        print(response)
        print("TRANSCRIPT_END")
        print("AUDIO_RESULT_END")

    print(f"BATCH_DONE count={len(audio_paths)}")
    print("[OK] Verification passed!")


if __name__ == "__main__":
    main()

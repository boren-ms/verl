#!/usr/bin/env python3
"""Decode audio with openai/whisper-small through vLLM."""

import argparse
import os
import time

from qwen35_audio_utils import add_input_arguments, load_audio, stage_inputs

MODEL_ID = "openai/whisper-small"
REMOTE_MODEL_PATH = "az://orngwus2cresco/data/boren/data/verl/models/openai-whisper-small/"
DEFAULT_AUDIO_PATH = (
    "az://orngwus2cresco/data/boren/data/LibriSpeech/train-clean-360/115/"
    "122944/115-122944-0036.flac"
)
DEFAULT_MODEL_PATH = REMOTE_MODEL_PATH
LOCAL_CACHE_ROOT = "/root/data/whisper_small_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=f"Original Hugging Face model: {MODEL_ID}",
    )
    add_input_arguments(
        parser,
        default_model_path=DEFAULT_MODEL_PATH,
        default_audio_path=DEFAULT_AUDIO_PATH,
        default_cache_root=LOCAL_CACHE_ROOT,
        model_env_names=("WHISPER_MODEL",),
        audio_env_names=("WHISPER_AUDIO_SAMPLE", "QWEN35_AUDIO_SAMPLE"),
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Whisper language token, such as en or zh; use auto to detect the language.",
    )
    parser.add_argument(
        "--task",
        choices=("transcribe", "translate"),
        default="transcribe",
    )
    parser.add_argument(
        "--timestamps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow Whisper to emit timestamp tokens.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=448)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def build_prompt(language: str, task: str, timestamps: bool) -> str:
    prompt = "<|startoftranscript|>"
    if language != "auto":
        prompt += f"<|{language}|>"
    prompt += f"<|{task}|>"
    if not timestamps:
        prompt += "<|notimestamps|>"
    return prompt


def main() -> None:
    args = parse_args()
    model_path, audio_paths = stage_inputs(args)

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams

    print(f"model_source={args.model}")
    print(f"model_path={model_path}")
    print(f"audio_count={len(audio_paths)}")

    loaded_audio = []
    for audio_source, audio_path in audio_paths:
        waveform, sample_rate = load_audio(audio_path)
        loaded_audio.append((audio_source, audio_path, waveform, sample_rate))
        print(
            f"audio_source={audio_source} audio_path={audio_path} "
            f"audio_seconds={len(waveform) / sample_rate:.2f} sample_rate={sample_rate}"
        )

    prompt = build_prompt(args.language, args.task, args.timestamps)
    print(f"prompt={prompt!r}")

    start_time = time.time()
    llm = LLM(
        model=model_path,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={"audio": 1},
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(f"load_seconds={time.time() - start_time:.1f}")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    start_time = time.time()
    outputs = llm.generate(
        [
            {
                "prompt": prompt,
                "multi_modal_data": {"audio": [(waveform, sample_rate)]},
            }
            for _, _, waveform, sample_rate in loaded_audio
        ],
        sampling_params=sampling_params,
    )
    print(f"generate_seconds={time.time() - start_time:.1f}")

    for (audio_source, audio_path, _, _), output in zip(loaded_audio, outputs, strict=True):
        print("AUDIO_RESULT_START")
        print(f"audio_source={audio_source}")
        print(f"audio_path={audio_path}")
        print("TRANSCRIPT_START")
        print(output.outputs[0].text.strip())
        print("TRANSCRIPT_END")
        print("AUDIO_RESULT_END")
    print(f"BATCH_DONE count={len(outputs)}")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Smoke test for Qwen3.5-Audio through the vLLM plugin."""

import argparse
import hashlib
import importlib.metadata as metadata
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

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
DEFAULT_INSTRUCTION = "Detect the language and transcribe the audio clip into text.<audio>"
DEFAULT_STOP_TOKEN_IDS = [248044, 248046]
MODEL_ARCHITECTURE = "Qwen3_5AudioForCausalLM"
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
        default=os.getenv("QWEN35_AUDIO_MODEL", DEFAULT_MODEL_PATH),
        help="Path to the converted Qwen3.5-Audio HuggingFace checkpoint.",
    )
    parser.add_argument(
        "--audio",
        default=os.getenv("QWEN35_AUDIO_SAMPLE", DEFAULT_AUDIO_PATH),
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
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--assistant-prefix", default="")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--stop-token-id",
        action="append",
        type=int,
        default=None,
        help="Stop token id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--keep-cudnn-enabled",
        action="store_true",
        help="Do not set QWEN35_AUDIO_DISABLE_CUDNN=1 before loading vLLM.",
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


def cache_dir_name(source: str) -> str:
    source_key = source.rstrip("/")
    source_name = Path(source_key).name or "model"
    digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:10]
    return f"{source_name}-{digest}"


def stage_inputs(args: argparse.Namespace) -> tuple[str, str]:
    if args.skip_stage:
        return args.model, args.audio

    cache_root = Path(args.local_cache_root)
    model_path = stage_input(
        args.model,
        cache_root / "models" / cache_dir_name(args.model),
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
        import torchaudio.functional as F

        waveform = F.resample(
            torch.from_numpy(waveform),
            sample_rate,
            TARGET_SAMPLE_RATE,
        ).numpy()
        sample_rate = TARGET_SAMPLE_RATE
    return waveform.astype(np.float32), sample_rate


def build_prompt(model_path: str, instruction: str, assistant_prefix: str) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    _chat_obj = tokenizer
    prompt = _chat_obj.apply_chat_template(
        [{"role": "user", "content": instruction}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return f"{prompt}{assistant_prefix}"


def package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "missing"


def print_environment() -> None:
    import torch

    for package in (
        "vllm",
        "torch",
        "transformers",
        "flashinfer-python",
        "flashinfer-cubin",
    ):
        print(f"{package}={package_version(package)}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device0={torch.cuda.get_device_name(0)}")


def register_local_plugin() -> None:
    plugin_src = Path(__file__).resolve().parents[1] / "src"
    if plugin_src.exists():
        sys.path.insert(0, str(plugin_src))

    from vllm import ModelRegistry

    if MODEL_ARCHITECTURE in ModelRegistry.get_supported_archs():
        return

    from vllm_qwen35_audio.plugin import register

    register()


def main() -> None:
    args = parse_args()
    model_path, audio_path = stage_inputs(args)

    os.environ.setdefault("VLLM_PLUGINS", "qwen35_audio")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if not args.keep_cudnn_enabled:
        os.environ.setdefault("QWEN35_AUDIO_DISABLE_CUDNN", "1")

    register_local_plugin()

    from vllm import LLM, SamplingParams

    print_environment()
    print(f"model_source={args.model}")
    print(f"audio_source={args.audio}")
    print(f"model_path={model_path}")
    print(f"audio_path={audio_path}")

    waveform, sample_rate = load_audio(audio_path)
    print(f"audio_seconds={len(waveform) / sample_rate:.2f} sample_rate={sample_rate}")
    prompt = build_prompt(model_path, args.instruction, args.assistant_prefix)
    print(f"prompt={prompt!r}")

    start_time = time.time()
    llm = LLM(
        model=model_path,
        trust_remote_code=args.trust_remote_code,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        load_format="auto",
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={"audio": 1},
        gpu_memory_utilization=args.gpu_memory_utilization,
        logits_processors=[
            "vllm.model_executor.models.deepseek_ocr:"
            "NGramPerReqLogitsProcessor"
        ],
    )
    print(f"load_seconds={time.time() - start_time:.1f}")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop_token_ids=args.stop_token_id or DEFAULT_STOP_TOKEN_IDS,
        repetition_penalty=1.0,
        extra_args={"ngram_size": 15, "window_size": 512},
    )
    start_time = time.time()
    outputs = llm.generate(
        [
            {
                "prompt": prompt,
                "multi_modal_data": {"audio": [(waveform, sample_rate)]},
            }
        ],
        sampling_params=sampling_params,
    )
    print(f"generate_seconds={time.time() - start_time:.1f}")
    print("TRANSCRIPT_START")
    print(outputs[0].outputs[0].text.strip())
    print("TRANSCRIPT_END")


if __name__ == "__main__":
    main()

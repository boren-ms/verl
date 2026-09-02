#!/usr/bin/env python3
"""Plot per-token prompt KL estimates on a student-generated audio response."""

import argparse
import importlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .qwen35_audio_utils import add_input_arguments, load_audio, stage_inputs
except ImportError:
    from qwen35_audio_utils import add_input_arguments, load_audio, stage_inputs


REMOTE_MODEL_PATH = (
    "az://orngwus2cresco/data/speech/projects/phi-fastllm-2607/amlt-results/"
    "fast-llm-2607-qwen3-5-9b-s2-data-v3.4-sr-afteraudio-lexical-fix/50000/qwen_hf/"
)
REMOTE_AUDIO_PATH = (
    "az://orngwus2cresco/data/boren/data/LibriSpeech/test-clean/1089/"
    "134691/1089-134691-0024.flac"
)
LOCAL_CACHE_ROOT = "/root/data/qwen35_audio_test"
DEFAULT_INSTRUCTION = "Detect the language and transcribe the audio clip into text.<audio>"
DEFAULT_STOP_TOKEN_IDS = [248044, 248046]
MODEL_ARCHITECTURE = "Qwen3_5AudioForCausalLM"
DEFAULT_TRANSCRIPTION = "stephanos dedalos"
DEFAULT_KEYWORDS = ["stephanos"]


def build_teacher_instruction(instruction: str, keywords: list[str]) -> str:
    terms = ", ".join(f"*{keyword}*" for keyword in keywords)
    return f"{instruction}\nPay extra attention to the following phrases/words: {terms}."


def validate_keywords(transcription: str, keywords: list[str]) -> list[str]:
    cleaned = [keyword.strip() for keyword in keywords]
    if not cleaned or any(not keyword for keyword in cleaned):
        raise ValueError("at least one non-empty keyword is required")

    normalized_transcription = re.sub(r"\s+", " ", transcription).casefold()
    missing = [
        keyword
        for keyword in cleaned
        if re.sub(r"\s+", " ", keyword).casefold() not in normalized_transcription
    ]
    if missing:
        raise ValueError(f"keywords not found in transcription: {', '.join(missing)}")
    return cleaned


def extract_suffix_logprobs(
    prompt_token_ids: list[int],
    prompt_logprobs: list[dict[int, Any] | None],
    suffix_token_ids: list[int],
) -> list[float]:
    if not suffix_token_ids:
        raise ValueError("transcription produced no token IDs")
    if prompt_token_ids[-len(suffix_token_ids) :] != suffix_token_ids:
        raise ValueError("reference token IDs are not the final prompt-token suffix")
    if len(prompt_logprobs) != len(prompt_token_ids):
        raise ValueError("prompt token IDs and prompt log-probabilities have different lengths")

    values = []
    offset = len(prompt_token_ids) - len(suffix_token_ids)
    for position, token_id in enumerate(suffix_token_ids, start=offset):
        candidates = prompt_logprobs[position]
        if candidates is None or token_id not in candidates:
            raise ValueError(f"missing chosen-token log-probability at prompt position {position}")
        value = candidates[token_id]
        values.append(float(value.logprob if hasattr(value, "logprob") else value))
    return values


def extract_chosen_logprobs(
    token_ids: list[int],
    token_logprobs: list[dict[int, Any] | None],
    source: str,
) -> list[float]:
    if not token_ids:
        raise ValueError(f"{source} produced no token IDs")
    if len(token_logprobs) != len(token_ids):
        raise ValueError(f"{source} token IDs and log-probabilities have different lengths")

    values = []
    for position, (token_id, candidates) in enumerate(zip(token_ids, token_logprobs, strict=True)):
        if candidates is None or token_id not in candidates:
            raise ValueError(f"missing chosen-token log-probability at {source} position {position}")
        value = candidates[token_id]
        values.append(float(value.logprob if hasattr(value, "logprob") else value))
    return values


def compute_k3_estimates(student_logprobs: list[float], teacher_logprobs: list[float]) -> list[float]:
    if len(student_logprobs) != len(teacher_logprobs):
        raise ValueError("student and teacher log-probability lengths differ")

    estimates = []
    for student_logprob, teacher_logprob in zip(student_logprobs, teacher_logprobs, strict=True):
        log_ratio = max(-20.0, min(20.0, teacher_logprob - student_logprob))
        estimates.append(max(-10.0, min(10.0, math.exp(log_ratio) - log_ratio - 1.0)))
    return estimates


def compute_k2_estimates(student_logprobs: list[float], teacher_logprobs: list[float]) -> list[float]:
    if len(student_logprobs) != len(teacher_logprobs):
        raise ValueError("student and teacher log-probability lengths differ")
    return [
        0.5 * (student_logprob - teacher_logprob) ** 2
        for student_logprob, teacher_logprob in zip(
            student_logprobs,
            teacher_logprobs,
            strict=True,
        )
    ]


def visible_token(token: str) -> str:
    return token.replace(" ", "\\s").replace("\n", "\\n").replace("\t", "\\t") or "<empty>"


def transcript_token_fragments(token_text: list[str]) -> list[tuple[int, str]]:
    response = "".join(token_text)
    content_start = response.find("<TXT>")
    content_end = response.find("</TXT>", content_start + len("<TXT>"))
    if content_start < 0 or content_end < 0:
        return [(index, token) for index, token in enumerate(token_text)]

    content_start += len("<TXT>")
    fragments = []
    token_start = 0
    for index, token in enumerate(token_text):
        token_end = token_start + len(token)
        overlap_start = max(token_start, content_start)
        overlap_end = min(token_end, content_end)
        if overlap_start < overlap_end:
            fragment = token[overlap_start - token_start : overlap_end - token_start]
            fragments.append((index, fragment))
        token_start = token_end
    return fragments


def normalized_words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def student_response_ticks(fragments: list[str]) -> tuple[list[float], list[str]]:
    response = "".join(fragments)
    token_spans = []
    offset = 0
    for position, fragment in enumerate(fragments):
        token_spans.append((position, offset, offset + len(fragment)))
        offset += len(fragment)

    tick_positions = []
    tick_labels = []
    for word_match in re.finditer(r"\S+", response):
        positions = [
            position
            for position, token_start, token_end in token_spans
            if token_start < word_match.end() and token_end > word_match.start()
        ]
        if positions:
            tick_positions.append(sum(positions) / len(positions))
            tick_labels.append(word_match.group())
    return tick_positions, tick_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a response with the regular prompt, score those exact tokens under a "
            "keyword-augmented prompt, then plot the pointwise k3 KL estimate."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_input_arguments(
        parser,
        default_model_path=REMOTE_MODEL_PATH,
        default_audio_path=REMOTE_AUDIO_PATH,
        default_cache_root=LOCAL_CACHE_ROOT,
    )
    parser.add_argument(
        "--transcription",
        default=DEFAULT_TRANSCRIPTION,
        help="Ground-truth transcription used to validate the teacher keywords.",
    )
    parser.add_argument(
        "--keyword",
        action="append",
        default=None,
        help=(
            "Word or phrase from the transcription to emphasize. May be passed multiple times. "
            f"Defaults for the bundled audio: {', '.join(DEFAULT_KEYWORDS)}"
        ),
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--output", default="qwen35_audio_prompt_kl.png", help="Output PNG path.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--keep-cudnn-enabled", action="store_true")
    return parser.parse_args()


def build_chat_prefix(tokenizer: Any, instruction: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        add_generation_prompt=True,
        tokenize=False,
    )


def register_local_plugin() -> None:
    plugin_src = Path(__file__).resolve().parents[1] / "src"
    if plugin_src.exists():
        sys.path.insert(0, str(plugin_src))

    from vllm import ModelRegistry

    if MODEL_ARCHITECTURE not in ModelRegistry.get_supported_archs():
        importlib.import_module("vllm_qwen35_audio.plugin").register()


def generate_and_score(
    llm: Any,
    tokenizer: Any,
    waveform: Any,
    sample_rate: int,
    student_instruction: str,
    teacher_instruction: str,
    max_tokens: int,
    temperature: float,
) -> tuple[list[int], list[str], list[float], list[float]]:
    from vllm import SamplingParams

    student_prompt = build_chat_prefix(tokenizer, student_instruction)
    student_outputs = llm.generate(
        [
            {
                "prompt": student_prompt,
                "multi_modal_data": {"audio": [(waveform, sample_rate)]},
            }
        ],
        sampling_params=SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=0,
            stop_token_ids=DEFAULT_STOP_TOKEN_IDS,
            repetition_penalty=1.0,
            extra_args={"ngram_size": 15, "window_size": 512},
        ),
    )
    if len(student_outputs) != 1 or len(student_outputs[0].outputs) != 1:
        raise RuntimeError("expected one student completion")
    student_completion = student_outputs[0].outputs[0]
    student_token_ids = list(student_completion.token_ids)
    student_logprobs = extract_chosen_logprobs(
        student_token_ids,
        student_completion.logprobs,
        "student response",
    )

    teacher_prefix = build_chat_prefix(tokenizer, teacher_instruction)
    teacher_prefix_token_ids = tokenizer.encode(teacher_prefix, add_special_tokens=False)
    teacher_outputs = llm.generate(
        [
            {
                "prompt_token_ids": teacher_prefix_token_ids + student_token_ids,
                "multi_modal_data": {"audio": [(waveform, sample_rate)]},
            }
        ],
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=0,
            stop_token_ids=DEFAULT_STOP_TOKEN_IDS,
            repetition_penalty=1.0,
            extra_args={"ngram_size": 15, "window_size": 512},
        ),
    )
    if len(teacher_outputs) != 1:
        raise RuntimeError(f"expected one teacher scoring output, got {len(teacher_outputs)}")
    teacher_output = teacher_outputs[0]
    teacher_logprobs = extract_suffix_logprobs(
        teacher_output.prompt_token_ids,
        teacher_output.prompt_logprobs,
        student_token_ids,
    )

    token_text = [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for token_id in student_token_ids
    ]
    return student_token_ids, token_text, student_logprobs, teacher_logprobs


def write_outputs(output_path: Path, report: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokens = report["tokens"]
    tokens["k2_estimate"] = compute_k2_estimates(
        tokens["student_logprob"],
        tokens["teacher_logprob"],
    )
    tokens["student_probability"] = [math.exp(logprob) for logprob in tokens["student_logprob"]]
    tokens["teacher_probability"] = [math.exp(logprob) for logprob in tokens["teacher_logprob"]]
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fragments = transcript_token_fragments(tokens["text"])
    indices = [index for index, _ in fragments]
    fragment_text = [fragment for _, fragment in fragments]
    estimates = [tokens["k2_estimate"][index] for index in indices]
    student_probabilities = [tokens["student_probability"][index] for index in indices]
    teacher_probabilities = [tokens["teacher_probability"][index] for index in indices]
    student_transcript = "".join(fragment_text).strip()
    tick_positions, tick_labels = student_response_ticks(fragment_text)
    is_incorrect = normalized_words(student_transcript) != normalized_words(report["transcription"])

    figure, (k2_axis, probability_axis) = plt.subplots(
        2,
        1,
        figsize=(12.5, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.16},
    )
    positions = list(range(len(labels)))
    k2_axis.bar(positions, estimates, color="#c84c3f", width=0.72, zorder=3)
    k2_axis.set_ylabel("Token-wise k2", fontsize=11)
    k2_axis.set_title("Prompt disagreement: 0.5 x (student log p - teacher log p)^2", fontsize=12)
    k2_axis.grid(axis="y", alpha=0.2, zorder=0)
    if estimates:
        largest_position = estimates.index(max(estimates))
        k2_axis.annotate(
            "Largest student-teacher\ndistribution mismatch",
            xy=(largest_position, estimates[largest_position]),
            xytext=(largest_position, max(estimates) * 0.62),
            arrowprops={"arrowstyle": "->", "color": "#8f3028"},
            color="#8f3028",
            fontweight="bold",
            ha="center",
        )

    bar_width = 0.36
    probability_axis.bar(
        [position - bar_width / 2 for position in positions],
        student_probabilities,
        width=bar_width,
        color="#3676a8",
        label="Student prompt",
        zorder=3,
    )
    probability_axis.bar(
        [position + bar_width / 2 for position in positions],
        teacher_probabilities,
        width=bar_width,
        color="#d79032",
        label="Teacher keyword prompt",
        zorder=3,
    )
    probability_axis.set_ylim(0, 1.05)
    probability_axis.set_ylabel("Probability of generated token", fontsize=11)
    probability_axis.set_xlabel(f"Student response: {student_transcript}", fontsize=11, fontweight="bold")
    probability_axis.set_xticks(tick_positions, tick_labels, fontsize=11)
    probability_axis.grid(axis="y", alpha=0.2, zorder=0)
    probability_axis.legend(loc="upper right", frameon=False)

    verdict = "INCORRECT" if is_incorrect else "MATCHES REFERENCE"
    headline = "Token-wise k2 on the model's real <TXT> output"
    figure.suptitle(headline, fontsize=18, fontweight="bold")
    figure.text(0.08, 0.89, f"Reference:  {report['transcription']}", fontsize=12)
    figure.text(
        0.08,
        0.855,
        f"Student:     {student_transcript}  [{verdict}]",
        fontsize=12,
        color="#a33b32" if is_incorrect else "#2f7d5c",
        fontweight="bold",
    )
    figure.text(0.08, 0.82, f"Keyword:    {', '.join(report['keywords'])}", fontsize=12)
    figure.text(
        0.08,
        0.785,
        "Higher bars mean the regular and keyword prompts disagree more on that output token.",
        fontsize=12,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.09, top=0.7)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return json_path


def main() -> None:
    args = parse_args()
    transcription = args.transcription.strip()
    if not transcription:
        raise ValueError("--transcription must not be empty")
    keywords = validate_keywords(transcription, args.keyword or DEFAULT_KEYWORDS)
    model_path, audio_paths = stage_inputs(args)
    if len(audio_paths) != 1:
        raise ValueError(f"exactly one audio input is required, got {len(audio_paths)}")

    os.environ.setdefault("VLLM_PLUGINS", "qwen35_audio")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if not args.keep_cudnn_enabled:
        os.environ.setdefault("QWEN35_AUDIO_DISABLE_CUDNN", "1")
    register_local_plugin()

    from transformers import AutoTokenizer
    from vllm import LLM

    audio_source, audio_path = audio_paths[0]
    waveform, sample_rate = load_audio(audio_path)
    teacher_instruction = build_teacher_instruction(args.instruction, keywords)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

    start_time = time.time()
    llm = LLM(
        model=model_path,
        trust_remote_code=args.trust_remote_code,
        max_model_len=args.max_model_len,
        max_num_seqs=2,
        load_format="auto",
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={"audio": 1},
        gpu_memory_utilization=args.gpu_memory_utilization,
        logits_processors=["vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor"],
    )
    print(f"load_seconds={time.time() - start_time:.1f}")

    start_time = time.time()
    token_ids, token_text, student_logprobs, teacher_logprobs = generate_and_score(
        llm,
        tokenizer,
        waveform,
        sample_rate,
        args.instruction,
        teacher_instruction,
        args.max_tokens,
        args.temperature,
    )
    estimates = compute_k3_estimates(student_logprobs, teacher_logprobs)
    k2_estimates = compute_k2_estimates(student_logprobs, teacher_logprobs)
    log_ratios = [
        teacher - student
        for student, teacher in zip(student_logprobs, teacher_logprobs, strict=True)
    ]
    report = {
        "metric": "pointwise_k3_kl_estimate_on_student_response_tokens",
        "audio_source": audio_source,
        "audio_path": audio_path,
        "model_source": args.model,
        "model_path": model_path,
        "student_prompt": args.instruction,
        "teacher_prompt": teacher_instruction,
        "transcription": transcription,
        "keywords": keywords,
        "student_response": tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
        "tokens": {
            "id": token_ids,
            "text": token_text,
            "student_logprob": student_logprobs,
            "teacher_logprob": teacher_logprobs,
            "teacher_minus_student_logprob": log_ratios,
            "k2_estimate": k2_estimates,
            "k3_estimate": estimates,
        },
        "summary": {
            "token_count": len(token_ids),
            "k3_sum": sum(estimates),
            "k3_mean": sum(estimates) / len(estimates),
        },
    }
    output_path = Path(args.output)
    json_path = write_outputs(output_path, report)
    print(f"score_seconds={time.time() - start_time:.1f}")
    print(f"png_output={output_path}")
    print(f"json_output={json_path}")


if __name__ == "__main__":
    main()

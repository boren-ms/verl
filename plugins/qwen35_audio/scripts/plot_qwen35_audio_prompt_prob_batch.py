#!/usr/bin/env python3
"""Plot student-teacher token probability differences for biasing examples."""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

try:
    from .plot_qwen35_audio_prompt_kl import (
        DEFAULT_INSTRUCTION,
        build_teacher_instruction,
        generate_and_score,
        register_local_plugin,
        transcript_token_fragments,
        validate_keywords,
    )
    from .qwen35_audio_utils import cache_file_name, load_audio, stage_input
except ImportError:
    from plot_qwen35_audio_prompt_kl import (
        DEFAULT_INSTRUCTION,
        build_teacher_instruction,
        generate_and_score,
        register_local_plugin,
        transcript_token_fragments,
        validate_keywords,
    )
    from qwen35_audio_utils import cache_file_name, load_audio, stage_input


DEFAULT_MANIFEST = (
    "az://orngwus2cresco/data/boren/data/librispeech_biasing/ref/ls_other_kw_fix.jsonl"
)
DEFAULT_MODEL = (
    "az://orngwus2cresco/data/speech/projects/phi-fastllm-2607/amlt-results/"
    "fast-llm-2607-qwen3-5-9b-s2-data-v3.4-sr-afteraudio-lexical-fix/50000/qwen_hf/"
)
ORANGE_LIBRISPEECH_ROOT = "az://orngwus2cresco/data/boren/data/LibriSpeech"
LOCAL_CACHE_ROOT = "/root/data/qwen35_audio_prob_batch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--local-cache-root", default=LOCAL_CACHE_ROOT)
    parser.add_argument("--output", default="qwen35_audio_prompt_prob_batch.png")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--keep-cudnn-enabled", action="store_true")
    return parser.parse_args()


def orange_audio_path(audio_path: str) -> str:
    marker = "/LibriSpeech/"
    if audio_path.startswith("az://"):
        return audio_path
    if marker not in audio_path:
        raise ValueError(f"cannot map audio path to Orange LibriSpeech root: {audio_path}")
    relative_path = audio_path.split(marker, maxsplit=1)[1]
    return f"{ORANGE_LIBRISPEECH_ROOT}/{relative_path}"


def read_examples(manifest_path: Path, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError(f"--limit must be positive, got {limit}")
    examples = []
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            keywords = validate_keywords(record["text"], record["keywords"])
            examples.append(
                {
                    "line_number": line_number,
                    "id": str(record["id"]),
                    "audio_source": orange_audio_path(str(record["audio_path"])),
                    "transcription": str(record["text"]),
                    "keywords": keywords,
                }
            )
            if len(examples) == limit:
                break
    if len(examples) != limit:
        raise ValueError(f"requested {limit} examples but found {len(examples)}")
    return examples


def token_probability_rows(
    token_text: list[str],
    student_logprobs: list[float],
    teacher_logprobs: list[float],
) -> list[dict[str, Any]]:
    rows = []
    for index, fragment in transcript_token_fragments(token_text):
        student_probability = math.exp(student_logprobs[index])
        teacher_probability = math.exp(teacher_logprobs[index])
        rows.append(
            {
                "index": index,
                "text": fragment,
                "student_probability": student_probability,
                "teacher_probability": teacher_probability,
                "teacher_minus_student_probability": teacher_probability - student_probability,
            }
        )
    return rows


def write_report(output_path: Path, report: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    examples = report["examples"]
    figure, axes = plt.subplots(len(examples), 1, figsize=(18, 3.4 * len(examples)))
    if len(examples) == 1:
        axes = [axes]
    for axis, example in zip(axes, examples, strict=True):
        rows = example["tokens"]
        positions = list(range(len(rows)))
        differences = [row["teacher_minus_student_probability"] for row in rows]
        labels = [row["text"].replace(" ", "\\s") or "<empty>" for row in rows]
        colors = ["#2f7d5c" if difference >= 0 else "#c84c3f" for difference in differences]
        axis.bar(positions, differences, color=colors, width=0.72)
        axis.axhline(0, color="#252525", linewidth=0.8)
        axis.set_xticks(positions, labels, rotation=45, ha="right", fontsize=8)
        axis.set_ylabel("teacher - student")
        axis.set_title(
            f"{example['id']} | keywords: {', '.join(example['keywords'])}\n"
            f"student TXT: {example['student_transcript']}",
            loc="left",
            fontsize=10,
        )
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("Student response tokens inside <TXT>...</TXT>")
    figure.suptitle(
        "Keyword-prompt effect on generated-token probability\n"
        "green: teacher increases probability | red: teacher decreases probability",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return json_path


def main() -> None:
    args = parse_args()
    cache_root = Path(args.local_cache_root)
    manifest_path = Path(
        stage_input(args.manifest, cache_root / "manifest" / Path(args.manifest).name, is_dir=False)
    )
    model_path = stage_input(args.model, cache_root / "model", is_dir=True)
    examples = read_examples(manifest_path, args.limit)

    os.environ.setdefault("VLLM_PLUGINS", "qwen35_audio")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if not args.keep_cudnn_enabled:
        os.environ.setdefault("QWEN35_AUDIO_DISABLE_CUDNN", "1")
    register_local_plugin()

    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    load_start = time.time()
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
    print(f"load_seconds={time.time() - load_start:.1f}")

    result_examples = []
    for example_number, example in enumerate(examples, start=1):
        local_audio_path = stage_input(
            example["audio_source"],
            cache_root / "audio" / cache_file_name(example["audio_source"]),
            is_dir=False,
        )
        waveform, sample_rate = load_audio(local_audio_path)
        teacher_instruction = build_teacher_instruction(args.instruction, example["keywords"])
        score_start = time.time()
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
        rows = token_probability_rows(token_text, student_logprobs, teacher_logprobs)
        student_transcript = "".join(row["text"] for row in rows).strip()
        result_examples.append(
            {
                **example,
                "audio_path": local_audio_path,
                "teacher_prompt": teacher_instruction,
                "student_response": tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
                "student_transcript": student_transcript,
                "token_ids": token_ids,
                "tokens": rows,
                "score_seconds": time.time() - score_start,
            }
        )
        print(
            f"EXAMPLE_DONE {example_number}/{len(examples)} id={example['id']} "
            f"tokens={len(rows)} score_seconds={result_examples[-1]['score_seconds']:.2f}"
        )

    report = {
        "metric": "teacher_minus_student_probability_on_student_txt_tokens",
        "manifest_source": args.manifest,
        "model_source": args.model,
        "student_prompt": args.instruction,
        "selection": "first_valid_records",
        "count": len(result_examples),
        "examples": result_examples,
    }
    output_path = Path(args.output)
    json_path = write_report(output_path, report)
    print(f"png_output={output_path}")
    print(f"json_output={json_path}")


if __name__ == "__main__":
    main()
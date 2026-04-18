#!/usr/bin/env python3
"""
Qwen3 Forced Aligner: Align transcription with speech to get word-level timestamps.

Uses Qwen/Qwen3-ForcedAligner-0.6B to produce per-word start/end times from an
(audio, transcript) pair. Supports local audio files, URLs, and batch processing.

Usage:
    # Single file with known transcript
    python scripts/qwen_aligner.py --audio /path/to/audio.wav --text "Hello world" --language English

    # Transcribe first (ASR) then align
    python scripts/qwen_aligner.py --audio /path/to/audio.wav --language English

    # Batch from a JSONL file (each line: {"audio": "path", "text": "...", "language": "..."})
    python scripts/qwen_aligner.py --input batch.jsonl --output aligned.jsonl

    # Use flash attention for speed
    python scripts/qwen_aligner.py --audio audio.wav --text "Hello world" --language English --flash-attn
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def load_aligner(model_name, device, dtype, flash_attn=False):
    """Load Qwen3ForcedAligner model."""
    from qwen_asr import Qwen3ForcedAligner

    kwargs = dict(dtype=dtype, device_map=device)
    if flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    print(f"Loading aligner: {model_name} on {device} ({dtype})")
    model = Qwen3ForcedAligner.from_pretrained(model_name, **kwargs)
    print("Aligner loaded.")
    return model


def load_asr(model_name, device, dtype, flash_attn=False, aligner_model=None, aligner_kwargs=None):
    """Load Qwen3ASRModel with optional forced aligner for timestamps."""
    from qwen_asr import Qwen3ASRModel

    kwargs = dict(dtype=dtype, device_map=device, max_new_tokens=2048, max_inference_batch_size=8)
    if flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    if aligner_model:
        kwargs["forced_aligner"] = aligner_model
        fa_kwargs = dict(dtype=dtype, device_map=device)
        if flash_attn:
            fa_kwargs["attn_implementation"] = "flash_attention_2"
        if aligner_kwargs:
            fa_kwargs.update(aligner_kwargs)
        kwargs["forced_aligner_kwargs"] = fa_kwargs

    print(f"Loading ASR: {model_name} on {device} ({dtype})")
    model = Qwen3ASRModel.from_pretrained(model_name, **kwargs)
    print("ASR model loaded.")
    return model


def align_single(aligner, audio, text, language):
    """Align a single audio-text pair and return word timestamps."""
    results = aligner.align(audio=audio, text=text, language=language)
    # results[0] is a list of WordTimestamp objects
    word_timings = []
    for w in results[0]:
        word_timings.append({
            "text": w.text,
            "start": w.start_time,
            "end": w.end_time,
        })
    return word_timings


def transcribe_and_align(asr_model, audio, language=None):
    """Transcribe audio and return text + word timestamps."""
    results = asr_model.transcribe(
        audio=audio,
        language=language,
        return_time_stamps=True,
    )
    r = results[0]
    word_timings = []
    if r.time_stamps:
        for w in r.time_stamps[0]:
            word_timings.append({
                "text": w.text,
                "start": w.start_time,
                "end": w.end_time,
            })
    return r.language, r.text, word_timings


def format_timestamp(seconds):
    """Format seconds as HH:MM:SS.mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def print_word_timings(word_timings, file=sys.stdout):
    """Pretty-print word timings."""
    for w in word_timings:
        start_str = format_timestamp(w["start"])
        end_str = format_timestamp(w["end"])
        print(f"  [{start_str} --> {end_str}]  {w['text']}", file=file)


def main():
    parser = argparse.ArgumentParser(description="Qwen3 Forced Aligner for word-level timestamps")
    parser.add_argument("--audio", type=str, help="Path or URL to a single audio file")
    parser.add_argument("--text", type=str, default=None, help="Known transcript to align (if omitted, ASR is run first)")
    parser.add_argument("--language", type=str, default=None, help="Language (e.g. English, Chinese). Auto-detected if omitted.")
    parser.add_argument("--input", type=str, default=None, help="JSONL file for batch processing")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file for batch results")
    parser.add_argument("--aligner-model", type=str, default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--asr-model", type=str, default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--flash-attn", action="store_true", help="Use FlashAttention 2")
    parser.add_argument("--srt", action="store_true", help="Output in SRT subtitle format")
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # --- Single audio mode ---
    if args.audio:
        if args.text:
            # Alignment only (transcript provided)
            aligner = load_aligner(args.aligner_model, args.device, dtype, args.flash_attn)
            t0 = time.time()
            word_timings = align_single(aligner, args.audio, args.text, args.language or "English")
            elapsed = time.time() - t0
            print(f"\nAlignment ({elapsed:.2f}s):")
            print(f"Text: {args.text}")
            print_word_timings(word_timings)

            if args.srt:
                srt_path = args.output or "output.srt"
                with open(srt_path, "w") as f:
                    for i, w in enumerate(word_timings, 1):
                        start_str = format_timestamp(w["start"]).replace(".", ",")
                        end_str = format_timestamp(w["end"]).replace(".", ",")
                        f.write(f"{i}\n{start_str} --> {end_str}\n{w['text']}\n\n")
                print(f"SRT written to {srt_path}")
        else:
            # ASR + alignment
            asr_model = load_asr(
                args.asr_model, args.device, dtype, args.flash_attn,
                aligner_model=args.aligner_model,
            )
            t0 = time.time()
            language, text, word_timings = transcribe_and_align(asr_model, args.audio, args.language)
            elapsed = time.time() - t0
            print(f"\nASR + Alignment ({elapsed:.2f}s):")
            print(f"Language: {language}")
            print(f"Text: {text}")
            print_word_timings(word_timings)

        if args.output and not args.srt:
            out = {"audio": args.audio, "text": args.text or text, "word_timings": word_timings}
            with open(args.output, "w") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"Output written to {args.output}")
        return

    # --- Batch mode ---
    if args.input:
        aligner = None
        asr_model = None

        items = []
        with open(args.input) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))

        print(f"Processing {len(items)} items from {args.input}")
        results = []

        for i, item in enumerate(items):
            audio = item["audio"]
            text = item.get("text")
            language = item.get("language", args.language)

            if text:
                # Align only
                if aligner is None:
                    aligner = load_aligner(args.aligner_model, args.device, dtype, args.flash_attn)
                word_timings = align_single(aligner, audio, text, language or "English")
                results.append({"audio": audio, "text": text, "language": language, "word_timings": word_timings})
            else:
                # ASR + align
                if asr_model is None:
                    asr_model = load_asr(
                        args.asr_model, args.device, dtype, args.flash_attn,
                        aligner_model=args.aligner_model,
                    )
                lang, txt, word_timings = transcribe_and_align(asr_model, audio, language)
                results.append({"audio": audio, "text": txt, "language": lang, "word_timings": word_timings})

            print(f"  [{i+1}/{len(items)}] {audio}: {len(word_timings)} words")

        out_path = args.output or "aligned_output.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Results written to {out_path}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()

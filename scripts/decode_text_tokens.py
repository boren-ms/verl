#!/usr/bin/env python3
"""Simple hyp/ref token correctness labeling with Whisper normalization + difflib."""

from difflib import SequenceMatcher
import json
from pathlib import Path
import re

from transformers import AutoTokenizer
from whisper_normalizer.english import EnglishTextNormalizer


MODEL_PATH = str(Path("~/data/ckp/hf_models/Phi4-7b-STT-2603-SR2").expanduser())
HYP_TEXT = "At 07:45 on 2026-04-23, Dr. Li emailed ops-team+asr@example.com: 'Re-run patch #17 (en-US, 16kHz) with noise_level=0.12, beam=5, and reserved tokens like <edge_case>, C++17, and path /mnt/data/run_B/part-03.'"
REF_TEXT = "At 7:45 on 2026-04-23, Dr Lee emailed ops team asr@example.com: rerun batch 17 in en-US 16 kHz with noise level 0.12, beam 5, and preserve token like C++17 and path /mnt/data/run_A/part-3."
ADD_SPECIAL_TOKENS = False
OUTPUT_JSONL = Path("tmp/hyp_token_correctness.jsonl")


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
normalizer = EnglishTextNormalizer()


def find_pos(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    positions = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= start or tok_start >= end:
            continue
        positions.append(idx)
    return positions


def text_to_segments(
    text: str,
    offsets: list[tuple[int, int]],
    norm_fn=None,
) -> list[dict]:
    norm_fn = norm_fn or (lambda x: x)
    segments = []
    for match in re.finditer(r"\S+", text):
        start, end = match.span()
        segment = match.group(0)
        words = norm_fn(segment).strip().split()
        if not words:
            continue
        segments.append(
            {
                "segment": segment,
                "words": words,
                "span": [start, end],
                "idxs":  find_pos(offsets, start, end),
            }
        )

    return segments


encoded = tokenizer(
    HYP_TEXT,
    add_special_tokens=ADD_SPECIAL_TOKENS,
    return_offsets_mapping=True,
)
hyp_token_ids = encoded["input_ids"]
hyp_offsets = encoded["offset_mapping"]
hyp_tokens = tokenizer.convert_ids_to_tokens(hyp_token_ids)

hyp_normalized_text = normalizer(HYP_TEXT)
ref_normalized_text = normalizer(REF_TEXT)
ref_normalized_words = ref_normalized_text.split()

segments = text_to_segments(HYP_TEXT, hyp_offsets, normalizer)
hyp_normalized_words = [word for segment in segments for word in segment["words"]]

word_correctness = [0] * len(hyp_normalized_words)

matcher = SequenceMatcher(a=hyp_normalized_words, b=ref_normalized_words)
for tag, i1, i2, _, _ in matcher.get_opcodes():
    if tag == "equal":
        for i in range(i1, i2):
            word_correctness[i] = 1

token_correctness = [0] * len(hyp_token_ids)

wi = 0
for segment in segments:
    nw = len(segment["words"])
    label = sum(word_correctness[i] for i in range(wi, wi + nw))/nw
    for pos in segment["idxs"]:
        token_correctness[pos] = label
    wi += nw
    
rows = []
for i, (tid, tok, score) in enumerate(zip(hyp_token_ids, hyp_tokens, token_correctness)):
    rows.append(
        {
            "token_position": i,
            "token_id": tid,
            "token": tok,
            "correct": score,
        }
    )

OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("model:", MODEL_PATH)
print("hyp_text:", HYP_TEXT)
print("ref_text:", REF_TEXT)
print("hyp_normalized_text:", hyp_normalized_text)
print("ref_normalized_text:", ref_normalized_text)
print("hyp_normalized_words:", hyp_normalized_words)
print("ref_normalized_words:", ref_normalized_words)
print("token_correctness_rows:")
for row in rows:
    print(row)
print("jsonl_output:", str(OUTPUT_JSONL.resolve()))
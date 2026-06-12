"""
LLM Judge Client: Score ASR hypothesis quality on punctuation, capitalization, and digits.

Sends (baseline_hyp, target_hyp) pairs to a vLLM OpenAI-compatible server and
returns JSON scores {punc, cap, digital} each rated 1-5.

Usage:
    python scripts/llm_judge/client_score.py \
        --baseline "this is the baseline hypothesis" \
        --hypothesis "This is the target hypothesis." \
        --server http://localhost:8000

    # Batch mode from JSONL file (fields: baseline, hypothesis)
    python scripts/llm_judge/client_score.py \
        --input data.jsonl --output scored.jsonl \
        --server http://localhost:8000
"""

import argparse
import json
import re
import sys
from pathlib import Path

import requests

SYSTEM_PROMPT = """\
You are an expert ASR (Automatic Speech Recognition) quality evaluator. \
Your task is to score the quality of an ASR hypothesis compared to a baseline hypothesis. \
Evaluate the TARGET hypothesis on three dimensions, each scored from 1 (worst) to 5 (best):

1. **Punctuation (punc)**: How well does the target use punctuation? \
   Score 5 if punctuation is perfectly placed (periods, commas, question marks, etc.). \
   Score 1 if punctuation is entirely missing or severely incorrect.

2. **Capitalization (cap)**: How well does the target handle capitalization? \
   Score 5 if capitalization is correct (sentence starts, proper nouns, acronyms). \
   Score 1 if capitalization is entirely missing or severely incorrect.

3. **Digits (digital)**: How well does the target handle numbers and digits? \
   Score 5 if numbers are correctly formatted (e.g., "123" vs "one hundred twenty-three" as appropriate, dates, currencies). \
   Score 1 if numbers are severely misformatted or incorrect.

Compare the TARGET hypothesis against the BASELINE hypothesis. The baseline serves as a reference point. \
If the target is better than the baseline, score higher; if worse, score lower; if similar, score around 3.

You MUST respond with ONLY a JSON object in this exact format (no other text):
{"punc": <1-5>, "cap": <1-5>, "digital": <1-5>}

IMPORTANT: Do NOT include any thinking, explanation, or analysis. Output ONLY the JSON object, nothing else.
"""

USER_TEMPLATE = """\
BASELINE: {baseline}
TARGET: {hypothesis}

Score the TARGET hypothesis quality compared to the BASELINE. Respond with JSON only."""


def build_messages(baseline: str, hypothesis: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(baseline=baseline, hypothesis=hypothesis)},
    ]


def query_server(messages: list[dict], server: str, model: str, temperature: float = 0.0) -> str:
    url = f"{server}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def parse_scores(raw: str) -> dict:
    """Parse JSON scores from LLM response. Handles thinking blocks, markdown fences."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    # Find JSON object containing our keys anywhere in the response
    # This handles cases where the model outputs thinking text before JSON
    for match in re.finditer(r"\{[^}]*\}", cleaned):
        try:
            obj = json.loads(match.group())
            if "punc" in obj and "cap" in obj and "digital" in obj:
                scores = {
                    "punc": int(obj["punc"]),
                    "cap": int(obj["cap"]),
                    "digital": int(obj["digital"]),
                }
                for k, v in scores.items():
                    if not 1 <= v <= 5:
                        raise ValueError(f"Score {k}={v} out of range [1,5]")
                return scores
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    raise ValueError(f"Could not parse JSON scores from response: {raw!r}")


def score_single(baseline: str, hypothesis: str, server: str, model: str) -> dict:
    messages = build_messages(baseline, hypothesis)
    raw = query_server(messages, server, model)
    return parse_scores(raw)


def main():
    parser = argparse.ArgumentParser(description="LLM Judge: Score ASR hypothesis quality")
    parser.add_argument("--baseline", type=str, help="Baseline hypothesis text")
    parser.add_argument("--hypothesis", type=str, help="Target hypothesis text to score")
    parser.add_argument("--input", type=str, help="Input JSONL file with 'baseline' and 'hypothesis' fields")
    parser.add_argument("--output", type=str, help="Output JSONL file for batch mode")
    parser.add_argument("--server", type=str, default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--model", type=str, default="Qwen3.5-35B-A3B", help="Model name on the server")
    args = parser.parse_args()

    if args.input:
        # Batch mode
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else input_path.with_suffix(".scored.jsonl")
        results = []
        with open(input_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        print(f"Scoring {len(lines)} pairs...")
        for i, item in enumerate(lines):
            baseline = item["baseline"]
            hypothesis = item["hypothesis"]
            try:
                scores = score_single(baseline, hypothesis, args.server, args.model)
                item.update(scores)
                results.append(item)
                print(f"  [{i+1}/{len(lines)}] punc={scores['punc']} cap={scores['cap']} digital={scores['digital']}")
            except Exception as e:
                print(f"  [{i+1}/{len(lines)}] ERROR: {e}", file=sys.stderr)
                item.update({"punc": None, "cap": None, "digital": None, "error": str(e)})
                results.append(item)

        with open(output_path, "w") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Results written to {output_path}")

    elif args.baseline and args.hypothesis:
        # Single mode
        scores = score_single(args.baseline, args.hypothesis, args.server, args.model)
        print(json.dumps(scores, indent=2))

    else:
        parser.error("Provide either --baseline + --hypothesis, or --input for batch mode")


if __name__ == "__main__":
    main()

"""LLM Judge client for ASR formatting quality scoring.

Queries a vLLM OpenAI-compatible server to evaluate ASR hypothesis quality
on punctuation, capitalization, and digit formatting (each scored 1-3).

Scoring: 3 = better than baseline, 2 = similar, 1 = worse.

Can be used standalone or imported by reward functions.
"""

from __future__ import annotations

import json
import re

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert ASR (Automatic Speech Recognition) quality evaluator. \
Your task is to compare the quality of a TARGET hypothesis against a BASELINE hypothesis. \
Evaluate the TARGET on three dimensions, each scored 1, 2, or 3:

1. **Punctuation (punc)**: Compare punctuation quality (periods, commas, question marks, etc.). \
   Score 3 if TARGET has better punctuation than BASELINE. \
   Score 2 if TARGET has similar punctuation to BASELINE. \
   Score 1 if TARGET has worse punctuation than BASELINE.

2. **Capitalization (cap)**: Compare capitalization quality (sentence starts, proper nouns, acronyms). \
   Score 3 if TARGET has better capitalization than BASELINE. \
   Score 2 if TARGET has similar capitalization to BASELINE. \
   Score 1 if TARGET has worse capitalization than BASELINE.

3. **Digits (digital)**: Compare number/digit formatting (e.g., "123" vs "one hundred twenty-three", dates, currencies). \
   Score 3 if TARGET has better digit formatting than BASELINE. \
   Score 2 if TARGET has similar digit formatting to BASELINE. \
   Score 1 if TARGET has worse digit formatting than BASELINE.

You MUST respond with ONLY a JSON object in this exact format (no other text):
{"punc": <1-3>, "cap": <1-3>, "digital": <1-3>}

IMPORTANT: Do NOT include any thinking, explanation, or analysis. Output ONLY the JSON object, nothing else.
"""

USER_TEMPLATE = """\
BASELINE: {baseline}
TARGET: {hypothesis}

Compare TARGET against BASELINE. Respond with JSON only."""


# ---------------------------------------------------------------------------
# Client functions
# ---------------------------------------------------------------------------


def build_messages(baseline: str, hypothesis: str) -> list[dict]:
    """Build chat messages for the LLM judge."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(baseline=baseline, hypothesis=hypothesis)},
    ]


def query_server(messages: list[dict], server: str, model: str, temperature: float = 0.0) -> str:
    """Send messages to the vLLM server and return raw response text."""
    url = f"{server}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return content


def parse_scores(raw: str) -> dict:
    """Parse JSON scores from LLM response. Handles thinking blocks, markdown fences.

    Returns dict with keys: punc, cap, digital (each int 1-3).
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
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
                    if not 1 <= v <= 3:
                        raise ValueError(f"Score {k}={v} out of range [1,3]")
                return scores
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    raise ValueError(f"Could not parse JSON scores from response: {raw!r}")


@retry(
    stop=stop_after_attempt(30),
    wait=wait_exponential(multiplier=2, max=60),
    retry=retry_if_exception_type((requests.RequestException, ValueError)),
    reraise=True,
)
def query_judge(baseline: str, hypothesis: str, server: str, model: str) -> dict:
    """Query the LLM judge server and return {punc, cap, digital} scores (1-3).

    Scores: 3 = better than baseline, 2 = similar, 1 = worse.
    Retries up to 5 times with exponential backoff on request/parse failures.
    """
    messages = build_messages(baseline, hypothesis)
    raw = query_server(messages, server, model)
    return parse_scores(raw)

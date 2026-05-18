"""Send a single ASR transcription request to the vLLM proxy to smoke-test the
server end-to-end (proxy → worker → vLLM).

Usage:
    python -m recipe.phimm.vllm_server.test_request
    python -m recipe.phimm.vllm_server.test_request --proxy-url http://localhost:8000
    python -m recipe.phimm.vllm_server.test_request --audio-path az://... --prompt "..."
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

from recipe.phimm.data.prompts import get_task_prompt


DEFAULT_AUDIO_PATH = (
    "az://orngwus2cresco/data/boren/data/LibriSpeech/"
    "train-clean-360/115/122944/115-122944-0026.flac"
)
DEFAULT_TASK = "lang_asr"
DEFAULT_PROMPT = get_task_prompt(DEFAULT_TASK, rand=False)


def send_test_request(
    proxy_url: str = "http://localhost:8000",
    audio_path: str = DEFAULT_AUDIO_PATH,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 1024,
    max_audio_dur: float = 40.0,
    timeout: float = 600.0,
) -> dict:
    """Send one /asr/transcribe request and return the parsed response."""
    payload = {
        "audio_path": audio_path,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "max_audio_dur": max_audio_dur,
    }
    url = f"{proxy_url.rstrip('/')}/asr/transcribe"
    print(f"POST {url}")
    print(f"  audio_path = {audio_path}")
    t0 = time.time()
    resp = httpx.post(url, json=payload, timeout=timeout)
    elapsed = time.time() - t0
    print(f"  status     = {resp.status_code}  ({elapsed:.2f}s)")
    resp.raise_for_status()
    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"]
        print(f"  response   = {text!r}")
    except (KeyError, IndexError):
        print(f"  raw        = {json.dumps(data)[:500]}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-url", default="http://localhost:8000")
    parser.add_argument("--audio-path", default=DEFAULT_AUDIO_PATH)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-audio-dur", type=float, default=40.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    try:
        send_test_request(
            proxy_url=args.proxy_url,
            audio_path=args.audio_path,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            max_audio_dur=args.max_audio_dur,
            timeout=args.timeout,
        )
    except httpx.HTTPError as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

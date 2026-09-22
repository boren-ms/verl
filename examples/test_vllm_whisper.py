#!/usr/bin/env python3
"""Compatibility entry point for the Whisper-small vLLM smoke test."""

import sys
from pathlib import Path


def main() -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "plugins" / "qwen35_audio" / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from run_whisper_vllm import main as run_whisper

    run_whisper()


if __name__ == "__main__":
    main()

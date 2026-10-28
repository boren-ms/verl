#!/usr/bin/env python3
"""Create a collision-resistant experiment name for mixed OpenML AA evaluation."""

from __future__ import annotations

import argparse
import hashlib
import re


SUITE_LABEL = "mix-openml-aa"


def sanitize(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()


def checkpoint_label(checkpoint: str) -> str:
    parts = [part for part in checkpoint.rstrip("/").split("/") if part]
    if parts and parts[-1] in {"qwen_hf", "hf", "huggingface"}:
        parts.pop()
    for part in reversed(parts):
        match = re.search(r"global[_-]step[_-]?(\d+)", part, re.IGNORECASE)
        if match:
            return f"step{match.group(1)}"
    return sanitize(parts[-1] if parts else "checkpoint") or "checkpoint"


def make_experiment_name(model_label: str, checkpoint: str, role: str, attempt: int = 1) -> str:
    digest = hashlib.sha256(checkpoint.rstrip("/").encode("utf-8")).hexdigest()[:6]
    fields = [SUITE_LABEL, sanitize(model_label), checkpoint_label(checkpoint), sanitize(role), digest]
    if attempt > 1:
        fields.append(f"try{attempt}")
    return "-".join(fields)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--role", choices=("candidate", "reference"), required=True)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.attempt < 1:
        parser.error("--attempt must be at least 1")
    print(make_experiment_name(args.model_label, args.checkpoint, args.role, args.attempt))


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Launch one independent OpenLong Qwen shard per GPU and record outcomes."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Continue the same shard files after a failed attempt.")
    parser.add_argument("--attempt", default="initial", help="Unique attempt label; use a new label when resuming.")
    args = parser.parse_args()
    if args.gpu_count < 1:
        parser.error("gpu-count must be positive")
    if not args.attempt or not all(c.isalnum() or c in "-_" for c in args.attempt):
        parser.error("attempt may contain only letters, digits, hyphens, and underscores")
    if args.resume and (args.attempt == "initial" or not args.run_dir.is_dir()):
        parser.error("resume requires an existing run directory and a new attempt label")
    if not args.resume and args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.run_dir / ("status.json" if args.attempt == "initial" else f"status_{args.attempt}.json")
    if status_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {status_path}")
    jobs = []
    env = dict(os.environ, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    for index in range(args.gpu_count):
        log_path = args.run_dir / (f"shard_{index}.log" if args.attempt == "initial" else f"shard_{index}_{args.attempt}.log")
        log = log_path.open("x", encoding="utf-8")
        command = [sys.executable, "-u", "-m", "scripts.run_openlong_qwen_windows",
                   "--windows", str(args.windows), "--output", str(args.run_dir / f"shard_{index}.jsonl"),
                   "--model-path", args.model_path, "--device", f"cuda:{index}", "--flash-attn",
                   "--shard-count", str(args.gpu_count), "--shard-index", str(index)]
        if args.resume:
            command.append("--resume")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        jobs.append((index, process, log))
    while True:
        statuses = [{"shard": index, "pid": process.pid, "exit_code": process.poll()}
                    for index, process, _ in jobs]
        tmp = args.run_dir / "status.json.tmp"
        tmp.write_text(json.dumps({"shards": statuses}, indent=2) + "\n", encoding="utf-8")
        tmp.replace(status_path)
        if all(item["exit_code"] is not None for item in statuses):
            break
        time.sleep(10)
    for _, _, log in jobs:
        log.close()
    if any(item["exit_code"] != 0 for item in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

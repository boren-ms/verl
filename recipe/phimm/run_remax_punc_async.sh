#!/usr/bin/env bash
# ReMax ASR Punc+Cap Training | Qwen3.5-9B Audio | vLLM rollout | FSDP2 LoRA
# SPMD trainer (main_ppo.py) — serialized rollout/train phases, no GPU contention
#
# Uses Hydra config at recipe/phimm/config/remax_punc_async.yaml
# with optional CLI overrides.
#
# Usage:
#   bash recipe/phimm/run_remax_punc_async.sh
#   # Override steps/freq:
#   bash recipe/phimm/run_remax_punc_async.sh trainer.total_training_steps=1000 trainer.test_freq=50

set -xeuo pipefail

# Ensure we're in the repo root (needed for relative config paths)
cd "$(dirname "$0")/../.."

python3 -m verl.trainer.main_ppo \
    --config-path=../../recipe/phimm/config \
    --config-name=remax_punc_async \
    "$@"

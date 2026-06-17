#!/usr/bin/env bash
# Fully Async ReMax ASR Training | Qwen3.5-9B Audio | vLLM rollout | FSDP2 LoRA
#
# Fully async: rollout and training run on separate GPU pools concurrently.
# Default: 4 GPUs for rollout + 4 GPUs for training = 8 GPUs total.
#
# Uses Hydra config at recipe/phimm/config/remax_asr_fully_async.yaml
#
# Usage:
#   bash recipe/phimm/run_remax_asr_fully_async.sh
#   # Override GPU split:
#   N_GPUS_ROLLOUT=4 N_GPUS_TRAINING=4 bash recipe/phimm/run_remax_asr_fully_async.sh
#   # Override total rollout steps:
#   bash recipe/phimm/run_remax_asr_fully_async.sh rollout.total_rollout_steps=256

set -xeuo pipefail

# Ensure we're in the repo root
cd "$(dirname "$0")/../.."

NUM_GPUS=${NUM_GPUS:-8}
N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-$((NUM_GPUS / 2))}
N_GPUS_TRAINING=${N_GPUS_TRAINING:-$((NUM_GPUS / 2))}

export VLLM_USE_V1=${VLLM_USE_V1:-1}

echo "Running fully async ASR training"
echo "Total GPUs: ${NUM_GPUS}, Rollout GPUs: ${N_GPUS_ROLLOUT}, Training GPUs: ${N_GPUS_TRAINING}"

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=../../../recipe/phimm/config \
    --config-name=remax_asr_fully_async \
    rollout.n_gpus_per_node=${N_GPUS_ROLLOUT} \
    trainer.n_gpus_per_node=${N_GPUS_TRAINING} \
    "$@"

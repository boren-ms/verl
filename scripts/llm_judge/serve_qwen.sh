#!/bin/bash
# Serve Qwen3.5-35B-A3B with vLLM on 8 GPUs using tensor parallelism.
# Usage: bash scripts/llm_judge/serve_qwen.sh [MODEL_PATH] [PORT]
#
# On brix nodes without internet, sync model from blob first:
#   bbb sync az://orngwus2cresco/data/boren/data/verl/models/Qwen3.5-35B-A3B/ /root/data/models/Qwen3.5-35B-A3B/
# Then run:
#   bash scripts/llm_judge/serve_qwen.sh /root/data/models/Qwen3.5-35B-A3B 8000

MODEL="${1:-/root/data/models/Qwen3.5-35B-A3B}"
PORT="${2:-8000}"

echo "Starting vLLM server with model: $MODEL on port $PORT (TP=8)"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size 8 \
    --port "$PORT" \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --served-model-name Qwen3.5-35B-A3B

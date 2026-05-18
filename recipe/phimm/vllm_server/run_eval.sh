#!/usr/bin/env bash
# =============================================================================
# Multi-node vLLM Server Evaluation Runner
#
# Launches a FastAPI proxy + vLLM servers with worker sidecars (one per GPU,
# TP=1) and runs ASR evaluation on LibriSpeech using Qwen 3.5 Audio model.
#
# Architecture (per GPU):
#   ┌──────────────────────────────────────────────────────────────────────┐
#   │                      FastAPI Proxy (:8000)                          │
#   │          Least-connections routing + /asr/* + /v1/* routes          │
#   └──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬────────────┘
#          │      │      │      │      │      │      │      │
#   ┌──────▼─┐┌───▼──┐┌──▼───┐┌─▼────┐┌▼─────┐┌▼─────┐┌▼─────┐┌▼─────┐
#   │Worker0 ││Wkr1  ││Wkr2  ││Wkr3  ││Wkr4  ││Wkr5  ││Wkr6  ││Wkr7  │
#   │:8101   ││:8102 ││:8103 ││:8104 ││:8105 ││:8106 ││:8107 ││:8108  │
#   │audio+  ││audio ││audio ││audio ││audio ││audio ││audio ││audio  │
#   │chatmsg ││load  ││load  ││load  ││load  ││load  ││load  ││load   │
#   └───┬────┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬────┘
#       │        │       │       │       │       │       │       │
#   ┌───▼────┐┌──▼───┐┌──▼───┐┌─▼────┐┌─▼────┐┌─▼────┐┌─▼────┐┌─▼────┐
#   │vLLM0   ││vLLM1 ││vLLM2 ││vLLM3 ││vLLM4 ││vLLM5 ││vLLM6 ││vLLM7 │
#   │:8201   ││:8202 ││:8203 ││:8204 ││:8205 ││:8206 ││:8207 ││:8208 │
#   │GPU TP=1││GPU   ││GPU   ││GPU   ││GPU   ││GPU   ││GPU   ││GPU   │
#   └────────┘└──────┘└──────┘└──────┘└──────┘└──────┘└──────┘└──────┘
#
# Client sends {"audio_path": "az://..."} → Proxy → Worker loads audio → vLLM
#
# Multi-node: Run on each node. All nodes register workers with the same proxy.
#
# Usage:
#   # On the proxy node (also runs vLLM + workers):
#   bash recipe/phimm/vllm_server/run_eval.sh --role all
#
#   # On worker-only nodes (proxy runs elsewhere):
#   bash recipe/phimm/vllm_server/run_eval.sh --role worker --proxy-host <proxy-ip>
#
#   # Just run evaluation (servers already running):
#   bash recipe/phimm/vllm_server/run_eval.sh --role eval-only
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ---- Defaults (matching config.yaml) ----
MODEL_PATH="${MODEL_PATH:-az://orngwus2cresco/data/speech/projects/phi-fastllm-2605/amlt-results/fast-llm-2605-qwen3-5-9b-s2-st-example/90000/qwen_hf/}"
DATA_TSV="${DATA_TSV:-az://orngwus2cresco/data/boren/data/LibriSpeech/asr_train_transcribe.tsv}"
OUTPUT_PATH="${OUTPUT_PATH:-/tmp/vllm_eval_results}"
NUM_GPUS="${NUM_GPUS:-8}"
BASE_PORT="${BASE_PORT:-8101}"
PROXY_PORT="${PROXY_PORT:-8000}"
PROXY_HOST="${PROXY_HOST:-0.0.0.0}"
ROLE="${ROLE:-all}"  # all | proxy | worker | eval-only
AUDIO_WORKERS="${AUDIO_WORKERS:-8}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1024}"
NUM_EGS="${NUM_EGS:-}"  # empty = all
MAX_TOKENS="${MAX_TOKENS:-1024}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case $1 in
        --role) ROLE="$2"; shift 2;;
        --model) MODEL_PATH="$2"; shift 2;;
        --data-tsv) DATA_TSV="$2"; shift 2;;
        --output) OUTPUT_PATH="$2"; shift 2;;
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        --base-port) BASE_PORT="$2"; shift 2;;
        --proxy-port) PROXY_PORT="$2"; shift 2;;
        --proxy-host) PROXY_HOST="$2"; shift 2;;
        --audio-workers) AUDIO_WORKERS="$2"; shift 2;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2;;
        --num-egs) NUM_EGS="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# Resolve proxy URL
if [[ "$PROXY_HOST" == "0.0.0.0" ]]; then
    NODE_IP=$(hostname -I | awk '{print $1}')
    PROXY_URL="http://${NODE_IP}:${PROXY_PORT}"
else
    PROXY_URL="http://${PROXY_HOST}:${PROXY_PORT}"
fi

echo "============================================"
echo "vLLM Multi-Node Evaluation"
echo "============================================"
echo "Role:       $ROLE"
echo "Model:      $MODEL_PATH"
echo "Proxy URL:  $PROXY_URL"
echo "Num GPUs:   $NUM_GPUS"
echo "Base port:  $BASE_PORT"
echo "============================================"

# ---- Proxy ----
start_proxy() {
    echo "[INFO] Starting FastAPI proxy on port $PROXY_PORT..."
    python -m recipe.phimm.vllm_server.fastapi_proxy \
        --host 0.0.0.0 \
        --port "$PROXY_PORT" &
    PROXY_PID=$!
    echo "[INFO] Proxy PID: $PROXY_PID"

    # Wait for proxy to be ready
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:${PROXY_PORT}/health" > /dev/null 2>&1; then
            echo "[INFO] Proxy is ready"
            return 0
        fi
        sleep 1
    done
    echo "[ERROR] Proxy failed to start"
    return 1
}

# ---- vLLM + Worker Servers ----
start_servers() {
    echo "[INFO] Launching $NUM_GPUS GPU workers (vLLM + sidecar, TP=1)..."
    echo "[INFO]   Worker ports: ${BASE_PORT}-$((BASE_PORT + NUM_GPUS - 1))"
    echo "[INFO]   vLLM  ports: $((BASE_PORT + 100))-$((BASE_PORT + 100 + NUM_GPUS - 1))"
    # Hydra-style invocation with config overrides
    python -m recipe.phimm.vllm_server.launch_vllm_servers \
        model.path="$MODEL_PATH" \
        cluster.proxy_url="$PROXY_URL" \
        cluster.num_gpus="$NUM_GPUS" \
        cluster.base_port="$BASE_PORT" \
        server.gpu_memory_utilization="$GPU_MEM_UTIL" \
        server.max_model_len="$MAX_MODEL_LEN" \
        server.max_num_seqs="$MAX_NUM_SEQS" \
        server.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
        worker.audio_workers="$AUDIO_WORKERS" \
        server.disable_prefix_caching=true \
        ${HYDRA_OVERRIDES:-} &
    SERVERS_PID=$!
    echo "[INFO] Server launcher PID: $SERVERS_PID"
}

wait_for_backends() {
    local min_backends="${1:-$NUM_GPUS}"
    echo "[INFO] Waiting for $min_backends backends via proxy /admin/wait ..."
    # Use the proxy's built-in wait endpoint (blocks until enough backends are healthy)
    local result
    result=$(curl -sf "${PROXY_URL}/admin/wait?min_backends=${min_backends}&timeout=600" 2>/dev/null || echo "")
    if [[ -n "$result" ]]; then
        echo "[INFO] $result"
    else
        echo "[WARN] /admin/wait timed out, falling back to polling..."
        for i in $(seq 1 120); do
            BACKENDS=$(curl -sf "${PROXY_URL}/admin/backends" 2>/dev/null || echo "[]")
            COUNT=$(echo "$BACKENDS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(sum(1 for b in d if b.get('healthy',False)))" 2>/dev/null || echo 0)
            if [[ "$COUNT" -ge "$min_backends" ]]; then
                echo "[INFO] $COUNT/$min_backends healthy backends ready"
                return 0
            fi
            echo "[INFO] $COUNT/$min_backends healthy backends, waiting..."
            sleep 5
        done
    fi
}

warmup_backends() {
    echo "[INFO] Warming up backends with test requests..."
    # Send one request per backend to trigger CUDA graph compilation
    local backends
    backends=$(curl -sf "${PROXY_URL}/admin/backends" 2>/dev/null || echo "[]")
    local count
    count=$(echo "$backends" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d))" 2>/dev/null || echo 0)
    echo "[INFO] Sending $count warmup requests (one per GPU, may take 60-120s for CUDA graph capture)..."
    python3 -c "
import httpx, time, concurrent.futures

proxy_url = '${PROXY_URL}'
# Simple warmup: send one short request per backend
payload = {
    'audio_path': 'az://orngwus2cresco/data/boren/data/LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac',
    'prompt': 'Transcribe.',
    'max_tokens': 10,
    'max_audio_dur': 5.0,
}

def send_one(i):
    try:
        r = httpx.post(f'{proxy_url}/asr/transcribe', json=payload, timeout=300)
        return f'  Warmup {i}: status={r.status_code}'
    except Exception as e:
        return f'  Warmup {i}: error={e}'

start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=${count}) as pool:
    futs = [pool.submit(send_one, i) for i in range(${count})]
    for f in concurrent.futures.as_completed(futs):
        print(f.result())
print(f'[INFO] Warmup complete in {time.time()-start:.1f}s')
"
}

# ---- Evaluation ----
run_eval() {
    echo "[INFO] Starting ASR evaluation..."
    EVAL_ARGS=(
        --proxy-url "$PROXY_URL"
        --model "$MODEL_PATH"
        --data-tsv "$DATA_TSV"
        --max-concurrent "$MAX_CONCURRENT"
        --max-tokens "$MAX_TOKENS"
    )
    if [[ -n "$OUTPUT_PATH" ]]; then
        EVAL_ARGS+=(--output-path "$OUTPUT_PATH")
    fi
    if [[ -n "$NUM_EGS" ]]; then
        EVAL_ARGS+=(--num-egs "$NUM_EGS")
    fi

    python -m recipe.phimm.vllm_server.eval_asr "${EVAL_ARGS[@]}"
}

# ---- Cleanup ----
cleanup() {
    echo "[INFO] Cleaning up..."
    [[ -n "${PROXY_PID:-}" ]] && kill "$PROXY_PID" 2>/dev/null || true
    [[ -n "${SERVERS_PID:-}" ]] && kill "$SERVERS_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

# ---- Main ----
case "$ROLE" in
    all)
        start_proxy
        start_servers
        wait_for_backends "$NUM_GPUS"
        warmup_backends
        run_eval
        ;;
    proxy)
        start_proxy
        echo "[INFO] Proxy running. Press Ctrl+C to stop."
        wait "$PROXY_PID"
        ;;
    worker)
        start_servers
        wait_for_backends 1  # wait for at least this node's servers
        echo "[INFO] Servers running. New servers can be added dynamically."
        echo "[INFO] Register more: curl -X POST ${PROXY_URL}/admin/register -d '{\"url\":\"http://host:port\"}'"
        wait "$SERVERS_PID"
        ;;
    eval-only)
        run_eval
        ;;
    *)
        echo "[ERROR] Unknown role: $ROLE (use: all, proxy, worker, eval-only)"
        exit 1
        ;;
esac

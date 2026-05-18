#!/usr/bin/env bash
# =============================================================================
# Multi-node vLLM Server Evaluation Runner
#
# Per node: FastAPI proxy (1) + vLLM servers + worker sidecars (one per GPU,
# TP=1). Client sends {"audio_path": "az://..."} to the proxy, which routes
# to a worker that loads the audio locally and forwards to its vLLM server.
#
# Configuration:
#   Launcher → recipe/phimm/vllm_server/config.yaml
#   Eval     → recipe/phimm/vllm_server/eval_config.yaml
#   Override any field with Hydra-style key=value tokens after `--`.
#
# Usage:
#   bash run_eval.sh --role all                    # proxy + servers + eval
#   bash run_eval.sh --role worker                 # workers only (remote proxy)
#   bash run_eval.sh --role eval-only              # client only
#   bash run_eval.sh --role all -- server.max_num_seqs=64 data.num_egs=100
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ---- Args ----
ROLE="${ROLE:-all}"                                   # all | proxy | worker | eval-only
CONFIG_NAME="${CONFIG_NAME:-config}"                  # launcher Hydra config
EVAL_CONFIG_NAME="${EVAL_CONFIG_NAME:-eval_config}"   # eval Hydra config
OVERRIDES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --role)         ROLE="$2"; shift 2;;
        --config)       CONFIG_NAME="$2"; shift 2;;
        --eval-config)  EVAL_CONFIG_NAME="$2"; shift 2;;
        --)             shift; OVERRIDES+=("$@"); break;;
        -h|--help)      sed -n '2,18p' "$0"; exit 0;;
        *=*)            OVERRIDES+=("$1"); shift;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done
# Allow extra overrides via env var as well.
# shellcheck disable=SC2206
[[ -n "${HYDRA_OVERRIDES:-}" ]] && OVERRIDES+=($HYDRA_OVERRIDES)

# ---- Pull only what the shell needs from the launcher config ----
read -r PROXY_HOST PROXY_PORT NUM_GPUS BASE_PORT PROXY_URL_CFG < <(
    CONFIG="$SCRIPT_DIR/$CONFIG_NAME.yaml" \
    OVERRIDES_JSON=$(printf '%s\n' "${OVERRIDES[@]:-}" | python3 -c \
        'import json,sys; print(json.dumps([l for l in sys.stdin.read().splitlines() if l]))') \
    python3 - <<'PY'
import json, os
from omegaconf import OmegaConf
cfg = OmegaConf.load(os.environ["CONFIG"])
ov = json.loads(os.environ["OVERRIDES_JSON"])
if ov:
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(ov))
print(cfg.proxy.host, cfg.proxy.port, cfg.cluster.num_gpus,
      cfg.cluster.base_port, cfg.cluster.get("proxy_url") or "")
PY
)

# Resolve proxy URL (cluster.proxy_url wins; else derive from host/port).
if [[ -n "$PROXY_URL_CFG" ]]; then
    PROXY_URL="$PROXY_URL_CFG"
elif [[ "$PROXY_HOST" == "0.0.0.0" ]]; then
    PROXY_URL="http://$(hostname -I | awk '{print $1}'):${PROXY_PORT}"
else
    PROXY_URL="http://${PROXY_HOST}:${PROXY_PORT}"
fi

cat <<EOF
============================================
vLLM Multi-Node Evaluation
============================================
Role:       $ROLE
Config:     $SCRIPT_DIR/$CONFIG_NAME.yaml
Eval cfg:   $SCRIPT_DIR/$EVAL_CONFIG_NAME.yaml
Overrides:  ${OVERRIDES[*]:-(none)}
Proxy URL:  $PROXY_URL
Num GPUs:   $NUM_GPUS  (worker ports ${BASE_PORT}-$((BASE_PORT+NUM_GPUS-1)), vLLM $((BASE_PORT+100))-$((BASE_PORT+100+NUM_GPUS-1)))
============================================
EOF

# ---- Components ----
start_proxy() {
    echo "[INFO] Starting FastAPI proxy on ${PROXY_HOST}:${PROXY_PORT}..."
    python -m recipe.phimm.vllm_server.fastapi_proxy --host "$PROXY_HOST" --port "$PROXY_PORT" &
    PROXY_PID=$!
    for _ in $(seq 1 30); do
        curl -sf "http://localhost:${PROXY_PORT}/health" >/dev/null 2>&1 \
            && { echo "[INFO] Proxy ready (PID $PROXY_PID)"; return; }
        sleep 1
    done
    echo "[ERROR] Proxy failed to start"; return 1
}

start_servers() {
    echo "[INFO] Launching $NUM_GPUS GPU workers..."
    python -m recipe.phimm.vllm_server.launch_vllm_servers \
        --config-path "$SCRIPT_DIR" --config-name "$CONFIG_NAME" \
        "cluster.proxy_url=$PROXY_URL" \
        "${OVERRIDES[@]}" &
    SERVERS_PID=$!
}

run_eval() {
    echo "[INFO] Starting ASR evaluation..."
    # Drop launcher-only keys (server.*, worker.*, cluster.*, proxy.*) so the
    # eval Hydra config doesn't reject them in struct mode.
    local eval_ov=()
    for ov in "${OVERRIDES[@]:-}"; do
        case "$ov" in
            server.*|worker.*|cluster.*|proxy.*) ;;
            *) eval_ov+=("$ov");;
        esac
    done
    python -m recipe.phimm.vllm_server.eval_asr \
        --config-path "$SCRIPT_DIR" --config-name "$EVAL_CONFIG_NAME" \
        "eval.proxy_url=$PROXY_URL" \
        "${eval_ov[@]}"
}

cleanup() {
    [[ -n "${PROXY_PID:-}"   ]] && kill "$PROXY_PID"   2>/dev/null || true
    [[ -n "${SERVERS_PID:-}" ]] && kill "$SERVERS_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

# ---- Main ----
case "$ROLE" in
    all)
        start_proxy
        start_servers
        run_eval
        ;;
    proxy)
        start_proxy
        echo "[INFO] Proxy running. Ctrl+C to stop."
        wait "$PROXY_PID"
        ;;
    worker)
        start_servers
        echo "[INFO] Servers running. Register more: curl -X POST ${PROXY_URL}/admin/register -d '{\"url\":\"http://host:port\"}'"
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

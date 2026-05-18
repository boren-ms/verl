#!/usr/bin/env bash
# Convenience wrapper for the vLLM ASR serving stack.
#
# Roles:
#   eval   (default) — run eval_asr.py. It auto-launches the proxy + vLLM
#                       workers locally if the proxy isn't already reachable.
#   worker          — launch per-GPU vLLM workers (no proxy). Use this on a
#                       remote serving node; the eval client's local proxy
#                       talks to these workers via cluster.proxy_url.
#
# Usage:
#   bash run_eval.sh                                # role=eval
#   bash run_eval.sh --role worker                  # serve only
#   bash run_eval.sh data.num_egs=100               # role=eval + Hydra overrides
#   VLLM_CONFIG=vllm EVAL_CONFIG=eval bash run_eval.sh
#
# Examples:
#
#   # 1) Single-node: eval + auto-launched local proxy/workers (default).
#   bash run_eval.sh
#
#   # 2) Single-node with a different dataset / model checkpoint.
#   bash run_eval.sh \
#       data.source_config=data/train_data/openasr_en.yaml \
#       data.num_egs=1000 \
#       eval.launcher.vllm_config=vllm \
#       model.local_path=/tmp/vllm_models/my-ckpt
#
#   # 3) Multi-node — start workers on each serving node, pointing at a
#   #    remote proxy URL (cluster.proxy_url is required so the workers
#   #    register with that proxy via POST /admin/register).
#   #
#   #    On each worker node:
#   bash run_eval.sh --role worker \
#       cluster.proxy_url=http://<eval-node-ip>:8000 \
#       cluster.num_gpus=8
#
#   #    On the eval node, disable local auto-launch and point at the
#   #    proxy that the remote workers registered with:
#   bash run_eval.sh \
#       eval.proxy_url=http://<eval-node-ip>:8000 \
#       eval.launcher.enabled=false
#
#   # 4) Run a stand-alone proxy (no workers) — e.g. on a head node that
#   #    several remote worker nodes will register with:
#   python -m recipe.phimm.vllm_server.fastapi_proxy --host 0.0.0.0 --port 8000
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/config"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

ROLE="${ROLE:-eval}"
VLLM_CONFIG="${VLLM_CONFIG:-vllm}"
EVAL_CONFIG="${EVAL_CONFIG:-eval}"

ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --role)         ROLE="$2"; shift 2;;
        --vllm-config)  VLLM_CONFIG="$2"; shift 2;;
        --eval-config)  EVAL_CONFIG="$2"; shift 2;;
        -h|--help)      sed -n '2,46p' "$0"; exit 0;;
        *)              ARGS+=("$1"); shift;;
    esac
done

case "$ROLE" in
    eval)
        exec python -m recipe.phimm.vllm_server.eval_asr \
            --config-path "$CONFIG_DIR" --config-name "$EVAL_CONFIG" "${ARGS[@]}"
        ;;
    worker)
        exec python -m recipe.phimm.vllm_server.launch_vllm_servers \
            --config-path "$CONFIG_DIR" --config-name "$VLLM_CONFIG" "${ARGS[@]}"
        ;;
    *)
        echo "[ERROR] Unknown role: $ROLE (use: eval, worker)"; exit 1;;
esac

#!/bin/bash
set -xeuo pipefail

env_tag="torch2.10.0_vllm0.17.0_flashattn2.8.3_v1"
done_file=".env_done_${env_tag}"
running_file=".env_running_${env_tag}"

echo "[INFO] Checking environment installation status..."

while [ -f "${running_file}" ]; do
    echo "[INFO] Waiting for the installation to complete..."
    sleep 10
done

if [ -f "${done_file}" ] && ! python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

required_versions = {
    "torch": "2.10.0",
    "vllm": "0.17.0",
    "protobuf": "5.29.5",
}

for package, expected in required_versions.items():
    try:
        installed = version(package)
    except PackageNotFoundError:
        raise SystemExit(1)
    if installed != expected:
        raise SystemExit(1)

for package in ("tensordict", "TransferQueue"):
    try:
        version(package)
    except PackageNotFoundError:
        raise SystemExit(1)
PY
then
    echo "[WARN] Environment marker is stale; reinstalling."
    rm -f "${done_file}"
fi

if [ ! -f "${done_file}" ]; then
    touch "${running_file}"
    trap 'rm -f "${running_file}"' ERR
    echo "[INFO] Installing environment..."

    # IMPORTANT: The base image ships Ray 2.46.0+ with a running `ray start
    # --head` process.  We must NEVER change the Ray version because:
    #   - The Ray dashboard agent and pre-spawned workers keep loaded modules
    #     in memory.  A pip-level downgrade/upgrade causes import mismatches
    #     that cannot be fixed without a full pod restart.
    #
    # We also must NOT upgrade protobuf beyond the base-image version (5.29.5)
    # because pre-spawned Ray workers have protobuf 5.29.5 loaded in memory.
    # Installing vllm WITH deps pulls newer protobuf + opentelemetry-proto
    # gencode, which crashes workers with a gencode/runtime mismatch.
    #
    # Strategy:
    #   1. Install vllm --no-deps  (avoids protobuf/ray/opentelemetry upgrades)
    #   2. Install all deps from requirements_vllm.txt (pinned versions)
    #   3. Skip ray install entirely — use whatever the base image provides
    #   4. Install flash_attn from pre-built wheel on blob

    # 1. vllm without deps — avoids protobuf & ray conflicts
    pip install --no-deps vllm==0.17.0

    # 2. All project + vllm inference deps (pinned versions from working env)
    pip install -r requirements_vllm.txt "protobuf==5.29.5"

    # Ray: DO NOT install — keep the base image version
    # pip install --no-deps "ray[default]==2.46.0"  # REMOVED

    # 3. Install this repo in editable mode (no deps, they're in requirements_vllm.txt)
    pip install --no-deps -e .

    # 3a. TransferQueue is sourced from GitHub, which Brix nodes cannot access directly.
    transfer_queue_pkg="transferqueue-0.1.11.dev0-py3-none-any.whl"
    package_dir="/root/packages"
    transfer_queue_path="${package_dir}/${transfer_queue_pkg}"
    mkdir -p "${package_dir}"
    if [ ! -f "${transfer_queue_path}" ]; then
        command -v bbb >/dev/null || {
            echo "[ERROR] bbb is required to download ${transfer_queue_pkg}" >&2
            exit 1
        }
        bbb cp \
            "az://orngwus2cresco/data/boren/data/packages/${transfer_queue_pkg}" \
            "${transfer_queue_path}"
    fi
    pip install --no-deps "${transfer_queue_path}"

    # 3b. Install Qwen3.5-Audio vLLM plugin (out-of-tree model support)
    pip install --no-deps -e plugins/qwen35_audio

    # 4. Flash attention from pre-built wheel
    flash_attn_pkg="flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl"
    remote_pkg_path="az://orngwus2cresco/data/boren/data/packages/${flash_attn_pkg}"
    local_pkg_path="${package_dir}/${flash_attn_pkg}"
    if [ ! -f "${local_pkg_path}" ]; then
        command -v bbb >/dev/null || {
            echo "[ERROR] bbb is required to download ${remote_pkg_path}" >&2
            exit 1
        }
        bbb cp "${remote_pkg_path}" "${local_pkg_path}"
    fi
    pip install --no-deps "${local_pkg_path}"

    # 5. Install lsof for GPU cleanup helpers
    if ! command -v lsof >/dev/null; then
        apt install -y lsof || echo "[WARN] Could not install lsof; GPU cleanup helpers may be unavailable." >&2
    fi

    mv "${running_file}" "${done_file}"
else
    echo "[INFO] Environment already installed, skipping installation."
fi

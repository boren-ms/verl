#!/bin/bash
set -xeuo pipefail

env_tag="torch2.10.0_vllm0.17.0_flashattn2.8.3_v2"
done_file=".env_done_${env_tag}"
running_file=".env_running_${env_tag}"

echo "[INFO] Checking environment installation status..."

while [ -f "${running_file}" ]; do
    echo "[INFO] Waiting for the installation to complete..."
    sleep 10
done

if [ ! -f "${done_file}" ]; then
    touch "${running_file}"
    trap 'rm -f "${running_file}"' ERR
    echo "[INFO] Installing environment..."

    # IMPORTANT: The base image ships Ray 2.54.0 with a running `ray start
    # --head` process.  We must NEVER change the Ray version because:
    #   - The Ray dashboard agent and pre-spawned workers keep loaded modules
    #     in memory.  A pip-level downgrade/upgrade causes import mismatches
    #     (e.g. _get_uv_run_cmdline ImportError) that cannot be fixed without
    #     a full pod restart.
    #
    # We also must NOT upgrade protobuf beyond the base-image version (5.29.5)
    # because pre-spawned Ray workers have protobuf 5.29.5 loaded in memory.
    # Installing vllm WITH deps pulls protobuf 5.29.6 + opentelemetry-proto
    # gencode 5.29.6, which crashes workers with a gencode/runtime mismatch.
    #
    # Strategy:
    #   1. Install vllm --no-deps  (avoids protobuf/ray/opentelemetry upgrades)
    #   2. Install all deps from requirements_vllm.txt (includes vllm inference
    #      deps, torch, flashinfer, and project deps — but NOT protobuf/ray)
    #   3. Skip ray install entirely — use whatever the base image provides

    # 1. vllm without deps — avoids protobuf & ray conflicts
    pip install --no-deps vllm==0.17.0

    # 2. All project + vllm inference deps (torch, flashinfer, vllm deps all in requirements_vllm.txt)
    pip install -r requirements_vllm.txt

    # Ray: DO NOT install — keep the base image version
    # pip install --no-deps "ray[default]==2.46.0"  # REMOVED

    pip install --no-deps -e .
    pip install --no-deps -e plugins/qwen35_audio

    flash_attn_pkg="flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl"
    remote_pkg_path="az://orngwus2cresco/data/boren/data/packages/${flash_attn_pkg}"
    local_pkg_dir="/root/packages"
    local_pkg_path="${local_pkg_dir}/${flash_attn_pkg}"
    mkdir -p "${local_pkg_dir}"
    if [ ! -f "${local_pkg_path}" ]; then
        command -v bbb >/dev/null || {
            echo "[ERROR] bbb is required to download ${remote_pkg_path}" >&2
            exit 1
        }
        bbb cp "${remote_pkg_path}" "${local_pkg_path}"
    fi
    pip install --no-deps "${local_pkg_path}"
    if ! command -v lsof >/dev/null; then
        apt install -y lsof || echo "[WARN] Could not install lsof; GPU cleanup helpers may be unavailable." >&2
    fi
    mv "${running_file}" "${done_file}"
else
    echo "[INFO] Environment already installed, skipping installation."
fi

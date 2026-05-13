#!/bin/bash
set -xeuo pipefail

env_tag="torch2.10.0_ray2.46.0_transformers5.7.0_vllm0.17.0_flashattn2.8.3"
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
    # Install vllm with --no-deps so it does NOT drag in its own torch/ray.
    pip install --no-deps vllm==0.17.0
    # Install torch + flashinfer first to establish the correct ABI.
    pip install torch==2.10.0 flashinfer-python==0.6.4 flashinfer-cubin==0.6.4
    # Install remaining deps (vllm/torch/ray already handled above).
    pip install -r requirements_vllm.txt
    pip install --no-deps "ray[default]==2.46.0"
    pip install --no-deps -e .
    pip install --no-deps -e plugins/qwen35_audio
    flash_attn_pkg="flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
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

#!/bin/bash
set -xeuo pipefail

env_tag="torch2.8.0_ray2.46.0_transformers5.7.0_vllm0.17.0_flashinfer0.6.4"
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
    # Uninstall flash-attn if present (incompatible with flashinfer stack)
    pip show flash-attn >/dev/null 2>&1 && pip uninstall -y flash-attn || true
    pip install -r requirements_vllm.txt
    pip install --no-deps \
        transformers==5.7.0 \
        huggingface-hub==1.13.0 \
        tokenizers==0.22.2 \
        regex==2026.4.4 \
        packaging==26.0 \
        tqdm==4.67.3 \
        typer
    pip install --no-deps "ray[default]==2.46.0"
    pip install --no-deps -e .
    pip install --no-deps -e plugins/qwen35_audio
    if ! command -v lsof >/dev/null; then
        apt install -y lsof || echo "[WARN] Could not install lsof; GPU cleanup helpers may be unavailable." >&2
    fi
    mv "${running_file}" "${done_file}"
else
    echo "[INFO] Environment already installed, skipping installation."
fi

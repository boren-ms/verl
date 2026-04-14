#!/bin/bash
# Update torch, flash-attn, and vllm on the remote pyenv Python
# This installs packages that are ABI-compatible with each other.
set -euo pipefail

PIP=/root/.pyenv/versions/3.12.9/bin/pip
PYTHON=/root/.pyenv/versions/3.12.9/bin/python3
BLOB_ROOT="az://orngwus2cresco/data/boren/data/verl/wheels"
LOCAL_DIR="/tmp/verl_wheels"

echo "=== Step 1: Download wheels from blob ==="
mkdir -p "$LOCAL_DIR"
cd "$LOCAL_DIR"

# Download all wheels from blob
for whl in \
    "torch-2.8.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl" \
    "torchvision-0.23.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl" \
    "torchaudio-2.8.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl" \
    "flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
    "vllm-0.11.0-cp38-abi3-manylinux1_x86_64.whl"; do
    if [ ! -f "$whl" ]; then
        echo "Downloading $whl ..."
        az storage blob download \
            --account-name orngwus2cresco \
            --container-name data \
            --name "boren/data/verl/wheels/$whl" \
            --file "$whl" \
            --auth-mode login \
            --no-progress \
            --only-show-errors 2>/dev/null || {
            echo "ERROR: Failed to download $whl"
            exit 1
        }
    else
        echo "Already have $whl"
    fi
done

echo "=== Step 2: Install torch, torchvision, torchaudio ==="
$PIP install --no-deps --force-reinstall \
    "${LOCAL_DIR}/torch-2.8.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl" \
    "${LOCAL_DIR}/torchvision-0.23.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl" \
    "${LOCAL_DIR}/torchaudio-2.8.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl"

echo "=== Step 3: Install flash-attn ==="
$PIP install --no-deps --force-reinstall \
    "${LOCAL_DIR}/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

echo "=== Step 4: Install vllm (--no-deps to skip ray>=2.48 requirement) ==="
$PIP install --no-deps --force-reinstall \
    "${LOCAL_DIR}/vllm-0.11.0-cp38-abi3-manylinux1_x86_64.whl"

echo "=== Step 5: Install/upgrade missing vllm deps from PyPI ==="
$PIP install --upgrade \
    "transformers>=4.55.2" \
    "tokenizers>=0.21.1" \
    "xgrammar==0.1.25" \
    "openai-harmony>=0.0.3" \
    "cbor2" \
    "msgspec" \
    "partial-json-parser" \
    "watchfiles" \
    "python-json-logger" \
    "prometheus-fastapi-instrumentator>=7.0.0"

echo "=== Step 6: Verify installations ==="
$PYTHON -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')

import flash_attn
print(f'flash_attn: {flash_attn.__version__}')

import vllm
print(f'vllm: {vllm.__version__}')

import transformers
print(f'transformers: {transformers.__version__}')

import ray
print(f'ray: {ray.__version__}')
"

echo "=== Done! ==="

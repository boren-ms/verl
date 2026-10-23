#!/bin/bash
# Run after quick_install.sh has installed the supported Torch/CUDA stack.
set -euo pipefail

export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:/root/.pyenv/versions/3.12.9/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

check_linear_attention() {
    python3 - <<'PY'
from importlib.metadata import version

from packaging.version import Version

for package, expected in {
    "flash-linear-attention": "0.6.0",
    "fla-core": "0.6.0",
    "causal-conv1d": "1.7.0",
    "tilelang": "0.1.9",
    "apache-tvm-ffi": "0.1.9",
    "ml-dtypes": "0.5.4",
}.items():
    installed = version(package)
    if Version(installed) != Version(expected):
        raise RuntimeError(f"{package}: expected {expected}, got {installed}")

import os
if os.environ.get("VERL_LINEAR_INSTALL_ONLY") == "1":
    # CUDA execution is verified explicitly on the fresh GPU node.
    import torch  # Load libtorch before importing the compiled extension.
    import causal_conv1d_cuda
    raise SystemExit(0)

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as native

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable for Qwen3.5 linear attention")
if not native.is_fast_path_available:
    raise RuntimeError("Qwen3.5 fast linear-attention kernels are unavailable")
if not native.chunk_gated_delta_rule.__module__.startswith("fla."):
    raise RuntimeError("Qwen3.5 selected the recurrent fallback instead of FLA")

# Exercise the compiled convolution extension, catching Torch/CUDA ABI failures.
from causal_conv1d import causal_conv1d_fn

x = torch.randn(1, 8, 8, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(8, 4, device="cuda", dtype=torch.bfloat16)
result = causal_conv1d_fn(x, weight, activation="silu")
torch.cuda.synchronize()
if not torch.isfinite(result).all():
    raise RuntimeError("causal-conv1d CUDA smoke check produced nonfinite output")
# FLA 0.6 rejects gated backward on Hopper with the base Triton 3.6.
# TileLang supplies its supported backend; test backward to catch this class
# of failure before a training job starts.
q = torch.randn(1, 16, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
g = torch.full((1, 16, 2), -0.1, device="cuda", dtype=torch.float32)
beta = torch.full((1, 16, 2), 0.5, device="cuda", dtype=torch.bfloat16)
output, _ = native.chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
output.float().square().mean().backward()
torch.cuda.synchronize()
if not torch.isfinite(output).all() or any(
    t.grad is None or not torch.isfinite(t.grad).all() for t in (q, k, v)
):
    raise RuntimeError("FLA forward/backward smoke check produced nonfinite output")
print("[INFO] Qwen3.5 fast linear-attention kernels verified.")
PY
}

if [[ "${1:-}" == "--install-only" ]]; then
    export VERL_LINEAR_INSTALL_ONLY=1
elif [[ -n "${1:-}" && "${1:-}" != "--check" ]]; then
    echo "Usage: $0 [--check|--install-only]" >&2
    exit 2
fi
if [[ "${1:-}" == "--check" ]]; then
    check_linear_attention
    exit 0
fi

if check_linear_attention; then
    exit 0
fi

echo "[INFO] Installing Qwen3.5 linear-attention kernels..."
# Avoid pulling a different Torch, Transformers, Ray, or protobuf version.
python3 -m pip install --no-deps \
    "fla-core==0.6.0" "flash-linear-attention==0.6.0" \
    "tilelang==0.1.9" "apache-tvm-ffi==0.1.9" "ml-dtypes==0.5.4"
# Installing FLA alone may have repaired the environment.
if check_linear_attention; then
    exit 0
fi

# Wheel filenames omit the Torch ABI. Keep wheels under an explicit stack key,
# including compiler/runtime versions, rather than trusting pip's wheel cache.
stack_key=$(python3 - <<'PYKEY'
import hashlib
import json
import platform
import sys
from importlib.metadata import version
import torch

stack = {
    "python": sys.implementation.cache_tag,
    "machine": platform.machine(),
    "libc": platform.libc_ver(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
    "cuda_nvcc": version("nvidia-cuda-nvcc"),
    "cuda_runtime": version("nvidia-cuda-runtime"),
    "causal_conv1d": "1.7.0",
}
print(hashlib.sha256(json.dumps(stack, sort_keys=True).encode()).hexdigest()[:24])
PYKEY
)
cache_dir="/root/packages/causal-conv1d/1.7.0/${stack_key}"
wheel_name="causal_conv1d-1.7.0-cp312-cp312-linux_x86_64.whl"
wheel_path="${cache_dir}/${wheel_name}"
wheel_blob="az://orngwus2cresco/data/boren/data/packages/causal-conv1d/1.7.0/${stack_key}/${wheel_name}"
mkdir -p "${cache_dir}"

if [[ ! -f "${wheel_path}" ]] && command -v bbb >/dev/null; then
    echo "[INFO] Fetching compiled causal-conv1d wheel for stack ${stack_key}..."
    if bbb cp "${wheel_blob}" "${wheel_path}.download"; then
        mv "${wheel_path}.download" "${wheel_path}"
    else
        rm -f "${wheel_path}.download"
        echo "[INFO] No shared wheel available; building it once for this stack."
    fi
fi

if [[ -f "${wheel_path}" ]]; then
    if python3 -m pip install --force-reinstall --no-deps "${wheel_path}" && check_linear_attention; then
        echo "[INFO] Reused compiled causal-conv1d wheel for stack ${stack_key}."
        exit 0
    fi
    echo "[WARN] Cached wheel failed validation; rebuilding."
    rm -f "${wheel_path}"
fi

# Build once, retain the wheel, and share only after the real CUDA check passes.
build_dir=$(mktemp -d "${cache_dir}/build.XXXXXX")
trap 'rm -rf "${build_dir}"' EXIT
MAX_JOBS="${MAX_JOBS:-8}" CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    python3 -m pip wheel --no-cache-dir --wheel-dir "${build_dir}" \
    --no-binary=causal-conv1d --no-build-isolation --no-deps \
    "causal-conv1d==1.7.0"
python3 -m pip install --force-reinstall --no-deps "${build_dir}/${wheel_name}"
check_linear_attention
mv "${build_dir}/${wheel_name}" "${wheel_path}"
if command -v bbb >/dev/null; then
    bbb cp "${wheel_path}" "${wheel_blob}" || \
        echo "[WARN] Could not publish compiled wheel; the local cache remains available." >&2
fi

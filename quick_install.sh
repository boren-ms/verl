#!/bin/bash
set -xeuo pipefail

env_tag="torch2.11.0_vllm0.24.0_flashattn2.8.3_cu130_v3"
done_file=".env_done_${env_tag}"
running_file=".env_running_${env_tag}"

echo "[INFO] Checking environment installation status..."

while [ -f "${running_file}" ]; do
    echo "[INFO] Waiting for the installation to complete..."
    sleep 10
done

if [ -f "${done_file}" ] && ! python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.version import Version

required_versions = {
    "torch": "2.11.0",
    "vllm": "0.24.0",
    "transformers": "5.7.0",
    "flashinfer-python": "0.6.12",
    "flashinfer-cubin": "0.6.12",
    "protobuf": "5.29.5",
}

for package, expected in required_versions.items():
    try:
        installed = Version(version(package))
    except PackageNotFoundError:
        raise SystemExit(1)
    expected = Version(expected)
    if package == "torch":
        matches = installed.base_version == expected.base_version
    else:
        matches = installed == expected
    if not matches:
        raise SystemExit(1)

try:
    tensordict_version = Version(version("tensordict"))
except PackageNotFoundError:
    raise SystemExit(1)
if tensordict_version != Version("0.10.0"):
    raise SystemExit(1)

for package in ("TransferQueue",):
    try:
        version(package)
    except PackageNotFoundError:
        raise SystemExit(1)

cuda_compat_dir = Path("/usr/local/cuda-13.0/compat")
cuda_compat_conf = Path("/etc/ld.so.conf.d/00-oai-cuda-compat.conf")
if (
    not (cuda_compat_dir / "libcuda.so.1").exists()
    or not cuda_compat_conf.exists()
    or cuda_compat_conf.read_text().strip() != str(cuda_compat_dir)
):
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

    # IMPORTANT: The base image ships Ray with a running `ray start
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
    #   2. Install CUDA 13 forward compatibility for the host driver
    #   3. Install the PyTorch/CUDA stack from pre-staged internal wheels
    #   4. Install all remaining deps from requirements_vllm.txt
    #   5. Skip ray install entirely — use whatever the base image provides
    #   6. Install the pre-staged cu130 / torch-2.11 flash_attn wheel

    # 1. vllm without deps — avoids protobuf & ray conflicts
    pip install --no-deps vllm==0.24.0

    package_dir="/root/packages"
    mkdir -p "${package_dir}"

    # The Brix image uses a 550-series driver. CUDA 13 PyTorch requires the
    # forward-compatibility driver library, otherwise CUDA reports driver 12.9.
    cuda_compat_pkg="cuda-compat-13-0-580.95.05.tgz"
    cuda_compat_path="${package_dir}/${cuda_compat_pkg}"
    if [ ! -f "${cuda_compat_path}" ]; then
        command -v bbb >/dev/null || {
            echo "[ERROR] bbb is required to download ${cuda_compat_pkg}" >&2
            exit 1
        }
        bbb cp \
            "az://orngwus2cresco/data/boren/data/packages/${cuda_compat_pkg}" \
            "${cuda_compat_path}"
    fi
    tar -C / -xzf "${cuda_compat_path}"
    printf '%s\n' "/usr/local/cuda-13.0/compat" \
        > /etc/ld.so.conf.d/00-oai-cuda-compat.conf
    ldconfig

    torch_wheel_dir="${package_dir}/torch211-cu130"
    torch_wheel_blob="az://orngwus2cresco/data/boren/data/packages/torch211-cu130/"
    mkdir -p "${torch_wheel_dir}"
    command -v bbb >/dev/null || {
        echo "[ERROR] bbb is required to download ${torch_wheel_blob}" >&2
        exit 1
    }
    # Always sync: it is incremental and ensures newly added dependency wheels
    # are fetched even when an older, incomplete local bundle is present.
    bbb sync "${torch_wheel_blob}" "${torch_wheel_dir}/"

    # Install from local files only. cuda-toolkit's extras pull in the complete
    # CUDA 13 runtime set required by the torch 2.11 wheel.
    pip install --no-index --find-links "${torch_wheel_dir}" \
        "cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink]==13.0.2" \
        "cuda-bindings>=13.0.3,<14" \
        "torch==2.11.0+cu130" \
        "torchvision==0.26.0+cu130" \
        "torchaudio==2.11.0+cu130"

    # 3. Remaining project + inference dependencies come from the default
    # package index; the PyTorch requirements are already satisfied locally.
    pip install -r requirements_vllm.txt "protobuf==5.29.5"

    # Ray: DO NOT install — keep the base image version
    # Do not add a Ray install here: use the exact version from the base image.

    # 4. Install this repo in editable mode (no deps, they're in requirements_vllm.txt)
    pip install --no-deps -e .

    # 4a. TransferQueue is sourced from GitHub, which Brix nodes cannot access directly.
    transfer_queue_pkg="transferqueue-0.1.11.dev0-py3-none-any.whl"
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

    # 4b. Install Qwen3.5-Audio vLLM plugin (out-of-tree model support)
    pip install --no-deps -e plugins/qwen35_audio

    # 5. FlashAttention must match the new PyTorch 2.11 / CUDA 13.0 ABI.
    # Remote nodes cannot reach the public wheelhouse, so fetch the wheel from
    # internal blob storage after it has been prepared and uploaded locally.
    flash_attn_pkg="flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
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

    # 6. Install lsof for GPU cleanup helpers
    if ! command -v lsof >/dev/null; then
        apt install -y lsof || echo "[WARN] Could not install lsof; GPU cleanup helpers may be unavailable." >&2
    fi

    mv "${running_file}" "${done_file}"
else
    echo "[INFO] Environment already installed, skipping installation."
fi

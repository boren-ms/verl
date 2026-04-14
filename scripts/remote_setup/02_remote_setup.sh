#!/usr/bin/env bash
# ============================================================================
# 02_remote_setup.sh
# Run this ON THE REMOTE NODE (verl-n1-i1) to set up the Python environment,
# download data/model from Azure Blob, and install verl + dependencies.
#
# Uses the system pyenv Python 3.12.9 directly. Packages (torch 2.8.0,
# flash-attn 2.8.3, vllm 0.11.0) are installed from blob wheels via
# 04_update_packages.sh. This script installs verl + remaining deps and
# downloads data/models.
#
# The pod's entrypoint runs Ray on port 6380 with --block; we connect to it
# via RAY_ADDRESS. DO NOT stop it (kills the pod).
#
# This script is IDEMPOTENT - safe to re-run after pod restarts.
#
# Prerequisites (run from local machine first):
#   1. b sync verl-n1-i1              # sync code to remote
#   2. bash 01_local_prepare_and_upload.sh  # upload data/model/wheels to blob
#
# Usage (via rcall-brix tmux):
#   bash 02_remote_setup.sh
#
# Network constraints on remote:
#   - PyPI (pypi.org, files.pythonhosted.org): OK
#   - Azure blob (az storage blob download --auth-mode login): OK
#   - download.pytorch.org, github.com, huggingface.co: BLOCKED
#   - curl/wget HTTPS: BLOCKED (SSL interception)
#   - apt repositories: BLOCKED
# ============================================================================
set -xeuo pipefail

BLOB_ACCOUNT="orngwus2cresco"
BLOB_CONTAINER="data"
BLOB_PREFIX="boren/data/verl"
RAY_DATA_HOME="${HOME}/verl_data"
VERL_HOME="${HOME}/code/verl"
PIP="/root/.pyenv/versions/3.12.9/bin/pip"
PYTHON="/root/.pyenv/versions/3.12.9/bin/python3"
export PATH="/root/.pyenv/versions/3.12.9/bin:${PATH}"

# Helper: download a blob file using az cli
blob_download() {
    local blob_name="$1"
    local dest="$2"
    if [ -f "${dest}" ]; then
        echo "  [skip] ${dest} already exists"
        return 0
    fi
    mkdir -p "$(dirname "${dest}")"
    az storage blob download \
        --account-name "${BLOB_ACCOUNT}" \
        --container-name "${BLOB_CONTAINER}" \
        --name "${BLOB_PREFIX}/${blob_name}" \
        --file "${dest}" \
        --auth-mode login \
        --no-progress \
        --only-show-errors
}

# --------------------------------------------------------------------------
# 1. Install torch, flash-attn, vllm from blob wheels (04_update_packages.sh)
# --------------------------------------------------------------------------
if ! $PYTHON -c "import torch; assert torch.__version__.startswith('2.8')" 2>/dev/null; then
    echo ">>> Running 04_update_packages.sh to install torch/flash-attn/vllm..."
    bash "${VERL_HOME}/scripts/remote_setup/04_update_packages.sh"
fi
$PYTHON -c "import torch; print(f'torch={torch.__version__}, CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}')"

# --------------------------------------------------------------------------
# 2. Install verl + additional deps from PyPI
# --------------------------------------------------------------------------
if ! $PYTHON -c "import verl" 2>/dev/null; then
    echo ">>> Installing verl and dependencies..."
    cd "${VERL_HOME}"
    $PIP install --no-deps -e .
    $PIP install mbridge math_verify latex2sympy2_extended codetiming liger-kernel pybind11
fi

# --------------------------------------------------------------------------
# 3. Download data & model from Azure Blob
# --------------------------------------------------------------------------
echo ">>> Downloading data from blob..."
blob_download "data/dapo-math-17k.parquet" "${RAY_DATA_HOME}/data/dapo-math-17k.parquet"
blob_download "data/aime-2024.parquet" "${RAY_DATA_HOME}/data/aime-2024.parquet"

echo ">>> Downloading model from blob..."
for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         vocab.json merges.txt model.safetensors.index.json \
         model-00001-of-00004.safetensors model-00002-of-00004.safetensors \
         model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
    blob_download "models/Qwen2.5-Math-7B/${f}" "${RAY_DATA_HOME}/models/Qwen2.5-Math-7B/${f}"
done

# --------------------------------------------------------------------------
# 4. Verify
# --------------------------------------------------------------------------
echo ">>> Verifying..."
$PYTHON -c "
import torch, verl, ray, vllm, flash_attn
print(f'torch={torch.__version__}, CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}')
print(f'verl={verl.__version__}, vllm={vllm.__version__}, ray={ray.__version__}')
"
ls -lh "${RAY_DATA_HOME}/data/"
ls "${RAY_DATA_HOME}/models/Qwen2.5-Math-7B/" | head -5

echo "=== Setup complete. Run training with: ==="
echo "  bash ${VERL_HOME}/scripts/remote_setup/03_run_training.sh"

#!/usr/bin/env bash
# ============================================================================
# 01_local_prepare_and_upload.sh
# Run this LOCALLY to upload model, data, and tools to Azure Blob so the
# remote node (which cannot reach HF/GitHub/pytorch.org) can fetch them.
#
# Prerequisites:
#   - bbb CLI available
#   - Model at ~/data/ckp/hf_models/Qwen2.5-Math-7B/
#   - Data at ~/data/parquet/dapo-math-17k.parquet and aime-2024.parquet
#   - pip (for downloading pytorch wheels)
#
# If model/data don't exist locally, download them first:
#   pip install datasets huggingface_hub
#   python3 -c "import datasets; ds=datasets.load_dataset('BytedTsinghua-SIA/DAPO-Math-17k', split='train'); ds.to_parquet('~/data/parquet/dapo-math-17k.parquet')"
#   python3 -c "import datasets; ds=datasets.load_dataset('AI-MO/aimo-validation-aime', split='train'); ds.to_parquet('~/data/parquet/aime-2024.parquet')"
#   huggingface-cli download Qwen/Qwen2.5-Math-7B --local-dir ~/data/ckp/hf_models/Qwen2.5-Math-7B
# ============================================================================
set -xeuo pipefail

BLOB_BASE="az://orngwus2cresco/data/boren/data/verl"

# Local paths (adjust if your data is elsewhere)
LOCAL_DATA_DIR="${HOME}/data/parquet"
LOCAL_MODEL_DIR="${HOME}/data/ckp/hf_models/Qwen2.5-Math-7B"

# --------------------------------------------------------------------------
# 1. Upload training data
# --------------------------------------------------------------------------
echo ">>> Uploading data to blob..."
bbb cp "${LOCAL_DATA_DIR}/dapo-math-17k.parquet" "${BLOB_BASE}/data/dapo-math-17k.parquet"
bbb cp "${LOCAL_DATA_DIR}/aime-2024.parquet" "${BLOB_BASE}/data/aime-2024.parquet"

# --------------------------------------------------------------------------
# 2. Upload model
# --------------------------------------------------------------------------
echo ">>> Uploading model to blob..."
bbb cpr "${LOCAL_MODEL_DIR}" "${BLOB_BASE}/models/Qwen2.5-Math-7B"

# --------------------------------------------------------------------------
# 3. Upload get-pip.py (remote can't curl HTTPS)
# --------------------------------------------------------------------------
echo ">>> Uploading get-pip.py..."
wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py
bbb cp /tmp/get-pip.py "${BLOB_BASE}/tools/get-pip.py"

# --------------------------------------------------------------------------
# 4. Download & upload PyTorch cu128 wheels (download.pytorch.org blocked on remote)
# --------------------------------------------------------------------------
echo ">>> Downloading PyTorch cu128 wheels..."
mkdir -p /tmp/torch_wheels
pip download torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128 \
    --dest /tmp/torch_wheels \
    --python-version 3.12 \
    --only-binary=:all: \
    --platform manylinux_2_28_x86_64 \
    --no-deps

echo ">>> Uploading PyTorch wheels to blob..."
for f in /tmp/torch_wheels/*.whl; do
    bbb cp "$f" "${BLOB_BASE}/wheels/$(basename $f)"
done

# --------------------------------------------------------------------------
# 5. Verify uploads
# --------------------------------------------------------------------------
echo ">>> Verifying..."
bbb ls "${BLOB_BASE}/data/"
bbb ls "${BLOB_BASE}/models/Qwen2.5-Math-7B/" | head -5
bbb ls "${BLOB_BASE}/wheels/"
bbb ls "${BLOB_BASE}/tools/"

echo "=== Local preparation and upload complete ==="

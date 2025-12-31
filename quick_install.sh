#!/bin/bash
set -xeuo pipefail
done_file=".env_done"

running_file=".env_running"

echo "[INFO] Checking environment installation status..."

while [ -f ${running_file} ]; do
    echo "[INFO] Waiting for the installation to complete..."
    sleep 10
done

storage_account="orngwus2cresco"
rel_paths=(
    data/gsm8k
    data/ckp/hf_models/Qwen2.5-0.5B-Instruct
    data/packages
)

if [ ! -f ${done_file} ]; then
    touch ${running_file}

    echo "[INFO] Preparing data ..."    
    for rel_path in "${rel_paths[@]}"; do
        bbb sync az://${storage_account}/data/boren/"${rel_path}"/ /root/"${rel_path}"/
    done

    echo "[INFO] Installing environment..."
    pip install -r requirements_vllm.txt
    pip install --no-deps -e .
    pip install --no-build-isolation  /root/data/packages/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
    apt install lsof
    echo "Restart Ray server with new version"
    ray stop
    ray start --head
    
  
    mv ${running_file} ${done_file}
else
    echo "[INFO] Environment already installed, skipping installation."
fi

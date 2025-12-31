#!/bin/bash
set -xeuo pipefail
done_file=".data_done"

running_file=".data_running"

echo "[INFO] Checking data preparation status..."

while [ -f ${running_file} ]; do
    echo "[INFO] Waiting for the data preparation to complete..."
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
    
    mv ${running_file} ${done_file}
else
    echo "[INFO] Data already prepared, skipping data preparation."
fi

#!/bin/bash
set -xeuo pipefail
done_file=".env_done"

running_file=".env_running"

echo "[INFO] Checking environment installation status..."

while [ -f ${running_file} ]; do
    echo "[INFO] Waiting for the installation to complete..."
    sleep 10
done

if [ ! -f ${done_file} ]; then
    touch ${running_file}
    echo "[INFO] Installing environment..."
    pip install -r requirements_vllm.txt
    pip install --no-deps -e .
    apt install lsof
    echo "Restart Ray server with new version"
    ray stop
    ray start --head
    mv ${running_file} ${done_file}
else
    echo "[INFO] Environment already installed, skipping installation."
fi

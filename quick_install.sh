#!/bin/bash
set -xeuo pipefail
done_file=".env_done"

if [ ! -f ${done_file} ]; then
    pip install -r requirements_vllm.txt
    pip install --no-deps -e .
    pip install  flash-attn==2.7.4.post1
    apt install lsof
    touch ${done_file}
else
    echo "[INFO] ${done_file} already exists — skipping."
fi

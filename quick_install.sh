#!/bin/bash
set -xeuo pipefail

pip install -r requirements_vllm.txt
pip install --no-deps -e .
pip install  flash-attn==2.7.4.post1
apt install lsof

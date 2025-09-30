#!/bin/bash
set -xeuo pipefail

pip install -r requirements_vllm.txt
pip install --no-deps -e .
apt install lsof

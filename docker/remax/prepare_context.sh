#!/usr/bin/env bash
# Run on an Orange-connected builder. Build context contains no local secrets.
set -euo pipefail
repo=$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)
context=${1:?Usage: prepare_context.sh /absolute/build-context}
mkdir -p "$context/source" "$context/packages"
cd "$repo"
python3 docker/remax/stage_source.py "$context/source"
cp docker/remax/Dockerfile "$context/Dockerfile"
blob=az://orngwus2cresco/data/boren/data/packages
for file in cuda-compat-13-0-580.95.05.tgz transferqueue-0.1.11.dev0-py3-none-any.whl flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl; do
    bbb cp "$blob/$file" "$context/packages/$file"
done
for dir in torch211-cu130 causal-conv1d/1.7.0; do
    mkdir -p "$context/packages/$dir"
    bbb sync "$blob/$dir/" "$context/packages/$dir/"
done
mkdir -p "$context/packages/bitbank"
# These wheels are staged at the package root in Blob.
for file in bitbank-0.2.0-py3-none-any.whl aiofiles-25.1.0-py3-none-any.whl fsspec-2026.9.0-py3-none-any.whl; do
    bbb cp "$blob/$file" "$context/packages/bitbank/$file"
done
(cd "$context" && find packages source -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
printf 'Prepared context: %s\n' "$context"

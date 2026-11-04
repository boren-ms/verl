#!/usr/bin/env bash
set -euo pipefail
shard="${1:?shard 0 or 1 required}"
[[ "$shard" == 0 || "$shard" == 1 ]]
cd /root/code/verl
work=/root/data/short_librispeech_20261104
py="$work/env/bin/python"
export PYTHONPATH="/root/code/verl/recipe/phimm/data/short/english_shims:/root/code/verl"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:/root/.pyenv/versions/3.12.9/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false OAIPKG_DISABLE_META_MISSING=1
remote=az://orngwus2cresco/data/boren/data/LibriSpeech/short_2to5_v1
stage=az://orngwus2cresco/data/boren/data/verl/staging/short_librispeech_20261104/shards
mkdir -p "$work/shards/aligned"
for split in train test-other; do
  "$py" -m recipe.phimm.data.short.run_alignment --input "$work/shards/input/$split-$shard.jsonl" --output "$work/shards/aligned/$split-$shard.jsonl" --model "$work/model" --n-gpus 8 --batch-size 8
  "$py" -m recipe.phimm.data.short.build_parallel --input "$work/shards/aligned/$split-$shard.jsonl" --split "$split" --output-dir "$work/shards/output-$shard" --remote-root "$remote" --workers 16
  bbb sync "$work/shards/output-$shard/$split/audio/" "$remote/$split/audio/"
  for file in data.jsonl stats.json rejected.jsonl; do bbb cp "$work/shards/output-$shard/$split/$file" "$stage/output/$shard/$split/$file"; done
  bbb cp "$work/shards/aligned/$split-$shard.jsonl" "$stage/aligned/$split-$shard.jsonl"
done
printf 'SHARD %s COMPLETE\n' "$shard"

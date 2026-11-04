#!/usr/bin/env bash
set -euo pipefail
cd /root/code/verl
work=/root/data/short_librispeech_20261104
py="$work/env/bin/python"
export PYTHONPATH="/root/code/verl/recipe/phimm/data/short/english_shims:/root/code/verl"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:/root/.pyenv/versions/3.12.9/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export OAIPKG_DISABLE_META_MISSING=1
mkdir -p "$work/aligned"
"$py" - "$work" <<'PY'
import json,sys
from pathlib import Path
w=Path(sys.argv[1]); out=w/'input'/'smoke.jsonl'
with out.open('w') as f:
 for s in ('train','test-clean','test-other'):
  with (w/'input'/f'{s}.jsonl').open() as src:
   for _,line in zip(range(5),src): f.write(line)
PY
"$py" -m recipe.phimm.data.short.run_alignment --input "$work/input/smoke.jsonl" --output "$work/aligned/smoke.jsonl" --model "$work/model" --n-gpus 1 --batch-size 4
"$py" -m recipe.phimm.data.short.build_librispeech_short --input "$work/aligned/smoke.jsonl" --split train --output-dir "$work/smoke" --remote-root az://orngwus2cresco/data/boren/data/LibriSpeech/short_2to5_v1
"$py" - "$work/smoke/train/stats.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x.get('written',0)>=12, x
print('SMOKE GATE PASSED',x,flush=True)
PY
for split in test-clean test-other train; do
  "$py" -m recipe.phimm.data.short.run_alignment --input "$work/input/$split.jsonl" --output "$work/aligned/$split.jsonl" --model "$work/model" --n-gpus 8 --batch-size 8
  extra=()
  if [[ "$split" == train ]]; then extra=(--shuffle-input --limit 20424); fi
  "$py" -m recipe.phimm.data.short.build_librispeech_short --input "$work/aligned/$split.jsonl" --split "$split" --output-dir "$work/output" --remote-root az://orngwus2cresco/data/boren/data/LibriSpeech/short_2to5_v1 "${extra[@]}"
done
"$py" -m recipe.phimm.data.short.finalize_short

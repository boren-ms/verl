#!/bin/bash
# Pre-segment every corpus in inhouse_2605_alllocale_seg.yaml via SVAD (30s segments).
# Existing _30s JSONLs are skipped so interrupted batches can resume safely.
# Run on a Brix node with /root/code/verl checked out.
set -euo pipefail

mkdir -p /tmp/logs
cd /root/code/verl

run_lang() {
  local lang="$1"
  shift
  echo "=== explode $lang ==="
  python -m recipe.phimm.data.long_audio_vad_explode \
    --src-root "az://orngwus2cresco/data/speech/users/ruchaofan/Evaluation/InhouseASR_2605/${lang}" \
    --dst-root "az://orngwus2cresco/data/boren/data/Evaluation/InhouseASR_2605_seg_presegment_30s/${lang}" \
    --corpora "$@" \
    --max-len-sec 30 \
    --audio-key WavPath \
    --path-replace '{"/datablob1/": "az://orngwus2cresco/data/speech/"}' \
    --n-workers 32
}

while IFS=$'\t' read -r lang corpora; do
  read -r -a corpus_args <<< "$corpora"
  run_lang "$lang" "${corpus_args[@]}"
done < <(python - <<'PY'
import collections

import blobfile as bf
import yaml

with open("recipe/phimm/config/data/val_data/inhouse_2605_alllocale_seg.yaml") as stream:
    entries = yaml.safe_load(stream)

missing = collections.defaultdict(list)
source_prefix = "InhouseASR_2605_seg_presegment/"
target_prefix = "InhouseASR_2605_seg_presegment_30s/"
for entry in entries:
    path = entry["jsonl_paths"]
    locale, corpus = path.split(source_prefix, 1)[1].rsplit("/test.jsonl", 1)[0].split("/", 1)
    target = path.replace(source_prefix, target_prefix)
    if not bf.exists(target):
        missing[locale].append(corpus)

for locale, corpora in missing.items():
    print(locale, " ".join(corpora), sep="\t")
PY
)
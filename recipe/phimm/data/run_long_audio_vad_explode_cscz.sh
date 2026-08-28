#!/bin/bash
# Pre-segment all InhouseASR_2605 corpora for cs-CZ via SVAD (40s segments).
# Run on a brix node with /root/code/verl checked out.
set -euo pipefail

cd /root/code/verl

python -m recipe.phimm.data.long_audio_vad_explode \
  --src-root "az://orngwus2cresco/data/speech/users/ruchaofan/Evaluation/InhouseASR_2605/cs-CZ" \
  --dst-root "az://orngwus2cresco/data/boren/data/Evaluation/InhouseASR_2605_seg_presegment/cs-CZ" \
  --corpora \
    Conversation_DTEST_FY23Q2_cs-CZ_DTEST \
    Conversation_OnlineMeetings_DTEST_FY24Q2_cs-CZ \
    Dictation_DTEST_L_D_FY24Q2_cs-CZ_DTEST_OfflineDataCollection \
  --max-len-sec 40 \
  --audio-key WavPath \
  --path-replace '{"/datablob1/": "az://orngwus2cresco/data/speech/"}' \
  --n-workers 32

#!/bin/bash
# Pre-segment all 5 nl-NL InhouseASR_2605 corpora via SVAD (40s segments).
# Run on a brix node with /root/code/verl checked out.
set -euo pipefail

mkdir -p /tmp/logs
cd /root/code/verl

python -m recipe.phimm.data.long_audio_vad_explode \
  --src-root az://orngwus2cresco/data/speech/users/ruchaofan/Evaluation/InhouseASR_2605/nl-NL \
  --dst-root az://orngwus2cresco/data/boren/data/Evaluation/InhouseASR_2605_seg_presegment/nl-NL \
  --corpora \
    Conversation_DTEST_FY23Q2_nl-NL_DTEST \
    Conversation_DomainSet_DTEST_Banking_Entity_FY24Q2_nl-NL_DTEST_OfflineDataCollection \
    Conversation_DomainSet_DTEST_Medical_Entity_FY24Q2_nl-NL_DTEST_OfflineDataCollection \
    Conversation_OnlineMeetings_DTEST_FY23Q1_nl-NL_DTEST \
    Dictation_DTEST_L_D_FY23Q4_nl-NL_DTEST \
  --max-len-sec 40 \
  --audio-key WavPath \
  --path-replace '{"/datablob1/": "az://orngwus2cresco/data/speech/"}' \
  --n-workers 32

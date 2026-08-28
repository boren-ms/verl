#!/bin/bash
# Pre-segment all InhouseASR_2605 corpora for da-DK / hu-HU / nb-NO via SVAD
# (40s segments). Run on a brix node with /root/code/verl checked out.
set -euo pipefail

mkdir -p /tmp/logs
cd /root/code/verl

run_lang() {
  local lang="$1"
  shift
  echo "=== explode $lang ==="
  python -m recipe.phimm.data.long_audio_vad_explode \
    --src-root "az://orngwus2cresco/data/speech/users/ruchaofan/Evaluation/InhouseASR_2605/${lang}" \
    --dst-root "az://orngwus2cresco/data/boren/data/Evaluation/InhouseASR_2605_seg_presegment/${lang}" \
    --corpora "$@" \
    --max-len-sec 40 \
    --audio-key WavPath \
    --path-replace '{"/datablob1/": "az://orngwus2cresco/data/speech/"}' \
    --n-workers 32
}

run_lang da-DK \
  Conversation_DTEST_FY21Q3_da-DK_DTEST \
  Conversation_DomainSet_DTEST_Banking_Entity_FY24Q2_da-DK_DTEST_OfflineDataCollection \
  Conversation_DomainSet_DTEST_Medical_Entity_FY24Q2_da-DK_DTEST_OfflineDataCollection \
  Conversation_OnlineMeetings_DTEST_FY23Q1_da-DK_DTEST \
  Dictation_DTEST_L_D_FY23Q4_da-DK_DTEST

run_lang hu-HU \
  Conversation_DTEST_FY22Q4_hu-HU_DTEST \
  Conversation_OnlineMeetings_DTEST_FY24Q2_hu-HU \
  Dictation_DTEST_L_D_FY25Q2_hu-HU_DTEST_OfflineDataCollection

run_lang nb-NO \
  Conversation_DTEST_FY21Q3_nb-NO_DTEST \
  Conversation_DomainSet_DTEST_Banking_Entity_FY24Q2_nb-NO_DTEST_OfflineDataCollection \
  Conversation_DomainSet_DTEST_Medical_Entity_FY24Q2_nb-NO_DTEST_OfflineDataCollection \
  Conversation_OnlineMeetings_DTEST_FY23Q1_nb-NO_DTEST \
  Dictation_DTEST_L_D_FY23Q4_nb-NO_DTEST

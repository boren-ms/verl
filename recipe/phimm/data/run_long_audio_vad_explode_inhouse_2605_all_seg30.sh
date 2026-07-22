#!/bin/bash
# Pre-segment every corpus in inhouse_2605_all_seg.yaml via SVAD (30s segments).
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

run_lang en-US \
  Conversation_DTEST_FY21Q1_en-US \
  Conversation_OnlineMeetings_DTEST_FY25Q3_en-US_DTEST_OfflineDataCollection \
  Dictation_Commonset_OfficeOffline_FY24Q3_en-US_DTEST_OfflineDataCollection \
  OnlineMeetings_CS_Product_FY22_en-US_DTEST \
  OnlineMeetings_CS_Shiproom_FY22_en-US_DTEST

run_lang da-DK \
  Conversation_DTEST_FY21Q3_da-DK_DTEST \
  Conversation_OnlineMeetings_DTEST_FY23Q1_da-DK_DTEST \
  Dictation_DTEST_L_D_FY23Q4_da-DK_DTEST

run_lang hu-HU \
  Conversation_DTEST_FY22Q4_hu-HU_DTEST \
  Conversation_OnlineMeetings_DTEST_FY24Q2_hu-HU \
  Dictation_DTEST_L_D_FY25Q2_hu-HU_DTEST_OfflineDataCollection

run_lang nb-NO \
  Conversation_DTEST_FY21Q3_nb-NO_DTEST \
  Conversation_OnlineMeetings_DTEST_FY23Q1_nb-NO_DTEST \
  Dictation_DTEST_L_D_FY23Q4_nb-NO_DTEST

run_lang nl-NL \
  Conversation_DTEST_FY23Q2_nl-NL_DTEST \
  Conversation_OnlineMeetings_DTEST_FY23Q1_nl-NL_DTEST \
  Dictation_DTEST_L_D_FY23Q4_nl-NL_DTEST

run_lang cs-CZ \
  Conversation_DTEST_FY23Q2_cs-CZ_DTEST \
  Conversation_OnlineMeetings_DTEST_FY24Q2_cs-CZ \
  Dictation_DTEST_L_D_FY24Q2_cs-CZ_DTEST_OfflineDataCollection
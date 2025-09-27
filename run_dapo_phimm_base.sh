#!/usr/bin/env bash
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-1}
HOME=/home/boren
MODEL_PATH=/home/boren/data/ckp/hf_models/phi4_mm_bias_merged

train_data="${HOME}/data/parquet/ls_clean_sc1k_fn1.parquet"
test_data="${HOME}/data/parquet/ls_clean_sc1k_fn1_h100.parquet"


python3 -m recipe.dapo.main_dapo \
--config-name dapo_phimm_ls \
data.train_files="${train_data}" \
data.val_files="${test_data}" \
actor_rollout_ref.model.path="${MODEL_PATH}"
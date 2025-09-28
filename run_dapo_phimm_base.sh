#!/usr/bin/env bash
set -xeuo pipefail

HOME=/root
MODEL_PATH=${HOME}/data/ckp/hf_models/phi4_mm_bias_merged

train_data=${HOME}/data/parquet/ls_sc1k_fn1_remote.parquet
test_data=${HOME}/data/parquet/ls_sc1k_fn1_h100_remote.parquet


python3 -m recipe.dapo.main_dapo \
--config-name dapo_phimm_base \
data.train_files="${train_data}" \
data.val_files="${test_data}" \
actor_rollout_ref.model.path="${MODEL_PATH}"

# trainer.n_gpus_per_node="${NUM_GPUS}" \
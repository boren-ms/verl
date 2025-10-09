#!/usr/bin/env bash
set -xeuo pipefail

# export MODEL_PATH=/home/boren/data/ckp/hf_models/phi4_mm_bias_merged
export MODEL_PATH=/home/boren/data/ckp/hf_models/Phi-4-multimodal-instruct
export DATA_PATH=/home
# export MODEL_PATH=/home/boren/data/ckp/hf_models/phi-libri_ft_m1000_p8_new-QpHq/5000_hf
# export MODEL_PATH=/home/boren/data/ckp/hf_models/phi4-7b-fast-api-s2-final-v4

config_name=dapo_local_test
val_before_train=false

python3 -m recipe.phimm.main_asr_dapo \
--config-name ${config_name} \
trainer.n_gpus_per_node=1 \
trainer.logger=console \
trainer.val_before_train=${val_before_train} \
actor_rollout_ref.model.path="${MODEL_PATH}" 2>&1 | tee ${config_name}.log

#!/usr/bin/env bash
set -xeuo pipefail

HOME=/home/boren
MODEL_PATH=${HOME}/data/ckp/hf_models/phi4_mm_bias_merged

config_name=dapo_phimm_lora_new_local

python3 -m recipe.dapo.main_dapo \
--config-name ${config_name} \
+data.eval_num_examine=5 \
+data.train_num_examine=1 \
data.train_batch_size=4 \
data.gen_batch_size=4 \
actor_rollout_ref.rollout.n=4 \
trainer.n_gpus_per_node=1 \
trainer.logger=console \
actor_rollout_ref.model.path="${MODEL_PATH}" 2>&1 | tee ${config_name}.log

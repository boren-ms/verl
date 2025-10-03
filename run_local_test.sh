#!/usr/bin/env bash
set -xeuo pipefail

export MODEL_PATH=/home/boren/data/ckp/hf_models/phi4_mm_bias_merged
export DATA_PATH=/home # used in yamls

config_name=dapo_local_test

python3 -m recipe.phimm.main_dapo \
--config-name ${config_name} \
data.train_batch_size=4 \
data.gen_batch_size=4 \
actor_rollout_ref.rollout.n=4 \
trainer.n_gpus_per_node=1 \
trainer.logger=console \
actor_rollout_ref.model.path="${MODEL_PATH}" 2>&1 | tee ${config_name}.log

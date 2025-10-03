#!/usr/bin/env bash
set -xeuo pipefail

config_name=dapo_prod_fy22_bs8_n16_rep


export DATA_PATH=az://orngcresco/data
export OUTPUT_PATH=/root/outputs/${config_name}

python3 -m recipe.phimm.main_dapo \
--config-name ${config_name} \
actor_rollout_ref.rollout.n=4 \
trainer.n_gpus_per_node=1 \
trainer.logger=console \
trainer.default_local_dir="${OUTPUT_PATH}" 2>&1 | tee ${config_name}.log

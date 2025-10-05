#!/usr/bin/env bash
set -xeuo pipefail

config_name=dapo_ls_bs64

export OUTPUT_PATH=/root/outputs/${config_name}
val_before_train=false


python3 -m recipe.phimm.main_asr_dapo \
--config-name ${config_name} \
actor_rollout_ref.rollout.n=4 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
trainer.n_gpus_per_node=1 \
trainer.val_before_train=${val_before_train} \
trainer.logger=console \
trainer.default_local_dir="${OUTPUT_PATH}" 2>&1 | tee ${config_name}.log

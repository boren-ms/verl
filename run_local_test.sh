#!/usr/bin/env bash
set -xeuo pipefail

HOME=/root
# export DATA_PATH=az://${DATA_STORAGE}/data/boren/data
export DATA_PATH="az://orngcresco/data/boren/data"
export MODEL_PATH=${HOME}/data/ckp/hf_models/phi4_mm_bias_merged

batch_size=128
config_name=dapo_phimm_lora_ls
python3 -m recipe.phimm.main_asr_dapo \
--config-name ${config_name} \
data.train_batch_size=${batch_size} \
data.gen_batch_size=$((batch_size * 3 / 2)) \
actor_rollout_ref.rollout.n=8 \
actor_rollout_ref.model.path=${MODEL_PATH} \
actor_rollout_ref.actor.ppo_mini_batch_size=$((batch_size * 8)) \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${batch_size} \
actor_rollout_ref.actor.entropy_coeff=0 \
actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${batch_size} \
actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${batch_size} \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
trainer.experiment_name=${config_name} \
trainer.total_epochs=1  2>&1 | tee ${config_name}.log

# trainer.n_gpus_per_node="${NUM_GPUS}"
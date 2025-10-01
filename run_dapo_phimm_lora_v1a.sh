#!/usr/bin/env bash
set -xeuo pipefail

HOME=/root
model_path=${HOME}/data/ckp/hf_models/phi4_mm_bias_merged

train_data=${HOME}/data/parquet/ls_sc1k_fn1_remote.parquet
test_data=${HOME}/data/parquet/ls_sc1k_fn1_h100_remote.parquet

batch_size=128

python3 -m recipe.dapo.main_dapo \
--config-name dapo_phimm_lora \
data.train_files=${train_data} \
data.val_files=${test_data} \
data.train_batch_size=${batch_size} \
data.gen_batch_size=$((batch_size * 3 / 2)) \
actor_rollout_ref.rollout.n=8 \
+data.eval_num_examine=5 \
+data.train_num_examine=1 \
actor_rollout_ref.model.path=${model_path} \
actor_rollout_ref.actor.ppo_mini_batch_size=$((batch_size * 8)) \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${batch_size} \
actor_rollout_ref.actor.entropy_coeff=0 \
actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${batch_size} \
actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${batch_size} \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
trainer.total_epochs=1  2>&1 | tee run_dapo_phimm_lora_v1.log

# trainer.n_gpus_per_node="${NUM_GPUS}"
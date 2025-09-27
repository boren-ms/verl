#!/usr/bin/env bash
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-1}
HOME=/home/boren
# MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
# MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
#huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"
# model_path=/home/boren/data/ckp/hf_models/Phi-4-multimodal-instruct
# MODEL_PATH=/home/boren/data/ckp/hf_models/Qwen2.5-0.5B-Instruct
# MODEL_PATH=/home/boren/data/ckp/hf_models/phi4_mm_bias_merged
MODEL_PATH=/home/boren/data/ckp/hf_models/phi4_mm_bias_merged
# MODEL_PATH=/home/boren/data/ckp/hf_models/Phi-4-mini-instruct

# prepare data
# python3 examples/data_preprocess/gsm8k.py --local_save_dir /home/boren/data/parquet/gsm8k
# train_data="${HOME}/data/parquet/gsm8k/train.parquet"
# test_data="${HOME}/data/parquet/gsm8k/test.parquet"

train_data="${HOME}/data/parquet/ls_clean_sc1k_fn1.parquet"
test_data="${HOME}/data/parquet/ls_clean_sc1k_fn1_h100.parquet"

adv_estimator=grpo

kl_coef=0.0
use_kl_in_reward=False
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=1024
max_response_length=2048
enable_overlong_buffer=True
overlong_buffer_len=128
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=seq_reward
max_num_gen_batches=10

train_traj_micro_bsz_per_gpu=1 # b
n_resp_per_prompt=4 # g

train_traj_micro_bsz=$((train_traj_micro_bsz_per_gpu * NUM_GPUS)) # b * n
train_traj_mini_bsz=$((train_traj_micro_bsz * 1)) # 2 * b * n
train_prompt_mini_bsz=$((train_traj_mini_bsz * n_resp_per_prompt)) # 2 * b * n / g
train_prompt_bsz=$((train_prompt_mini_bsz * 2)) # 4 * b * n / g

gen_prompt_bsz=$((train_prompt_bsz * 4))


# exp_name="$(basename "${MODEL_ID,,}")-dapo-minimal"
exp_name="phi4mm-dapo-minimal"
python3 -m recipe.dapo.main_dapo \
data.train_files="${train_data}" \
data.val_files="${test_data}" \
reward_model.reward_manager=dapo \
algorithm.adv_estimator=${adv_estimator} \
algorithm.use_kl_in_reward=False \
algorithm.kl_ctrl.kl_coef=0.0 \
actor_rollout_ref.actor.use_kl_loss=False \
actor_rollout_ref.actor.kl_loss_coef=0.0 \
actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
data.prompt_key="prompt" \
+data.asr_dataset=True \
+data.audio_key="audio_path" \
data.trust_remote_code=True \
data.max_prompt_length=${max_prompt_length} \
data.max_response_length=${max_response_length} \
reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
reward_model.overlong_buffer.len=${overlong_buffer_len} \
reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
reward_model.model.trust_remote_code=True \
custom_reward_function.path=verl/utils/reward_score/speech_recognition.py \
custom_reward_function.name=compute_score \
actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
data.train_batch_size=${train_prompt_bsz} \
data.gen_batch_size=${gen_prompt_bsz} \
algorithm.filter_groups.enable=${enable_filter_groups} \
algorithm.filter_groups.metric=${filter_groups_metric} \
algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
actor_rollout_ref.model.path="${MODEL_PATH}" \
actor_rollout_ref.model.trust_remote_code=True \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.model.use_fused_kernels=False \
actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
actor_rollout_ref.actor.fsdp_config.param_offload=False \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
actor_rollout_ref.actor.strategy=fsdp2 \
actor_rollout_ref.ref.strategy=fsdp2 \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
trainer.logger='["console","wandb"]' \
trainer.project_name='verl-phi4mm' \
trainer.experiment_name="${exp_name}" \
trainer.n_gpus_per_node=${NUM_GPUS} \
trainer.nnodes=1 \
trainer.save_freq=-1 \
trainer.total_epochs=2 \
trainer.resume_mode=disable \
trainer.val_before_train=False \
trainer.total_training_steps=100 $@
# actor_rollout_ref.model.use_fused_kernels=False \ # buggy for phi4mm, need to check on  patch_forward_with_backends
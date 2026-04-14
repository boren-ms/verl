#!/usr/bin/env bash
# ============================================================================
# 03_run_training.sh
# Run this ON THE REMOTE NODE to launch FULLY ASYNC GRPO training.
# Uses Qwen2.5-Math-7B on 8 GPUs: 4 rollout + 4 training (disaggregated).
# Based on: verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_4_4.sh
#
# Uses the system pyenv Python 3.12.9 directly and connects to the pod's
# existing Ray cluster (port 6380). DO NOT 'ray stop' — it kills the pod.
#
# Usage:
#   cd ~/code/verl
#   bash scripts/remote_setup/03_run_training.sh
# ============================================================================
set -xeuo pipefail

# --------------------------------------------------------------------------
# Use system pyenv Python directly (has torch 2.8.0, ray 2.46.0, etc.)
# --------------------------------------------------------------------------
export PATH="/root/.pyenv/versions/3.12.9/bin:${PATH}"

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
RAY_DATA_HOME="${RAY_DATA_HOME:-${HOME}/verl_data}"

project_name='DAPO'
exp_name='DAPO-Qwen2.5-7b-MATH-fsdp2-fully-async-4-4'

MODEL_PATH="${RAY_DATA_HOME}/models/Qwen2.5-Math-7B"
CKPTS_DIR="${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"
TRAIN_FILE="${RAY_DATA_HOME}/data/dapo-math-17k.parquet"
TEST_FILE="${RAY_DATA_HOME}/data/aime-2024.parquet"

# Verify files exist
for f in "${MODEL_PATH}/config.json" "${TRAIN_FILE}" "${TEST_FILE}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing required file: $f"
        echo "Run 02_remote_setup.sh first."
        exit 1
    fi
done

# --------------------------------------------------------------------------
# Environment variables
# --------------------------------------------------------------------------
export VLLM_USE_V1=1

# --------------------------------------------------------------------------
# Connect to the pod's existing Ray cluster (port 6380).
# DO NOT 'ray stop' — it kills the pod.
# --------------------------------------------------------------------------
unset RAY_RUNTIME_ENV_PLUGINS
export RAY_ADDRESS="127.0.0.1:6380"

# --------------------------------------------------------------------------
# Fully async parameters
# --------------------------------------------------------------------------
# GPU split: 4 rollout + 4 training
NNODES=1
NGPUS_PER_NODE=8
n_gpus_rollout=4
n_gpus_training=4

# Algorithm
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# Sequence lengths — model config.json has max_position_embeddings=8192
max_prompt_length=1024
max_response_length=4096

# This controls overlong penalty
enable_overlong_buffer=True
overlong_buffer_len=2048
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

# Sampling
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# Performance
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
ref_offload=True
actor_offload=False
gen_tp=1
sp_size=1
fsdp_size=2

# Async-specific
gen_prompt_bsz=1
n_resp_per_prompt=16
train_prompt_mini_bsz=32
total_rollout_steps=$((512 * 100))
test_freq=10
staleness_threshold=0.1
trigger_parameter_sync_step=4
require_batches=4
partial_rollout=True

# --------------------------------------------------------------------------
# Launch fully async training
# --------------------------------------------------------------------------
cd "${HOME}/code/verl"

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=0 \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.hybrid_engine=False \
    +actor_rollout_ref.model.override_config.max_position_embeddings=8192 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=True \
    trainer.save_freq=-1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${n_gpus_training} \
    rollout.nnodes=${NNODES} \
    rollout.n_gpus_per_node=${n_gpus_rollout} \
    rollout.total_rollout_steps=${total_rollout_steps} \
    trainer.total_epochs=10 \
    trainer.test_freq=${test_freq} \
    async_training.staleness_threshold=${staleness_threshold} \
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step} \
    async_training.require_batches=${require_batches} \
    async_training.partial_rollout=${partial_rollout}

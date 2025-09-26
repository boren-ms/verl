#!/usr/bin/env bash
set -xeuo pipefail
data_path=/home/boren/data/parquet
# model_path=/home/boren/data/ckp/hf_models/Phi-4-multimodal-instruct
model_path=/home/boren/data/ckp/hf_models/phi4_mm_bias_merged
# model_id=Phi-4-mini-instruct
# model_id=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# huggingface-cli download $model_id --local-dir ${model_path}/${model_id}


test_name=ls_sc1k_fn1_h100.parquet
output_path=/home/boren/data/outputs/${test_name%.parquet}

# python3 -m verl.trainer.main_generation \
python3 -m verl.trainer.main_gen \
trainer.nnodes=1 \
trainer.n_gpus_per_node=1 \
data.path=${data_path}/${test_name} \
data.prompt_key="prompt" \
data.pad_to_max=False \ # use dynamic padding
data.batch_size=2 \
data.n_samples=1 \
data.output_path=${output_path}/${test_name} \
model.path=${model_path} \
+model.trust_remote_code=True \
rollout.temperature=0.6 \
rollout.top_p=0.95 \
rollout.prompt_length=1024 \
rollout.response_length=1024 \
rollout.tensor_model_parallel_size=1 \
rollout.gpu_memory_utilization=0.25 \
rollout.max_num_batched_tokens=10240 \
rollout.enforce_eager=False \
+rollout.enable_prefix_caching=False \
rollout.free_cache_engine=True

# +model.lora_rank=320 \
# +model.lora_alpha=16 \
# +model.target_modules=all-linear \
# python3 -m recipe.r1.main_eval \
# data.path=${output_path}/test-output-k1.parquet \
# data.prompt_key=question \
# data.response_key=answer \
# custom_reward_function.path=recipe/r1/reward_score.py \
# custom_reward_function.name=reward_func
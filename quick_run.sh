#!/bin/bash
set -xeuo pipefail

config_file=$1

config_base=$(basename "$config_file")
config_name=${config_base%.*}

cwd="$(dirname $(readlink -f $0))"
echo "Current working directory: ${cwd}"
pushd "$cwd" > /dev/null


module="recipe.phimm.main_asr_dapo"

if [[ "$config_name" == gen_* ]]; then
    module="recipe.phimm.main_asr_gen"
elif [[ "$config_name" == eval_* ]]; then
    module="recipe.phimm.main_asr_eval"
elif [[ "$config_name" == remax_* ]]; then
    module="recipe.phimm.main_asr_remax"
fi

# bash quick_install.sh # prepare on local node only
echo "[INFO] Preparing environment ..."
python3 ray_tool.py prepare_env # prepare on all ray nodes

echo "[INFO] Running ${config_name} ..."
# VLLM_USE_V1=0 forces the legacy vLLM engine to avoid a known v1 cudagraph+LoRA
# IndexError in column_parallel_linear.set_lora when target_modules uses packed
# names (qkv_proj, gate_up_proj). Override at job-submit time via VLLM_USE_V1
# env in your shell to opt back into v1.
: "${VLLM_USE_V1:=0}"
ray job submit --working-dir="${cwd}" \
--runtime-env-json="{\"env_vars\":{\"VLLM_USE_V1\":\"${VLLM_USE_V1}\"}}" \
--no-wait -- \
python3 -m ${module} \
--config-name "${config_name}" \
trainer.experiment_name="${config_name}" \
2>&1 | tee "${config_name}.log"

echo "[INFO] Finished ${config_name}."
popd > /dev/null

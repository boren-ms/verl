#!/bin/bash
set -xeuo pipefail

config_file=$1

config_base=$(basename "$config_file")
config_name=${config_base%.*}
config_dir=$(dirname "$config_file")

cwd="$(dirname $(readlink -f $0))"
echo "Current working directory: ${cwd}"
pushd "$cwd" > /dev/null


module="recipe.phimm.main_asr_dapo"

if [[ "$config_name" == long_eval_* ]]; then
    module="recipe.phimm.main_long_eval_asr"
elif [[ "$config_name" == gen_* ]]; then
    module="recipe.phimm.main_asr_gen"
elif [[ "$config_name" == eval_* ]]; then
    module="recipe.phimm.main_asr_eval"
elif [[ "$config_name" == remax_* ]]; then
    module="recipe.phimm.main_asr_remax"
elif [[ "$config_name" == grpo_* ]]; then
    module="recipe.phimm.main_asr_grpo"
elif [[ "$config_name" == gdpo_* ]]; then
    module="recipe.phimm.main_asr_grpo"
elif [[ "$config_name" == gmpo_* ]]; then
    module="recipe.phimm.main_asr_grpo"
elif [[ "$config_name" == gspo_* ]]; then
    module="recipe.phimm.main_asr_grpo"
elif [[ "$config_name" == rloo_* ]]; then
    module="recipe.phimm.main_asr_rloo"
fi

# bash quick_install.sh # prepare on local node only
echo "[INFO] Preparing environment ..."
python3 ray_tool.py prepare_env # prepare on all ray nodes

echo "[INFO] Running ${config_name} ..."
cuda_compat_ld_path="/usr/local/cuda-13.0/compat:/root/.pyenv/versions/3.12.9/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
ray job submit --working-dir="${cwd}" \
--runtime-env-json "{\"env_vars\":{\"LD_LIBRARY_PATH\":\"${cuda_compat_ld_path}\"}}" \
--no-wait -- \
python3 -m ${module} \
--config-dir "${config_dir}" \
--config-name "${config_name}" \
trainer.experiment_name="${config_name}" \
2>&1 | tee "${config_name}.log"

echo "[INFO] Finished ${config_name}."
popd > /dev/null

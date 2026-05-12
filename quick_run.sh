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
prepare_env_args=()
if [[ "$config_name" == *qwen* ]]; then
    export QWEN35_AUDIO_DISABLE_CUDNN=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_PLUGINS=qwen35_audio
    prepare_env_args=(--profile qwen35_audio)
fi
python3 ray_tool.py prepare_env "${prepare_env_args[@]}" # prepare on all ray nodes

echo "[INFO] Running ${config_name} ..."
ray job submit --working-dir="${cwd}"  \
--no-wait -- \
python3 -m ${module} \
--config-name "${config_name}" \
trainer.experiment_name="${config_name}" \
2>&1 | tee "${config_name}.log"

echo "[INFO] Finished ${config_name}."
popd > /dev/null

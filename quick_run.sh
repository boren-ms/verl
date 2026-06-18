#!/bin/bash
set -xeuo pipefail

config_file=$1

config_base=$(basename "$config_file")
config_name=${config_base%.*}

cwd="$(dirname $(readlink -f $0))"
echo "Current working directory: ${cwd}"
pushd "$cwd" > /dev/null

# Determine the module and config-path based on config name prefix
# config_path is relative to the module's file location (Hydra convention)
module="verl.trainer.main_ppo"
config_path="../../recipe/phimm/config"

if [[ "$config_name" == *fully_async* ]]; then
    module="verl.experimental.fully_async_policy.fully_async_main"
    config_path="../../../recipe/phimm/config"
elif [[ "$config_name" == long_eval_* ]]; then
    module="recipe.phimm.main_long_eval_asr"
    config_path="config"
elif [[ "$config_name" == gen_* ]]; then
    module="recipe.phimm.main_asr_gen"
    config_path="config/gen"
elif [[ "$config_name" == eval_* ]]; then
    module="recipe.phimm.main_asr_eval"
    config_path="config"
elif [[ "$config_name" == remax_* ]]; then
    module="recipe.phimm.main_asr_remax"
    config_path="config"
fi

echo "[INFO] Preparing environment ..."
bash quick_install.sh

echo "[INFO] Running ${config_name} with module ${module} ..."
ray job submit --working-dir=. \
--no-wait -- \
python3 -m ${module} \
--config-path="${config_path}" \
--config-name "${config_name}" \
trainer.experiment_name="${config_name}" \
"${@:2}" \
2>&1 | tee "recipe/phimm/${config_name}.log"

echo "[INFO] Finished submitting ${config_name}."
popd > /dev/null

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
fi

# bash quick_install.sh # prepare on local node only
echo "[INFO] Preparing environment ..."
python3 ray_tool.py prepare_env # prepare on all ray nodes

echo "[INFO] Running ${config_name} ..."
ray job submit --working-dir="${cwd}"  \
--no-wait -- \
python3 -m ${module} \
--config-name "${config_name}" \
trainer.experiment_name="${config_name}" \
2>&1 | tee "${config_name}.log"

echo "[INFO] Finished ${config_name}."
popd > /dev/null
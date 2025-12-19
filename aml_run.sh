#!/bin/bash
set -xeuo pipefail

config_file=$1

config_base=$(basename "$config_file")
config_name=${config_base%.*}
hostname=$(hostname)

echo "[${hostname}] Setting up AML environment ..."
python setup_aml_env.py --log_env True --log_pip True

if [[ "$hostname" != "node-0" ]]; then
    echo "[${hostname}] This script should be run on node-0 only. "
    exit 0
fi

cwd="$(dirname $(readlink -f $0))"
echo "[${hostname}] Current working directory: ${cwd}"
pushd "$cwd" > /dev/null


module="recipe.phimm.main_asr_dapo"

if [[ "$config_name" == gen_* ]]; then
    module="recipe.phimm.main_asr_gen"
fi

echo "[${hostname}] Running ${config_name} ..."
ray job submit --working-dir="${cwd}"  \
python3 -m ${module} \
--config-name "${config_name}" \
trainer.experiment_name="${config_name}" 

echo "[${hostname}] Finished ${config_name}."
popd > /dev/null
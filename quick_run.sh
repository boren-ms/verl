#!/bin/bash
set -xeuo pipefail

config_file=$1

config_base=$(basename "$config_file")
config_name=${config_base%.*}

pushd "$(dirname "$0")" > /dev/null

echo "[INFO] Installing environment..."
bash ./quick_install.sh

echo "[INFO] Running ${config_name} ..."

ray job submit --no-wait -- \
python3 -m recipe.phimm.main_asr_dapo \
--config-name "${config_name}" \
trainer.experiment_name="${config_name}" \
2>&1 | tee "${config_name}.log"

echo "[INFO] Finished ${config_name}."
popd > /dev/null
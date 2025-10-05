#!/bin/bash
set -xeuo pipefail

config_file=$1

config_dir=$(dirname "$config_file")
config_base=$(basename "$config_file")
config_name=${config_base%.*}

pushd "$(dirname "$0")" > /dev/null

echo "[INFO] Installing environment..."
bash ./quick_install.sh

echo "[INFO] Running ASR DAPO with config: ${config_name}..."

python3 -m recipe.phimm.main_asr_dapo \
--config-name "${config_name}" \
--config-path "${config_dir}" \
trainer.experiment_name="${config_name}"

echo "[INFO] Finished ASR DAPO."
popd > /dev/null
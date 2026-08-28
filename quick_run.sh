#!/bin/bash
set -xeuo pipefail

# Usage: bash quick_run.sh <config> [extra hydra overrides...]
#   <config> may be:
#     - a file path:           recipe/phimm/config/remax/remax_r2_..._binary_adv.yaml
#     - a name with subdir:    remax/remax_r2_..._binary_adv
#     - a bare config name:    gen_oss_ls   (located automatically under the config root)

config_input=$1

cwd="$(dirname $(readlink -f $0))"
echo "Current working directory: ${cwd}"
pushd "$cwd" > /dev/null

config_root="recipe/phimm/config"

# Normalize the input to a path relative to ${config_root} (no .yaml extension).
config_rel=${config_input%.yaml}
config_rel=${config_rel#./}
config_rel=${config_rel#${config_root}/}

config_file="${config_root}/${config_rel}.yaml"
if [[ ! -f "$config_file" ]]; then
    # Fall back to locating the config by its stem anywhere under the config root.
    stem=$(basename "$config_rel")
    found=$(find "$config_root" -name "${stem}.yaml" | head -n1)
    if [[ -z "$found" ]]; then
        echo "[ERROR] Could not find a config for '${config_input}' under ${config_root}" >&2
        exit 1
    fi
    config_file="$found"
    config_rel="${config_file#${config_root}/}"
    config_rel="${config_rel%.yaml}"
fi

# Stem (used for --config-name, experiment_name and the log file) and subdir.
config_name=$(basename "$config_rel")
subdir=$(dirname "$config_rel")
[[ "$subdir" == "." ]] && subdir=""

if [[ "$config_name" == long_rollout_* ]]; then
    # Fully-async long-audio ASR rollout. Module: recipe/phimm/long_asr_rollout.py.
    module="recipe.phimm.long_asr_rollout"
    base="config"
elif [[ "$subdir" == "rollout" || "$config_name" == rollout_* ]]; then
    # Fully-async ASR rollout/generation. Module: recipe/phimm/asr_rollout.py.
    module="recipe.phimm.asr_rollout"
    base="config"
else
    # Default PPO trainer. Module: verl/trainer/main_ppo.py.
    module="verl.trainer.main_ppo"
    base="../../recipe/phimm/config"
fi

if [[ -n "$subdir" ]]; then
    config_path="${base}/${subdir}"
else
    config_path="${base}"
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

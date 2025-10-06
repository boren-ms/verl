#!/bin/bash

set -euo pipefail

job_name=$1
config_file=$2
sync_code=${3:-"true"}


if [ -z "$job_name" ] || [ -z "$config_file" ]; then
    echo "Usage: $0 <job_name> <config_file> [sync_code]"
    exit 1
fi
if [ "$sync_code" == "true" ]; then
    echo "syncing code to $job_name"
    rcall-brix sync "$job_name"
fi

config_name=$(basename "${config_file%.*}")
echo "Submitting job to $job_name with config $config_file"
rcall-brix ssh  "$job_name" "bash -l /root/code/verl/quick_run.sh $config_file" | tee "${job_name}_${config_name}.log"
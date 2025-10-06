#!/bin/bash

set -euo pipefail

job_name=$1
config_file=$2
val_only=${3:-"false"}
cleanup=${4:-"false"}
dry_run=${5:-"false"}
sync_code=${6:-"true"}
config_name=$(basename "${config_file%.*}")
echo "Job: $job_name $config_file"

if [ -z "$job_name" ] || [ -z "$config_file" ]; then
    echo "Usage: $0 <job_name> <config_file> [sync_code]"
    exit 1
fi
if [ "$sync_code" == "true" ]; then
    echo "syncing code to $job_name"
    rcall-brix sync "$job_name"
fi
if [ "$cleanup" == "true" ]; then
    echo "cleaning up old ${config_name} on $job_name"
    rcall-brix ssh "$job_name" "bash -lc \"python /root/code/verl/ray_job.py cleanup ${config_name}\""
fi

if [ "$dry_run" == "true" ]; then
    echo "Dry run"
    echo rcall-brix ssh  "$job_name" "bash -l /root/code/verl/quick_run.sh $config_file $val_only"  | tee "${job_name}_${config_name}.log"
    exit 0
fi

log_dir=logs/$job_name
mkdir -p $log_dir

rcall-brix ssh  "$job_name" "bash -l /root/code/verl/quick_run.sh $config_file $val_only" | tee "${log_dir}/${config_name}.log"
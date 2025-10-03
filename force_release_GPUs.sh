#!/bin/bash
# Force release all GPUs

# This script will kill all processes using NVIDIA GPUs that are related to Python or vLLM.
lsof /dev/nvidia* | grep -E 'ray' | awk '{print $2}' | sort -u
lsof /dev/nvidia* | grep -E 'ray' | awk '{print $2}' | sort -u | xargs -r kill -9
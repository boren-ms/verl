#!/bin/bash
set -xeuo pipefail

cwd="$(dirname $(readlink -f $0))"

echo "Submitting Ray job ..."
echo "Current working directory: ${cwd}"
echo "Cmd: ${*}"
pushd "$cwd" > /dev/null
ray job submit --working-dir="${cwd}" --no-wait -- \
bash quick_run.sh "$*"
popd > /dev/null
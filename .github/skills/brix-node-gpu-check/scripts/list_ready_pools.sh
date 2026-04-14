#!/usr/bin/env bash
# List Brix pools and output only Ready ones as tab-separated values.
# Usage: list_ready_pools.sh [PATTERN]
#   PATTERN  optional glob to filter pool names (e.g. "bus-ats-*")

set -euo pipefail

PATTERN="${1:-}"

# Strip ANSI escape codes from brix output, then filter Ready rows
rcall-brix ls 2>&1 \
  | sed 's/\x1b\[[0-9;]*m//g' \
  | grep -E '\bReady\b' \
  | while read -r line; do
      name=$(echo "$line" | awk '{print $1}')
      cluster=$(echo "$line" | awk '{print $2}')
      size=$(echo "$line" | grep -oP '\d+ x \d+ GPU' || true)
      if [ -n "$PATTERN" ]; then
        # shellcheck disable=SC2254
        case "$name" in $PATTERN) ;; *) continue ;; esac
      fi
      printf '%s\t%s\t%s\n' "$name" "$cluster" "$size"
    done

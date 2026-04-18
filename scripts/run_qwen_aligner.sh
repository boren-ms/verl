#!/bin/bash
# Run Qwen3 Forced Aligner on a remote verl-n1-* node.
#
# Usage:
#   bash scripts/run_qwen_aligner.sh <NODE> <AUDIO_PATH> [TEXT] [LANGUAGE]
#
# Examples:
#   # ASR + alignment (auto-transcribe then align)
#   bash scripts/run_qwen_aligner.sh verl-n1-i2 /path/to/audio.wav "" English
#
#   # Alignment only (provide transcript)
#   bash scripts/run_qwen_aligner.sh verl-n1-i2 /path/to/audio.wav "Hello world" English
#
# The script will:
#   1. Sync code to the remote node
#   2. Install qwen-asr if needed
#   3. Run the aligner script

set -euo pipefail

NODE=${1:?Usage: $0 <NODE> <AUDIO_PATH> [TEXT] [LANGUAGE]}
AUDIO=${2:?Usage: $0 <NODE> <AUDIO_PATH> [TEXT] [LANGUAGE]}
TEXT=${3:-""}
LANGUAGE=${4:-"English"}

echo "=== Qwen3 Forced Aligner ==="
echo "Node: $NODE"
echo "Audio: $AUDIO"
echo "Text: ${TEXT:-'(will auto-transcribe)'}"
echo "Language: $LANGUAGE"

# Step 1: sync code
echo ""
echo "[1/3] Syncing code to $NODE..."
rcall-brix sync "$NODE"

# Step 2: install qwen-asr
echo ""
echo "[2/3] Installing qwen-asr on $NODE..."
rcall-brix ssh "$NODE" "bash -lc 'pip install -U qwen-asr 2>&1 | tail -3'"

# Step 3: run aligner
echo ""
echo "[3/3] Running aligner..."
if [ -n "$TEXT" ]; then
    # Alignment only mode
    rcall-brix ssh "$NODE" "bash -lc 'cd /root/code/verl && python scripts/qwen_aligner.py \
        --audio \"$AUDIO\" \
        --text \"$TEXT\" \
        --language \"$LANGUAGE\" \
        --flash-attn'"
else
    # ASR + alignment mode
    rcall-brix ssh "$NODE" "bash -lc 'cd /root/code/verl && python scripts/qwen_aligner.py \
        --audio \"$AUDIO\" \
        --language \"$LANGUAGE\" \
        --flash-attn'"
fi

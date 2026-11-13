#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <publish_blob_root> <local_dest_dir> [parallel_downloads]" >&2
  exit 2
fi

ROOT=${1%/}
DST=$2
DOWNLOAD_JOBS=${3:-4}
LORA=lora_weights.pt
STEM=${LORA%.pt}
SPLIT="$ROOT/${STEM}_split"

[[ "$ROOT" == az://*/*/* ]] || {
  echo "Expected az://<account>/<container>/<blob-root>: $ROOT" >&2
  exit 2
}
[[ "$DOWNLOAD_JOBS" =~ ^[1-9][0-9]*$ ]] || {
  echo "parallel_downloads must be a positive integer: $DOWNLOAD_JOBS" >&2
  exit 2
}
command -v bbb >/dev/null || {
  echo "bbb is required on the local node" >&2
  exit 1
}

mkdir -p "$DST/splits"
cd "$DST"

if bbb ls "$SPLIT/$STEM.md5" >/dev/null 2>&1; then
  bbb cp "$SPLIT/$STEM.md5" "$STEM.md5"
  bbb cp "$SPLIT/splits.md5" splits/splits.md5
  bbb ls "$SPLIT/" | grep -E "$STEM\.(z[0-9]+|zip)$" > parts.list
  [[ -s parts.list ]] || {
    echo "No split parts found under $SPLIT/" >&2
    exit 1
  }
  xargs -a parts.list -P "$DOWNLOAD_JOBS" -I {} bbb cp -q "{}" splits/

  (cd splits && md5sum -c splits.md5)
  (
    cd splits
    shopt -s nullglob
    data_parts=("$STEM".z[0-9]*)
    shopt -u nullglob
    (( ${#data_parts[@]} > 0 )) || {
      echo "No numbered split parts were downloaded" >&2
      exit 1
    }
    mapfile -t data_parts < <(printf '%s\n' "${data_parts[@]}" | sort -V)
    [[ -f "$STEM.zip" ]] || {
      echo "Missing final split part: $STEM.zip" >&2
      exit 1
    }
    cat "${data_parts[@]}" "$STEM.zip"
  ) > joined.zip
  unzip -p joined.zip > "$LORA" 2>/dev/null || true
  md5sum -c "$STEM.md5"
  rm -rf splits joined.zip parts.list
else
  rmdir splits
  bbb cp "$ROOT/$LORA" "$LORA"
fi

"${PYTHON:-python}" - "$DST/$LORA" <<'PY'
from pathlib import Path
import re
import sys

import torch

path = Path(sys.argv[1])
weights = torch.load(path, map_location="cpu", weights_only=False)
assert isinstance(weights, dict) and weights, "empty LoRA weight artifact"

pattern = re.compile(
    r"^(?P<prefix>.+)\.lora_(?P<side>A|B)(?:\.(?P<adapter>[^.]+))?\.weight$"
)
pairs = {}
for key, value in weights.items():
    match = pattern.match(key)
    assert match, f"non-LoRA key: {key}"
    assert torch.is_tensor(value), f"non-tensor value: {key}"
    pair_id = (match.group("prefix"), match.group("adapter") or "")
    pairs.setdefault(pair_id, set()).add(match.group("side"))

incomplete = [pair_id for pair_id, sides in pairs.items() if sides != {"A", "B"}]
assert not incomplete, f"incomplete LoRA pairs: {incomplete[:10]}"
print(
    f"{path}: {len(weights)} tensors, {len(pairs)} complete A/B pairs, "
    f"{path.stat().st_size / 2**20:.1f} MiB"
)
PY

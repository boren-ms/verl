#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <publish_blob_root> <actor_dir|model_world_size_*_rank_*.pt>" >&2
  exit 2
fi

ROOT_URI=${1%/}
SOURCE_SPEC=$2
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORK=/tmp/lora_weight_transfer
LORA=lora_weights.pt
STEM=${LORA%.pt}

[[ "$ROOT_URI" == az://*/*/* ]] || {
  echo "Expected az://<account>/<container>/<blob-root>: $ROOT_URI" >&2
  exit 2
}
command -v bbb >/dev/null || {
  echo "bbb is required on the remote node" >&2
  exit 1
}

rm -rf "$WORK"
mkdir -p "$WORK/model_shards" "$WORK/splits"

if [[ "$SOURCE_SPEC" == az://* ]]; then
  if [[ "$SOURCE_SPEC" == *model_world_size_* ]]; then
    SOURCE_DIR=${SOURCE_SPEC%/*}
  else
    SOURCE_DIR=${SOURCE_SPEC%/}
  fi
  mapfile -t source_shards < <(
    bbb ls "$SOURCE_DIR/" \
      | grep -E 'model_world_size_[0-9]+_rank_[0-9]+\.pt$' \
      | sort -V
  )
else
  if [[ -d "$SOURCE_SPEC" ]]; then
    SOURCE_DIR=${SOURCE_SPEC%/}
  else
    SOURCE_DIR=${SOURCE_SPEC%/*}
    [[ "$SOURCE_DIR" != "$SOURCE_SPEC" ]] || SOURCE_DIR=.
  fi
  shopt -s nullglob
  source_shards=("$SOURCE_DIR"/model_world_size_*_rank_*.pt)
  shopt -u nullglob
fi

(( ${#source_shards[@]} > 0 )) || {
  echo "No model_world_size_*_rank_*.pt files found for: $SOURCE_SPEC" >&2
  exit 1
}

local_shards=()
for source_shard in "${source_shards[@]}"; do
  local_shard="$WORK/model_shards/$(basename "$source_shard")"
  if [[ "$source_shard" == az://* ]]; then
    bbb cp "$source_shard" "$local_shard"
  else
    cp "$source_shard" "$local_shard"
  fi
  local_shards+=("$local_shard")
done

"${PYTHON:-python}" "$SCRIPT_DIR/extract_lora_weights.py" \
  "${local_shards[@]}" --output "$WORK/$LORA"

cd "$WORK"
md5sum "$LORA" > "$STEM.md5"
(
  cd splits
  zip -s 48m -0 -q "$STEM.zip" "../$LORA"
)
shopt -s nullglob
split_parts=("$WORK"/splits/*)
shopt -u nullglob
(( ${#split_parts[@]} > 0 )) || {
  echo "No split files were created" >&2
  exit 1
}
md5sum "${split_parts[@]}" | sed "s|$WORK/splits/||" > splits.md5

SPLIT_URI="$ROOT_URI/${STEM}_split"
bbb rmtree "$SPLIT_URI/" >/dev/null 2>&1 || true
bbb cp "$WORK/$LORA" "$ROOT_URI/$LORA"
for part in "${split_parts[@]}"; do
  bbb cp "$part" "$SPLIT_URI/$(basename "$part")"
done
bbb cp "$WORK/$STEM.md5" "$SPLIT_URI/$STEM.md5"
bbb cp "$WORK/splits.md5" "$SPLIT_URI/splits.md5"

echo "Prepared $ROOT_URI/$LORA and $ROOT_URI/${STEM}_split/"
rm -rf "$WORK"

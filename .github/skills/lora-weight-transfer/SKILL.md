---
name: lora-weight-transfer
description: "Fast-download an existing LoRA changed-weight artifact from a verl global_step checkpoint root. Prefer its lora_weights_split parts, verify MD5, and reconstruct lora_weights.pt locally without reading qwen_hf merged weights. Use when: transfer LoRA weights, download lora_weights.pt, download checkpoint LoRA weights, or fast-download LoRA changed tensors."
argument-hint: "<global_step_blob> <local_dest_dir> [delta_name]"
---

# LoRA Weight Transfer

Download the existing LoRA changed-weight file from a `global_step_*`
checkpoint root. Do not read, diff, or download `qwen_hf/model.safetensors`.

Example source:

```text
az://orngwus2cresco/data/boren/outputs/ver_2607/remax_2607v1_openml_verb_s100_bs256_lid/global_step_100
```

This checkpoint already contains:

```text
lora_weights.pt
lora_weights_split/lora_weights.z01 ... lora_weights.zip
lora_weights_split/lora_weights.md5
lora_weights_split/splits.md5
```

## Transfer

```bash
set -euo pipefail

ROOT=<global_step_blob_without_trailing_slash>
DST=<local_dest_dir>
LORA=lora_weights.pt
STEM=${LORA%.pt}
SPLIT="$ROOT/${STEM}_split"

mkdir -p "$DST/splits"
cd "$DST"

if bbb ls "$SPLIT/$STEM.md5" >/dev/null 2>&1; then
  bbb cp "$SPLIT/$STEM.md5" "$STEM.md5"
  bbb cp "$SPLIT/splits.md5" splits/splits.md5
  bbb ls "$SPLIT/" | grep -E "$STEM\.(z[0-9]+|zip)$" > parts.list
  xargs -a parts.list -P 4 -I {} bbb cp -q "{}" splits/

  (cd splits && md5sum -c splits.md5)
  (cd splits && cat $(ls "$STEM".z[0-9][0-9] | sort) "$STEM.zip") > joined.zip
  unzip -p joined.zip > "$LORA" 2>/dev/null || true
  md5sum -c "$STEM.md5"
  rm -rf splits joined.zip parts.list
else
  rmdir splits
  bbb cp "$ROOT/$LORA" "$LORA"
fi

ls -lh "$DST/$LORA"
```

Always concatenate `.z01`, `.z02`, and later parts in order, followed by
`.zip`. Do not use `zip -s 0 --out`, `zip -F`, or direct unzip of the final
split part.

## Verify

```bash
python - <<'PY'
from pathlib import Path
import torch

path = Path("<local_dest_dir>/lora_weights.pt")
weights = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
assert isinstance(weights, dict) and weights, "empty LoRA weight artifact"
print(f"{path}: {len(weights)} tensors, {path.stat().st_size / 2**30:.2f} GiB")
PY
```

The output is the existing changed-tensor artifact, not a PEFT
`adapter_model.safetensors` and not a reconstructed full checkpoint.

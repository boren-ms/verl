---
name: lora-weight-transfer
description: "Extract and transfer only LoRA-changed tensors from a merged verl qwen_hf checkpoint. Compare model.safetensors with its baseline on an in-region Brix node, save the changed tensors as lora_weights.safetensors, and fast-download that file locally without transferring or reconstructing the full model. Use when: transfer LoRA weights, download LoRA changes from qwen_hf, extract LoRA-only weights, or avoid downloading a full merged checkpoint."
argument-hint: "<new_checkpoint_blob> <baseline_checkpoint_or_qwen_hf_blob> <local_dest_dir> [remote_node] [output_name]"
---

# Transfer Only LoRA Weights from a Merged HF Checkpoint

Extract the tensors changed by LoRA training from a merged verl `qwen_hf/`
checkpoint, then transfer only those tensors to the local machine.

The source `qwen_hf/` directory normally contains a full
`model.safetensors`, not PEFT `lora_A`/`lora_B` factors. Therefore the output
contains the complete **new values of only the tensors that differ from the
baseline**. It is not a standalone PEFT adapter and must not be described as
`adapter_model.safetensors`.

Do not download or reconstruct the full merged model unless the user
explicitly asks for recovery.

## Canonical Example

```text
az://orngwus2cresco/data/boren/outputs/ver_2607/remax_2607v1_openml_verb_s100_bs256_lid/global_step_100/
```

Its merged weights are:

```text
az://orngwus2cresco/data/boren/outputs/ver_2607/remax_2607v1_openml_verb_s100_bs256_lid/global_step_100/qwen_hf/model.safetensors
```

## Arguments

| Argument | Description | Default |
|---|---|---|
| `NEW_CHECKPOINT_BLOB` | New `global_step_*` checkpoint root; a direct `qwen_hf/` path is also accepted | Required |
| `BASE_CHECKPOINT_BLOB` | Baseline checkpoint root or direct `qwen_hf/` path | Required |
| `LOCAL_DEST` | Local directory that receives the LoRA-only file | Required |
| `NODE` | Ready in-region Brix node | Pick a Ready `verl-n1-*` |
| `OUTPUT_NAME` | Changed-tensor output filename | `lora_weights.safetensors` |

Accept paths with or without a trailing slash. For a checkpoint root, append
`qwen_hf/` when locating the merged model. Both resolved HF directories must
contain a single `model.safetensors` with compatible keys and tensor shapes.

## Prerequisites

- `scripts/ckpt_delta.py` exists in the local verl repository and supports
  single-file safetensors.
- The repository is pushed to the selected Brix node with `bpush`.
- The node is in the same region as `orngwus2cresco`, has at least 40 GB of
  free disk, and has enough CPU RAM to load both full models.
- Remote blob operations use `az storage blob --auth-mode login`; never use
  `bbb` on Brix nodes.

## Procedure

### 1. Normalize paths and verify both HF checkpoints

Strip `az://orngwus2cresco/data/` and trailing slashes to obtain names under
the `data` container. Resolve either accepted input shape:

```bash
NEW_INPUT=<new_checkpoint_path_under_data_container>
BASE_INPUT=<baseline_checkpoint_or_hf_path_under_data_container>

case "${NEW_INPUT%/}" in
  */qwen_hf) NEW_HF="${NEW_INPUT%/}"; NEW_ROOT="${NEW_INPUT%/qwen_hf}" ;;
  *) NEW_ROOT="${NEW_INPUT%/}/"; NEW_HF="${NEW_INPUT%/}/qwen_hf" ;;
esac
case "${BASE_INPUT%/}" in
  */qwen_hf) BASE_HF="${BASE_INPUT%/}" ;;
  *) BASE_HF="${BASE_INPUT%/}/qwen_hf" ;;
esac

NEW_MODEL="$NEW_HF/model.safetensors"
BASE_MODEL="$BASE_HF/model.safetensors"
```

Verify both files before starting expensive downloads:

```bash
bbb ls "az://orngwus2cresco/data/$NEW_MODEL"
bbb ls "az://orngwus2cresco/data/$BASE_MODEL"
```

First check `<checkpoint_root>/actor/hf/lora_adapter/`,
`<checkpoint_root>/actor/lora_adapter/`, and
`<checkpoint_root>/lora_adapter/`. If any contains both
`adapter_model.safetensors` and `adapter_config.json`, skip the diff workflow
and copy only those two files directly. Otherwise resolve
`<checkpoint_root>/qwen_hf/model.safetensors` and use the changed-tensor
workflow below.

For a native adapter, copy only:

```bash
ADAPTER_SRC=az://orngwus2cresco/data/<resolved_adapter_path_under_container>
mkdir -p <LOCAL_DEST>
bbb cp "$ADAPTER_SRC/adapter_model.safetensors" \
  "<LOCAL_DEST>/adapter_model.safetensors"
bbb cp "$ADAPTER_SRC/adapter_config.json" \
  "<LOCAL_DEST>/adapter_config.json"
```

Verify both local files and finish; do not run the baseline diff.

### 2. Select and prepare an in-region node

```bash
brix ls 'verl-n1-*' 2>&1
bpush <NODE>
brix ls '<NODE>' 2>&1
```

Use a Ready node. Resume a paused node if necessary and wait until it is
Ready.

### 3. Download both merged models remotely and extract changed tensors

Run the downloads and diff in one remote command:

```bash
brix ssh <NODE> -- 'bash -l -c "
set -euo pipefail
WORK=/tmp/ckpt_lora_transfer
ACCT=orngwus2cresco
CONT=data
NEW_MODEL=<new_checkpoint_path_under_data_container>/qwen_hf/model.safetensors
BASE_MODEL=<baseline_resolved_qwen_hf_path_under_data_container>/model.safetensors
OUTPUT=lora_weights.safetensors

rm -rf \$WORK
mkdir -p \$WORK

echo \"=== Download new merged model ===\"
az storage blob download --auth-mode login --account-name \$ACCT --container-name \$CONT \
  --name \"\$NEW_MODEL\" --file \$WORK/new.safetensors \
  --max-connections 32 --no-progress -o none

echo \"=== Download baseline merged model ===\"
az storage blob download --auth-mode login --account-name \$ACCT --container-name \$CONT \
  --name \"\$BASE_MODEL\" --file \$WORK/base.safetensors \
  --max-connections 32 --no-progress -o none

echo \"=== Extract only changed tensors ===\"
cd /root/code/verl
python scripts/ckpt_delta.py diff \
  --new \$WORK/new.safetensors \
  --baseline \$WORK/base.safetensors \
  --delta \$WORK/\$OUTPUT

ls -lh \$WORK/\$OUTPUT
"'
```

Review the diff summary. A typical LoRA run changes only the configured target
modules. Stop if most model tensors changed, keys are missing unexpectedly, or
there are shape mismatches; that usually means the wrong baseline was chosen.

### 4. Upload the LoRA-only file beside the source checkpoint

```bash
brix ssh <NODE> -- 'bash -l -c "
set -euo pipefail
NEW_ROOT=<new_checkpoint_path_under_data_container>
OUTPUT=lora_weights.safetensors

az storage blob upload --auth-mode login \
  --account-name orngwus2cresco --container-name data \
  --name \"\${NEW_ROOT%/}/\$OUTPUT\" \
  --file /tmp/ckpt_lora_transfer/\$OUTPUT \
  --overwrite --no-progress -o none
echo UPLOAD_DONE
"'
```

The resulting blob for the canonical example is:

```text
az://orngwus2cresco/data/boren/outputs/ver_2607/remax_2607v1_openml_verb_s100_bs256_lid/global_step_100/lora_weights.safetensors
```

### 5. Fast-download only the LoRA weights

If the output is small enough for a normal copy, use:

```bash
mkdir -p <LOCAL_DEST>
bbb cp \
  "az://orngwus2cresco/data/<new_checkpoint_path_under_data_container>/lora_weights.safetensors" \
  "<LOCAL_DEST>/lora_weights.safetensors"
```

For a large output, split it into 48 MB zip parts on the same node:

```bash
brix ssh <NODE> -- 'bash -l -c "
set -euo pipefail
WORK=/tmp/ckpt_lora_transfer
NEW_ROOT=<new_checkpoint_path_under_data_container>
OUTPUT=lora_weights.safetensors
STEM=lora_weights
SPLIT_DIR=\${NEW_ROOT%/}/lora_weights_split

cd \$WORK
md5sum \$OUTPUT | tee \$STEM.md5
rm -rf splits
mkdir splits
cd splits
zip -s 48m -0 -q \$STEM.zip ../\$OUTPUT
cd ..
md5sum splits/* | sed \"s|splits/||\" > splits.md5

az storage blob delete-batch --auth-mode login --account-name orngwus2cresco \
  --source data --pattern \"\$SPLIT_DIR/*\" -o none 2>/dev/null || true
az storage blob upload-batch --auth-mode login --account-name orngwus2cresco \
  --destination data --destination-path \"\$SPLIT_DIR\" --source splits \
  --max-connections 16 --overwrite -o none
az storage blob upload --auth-mode login --account-name orngwus2cresco \
  --container-name data --name \"\$SPLIT_DIR/splits.md5\" \
  --file splits.md5 --overwrite -o none
az storage blob upload --auth-mode login --account-name orngwus2cresco \
  --container-name data --name \"\$SPLIT_DIR/\$STEM.md5\" \
  --file \$STEM.md5 --overwrite -o none
echo SPLIT_UPLOAD_DONE
"'
```

Download and reassemble locally:

```bash
DST=<LOCAL_DEST>
SRC=az://orngwus2cresco/data/<new_checkpoint_path_under_data_container>/lora_weights_split
STEM=lora_weights
OUTPUT=lora_weights.safetensors

mkdir -p "$DST/splits"
cd "$DST"
bbb cp "$SRC/$STEM.md5" "$STEM.md5"
bbb cp "$SRC/splits.md5" splits/splits.md5
bbb ls "$SRC/" | grep -E "$STEM\.(z[0-9]+|zip)$" > parts.list
xargs -a parts.list -P 4 -I {} bbb cp -q "{}" splits/
(cd splits && md5sum -c splits.md5)
(cd splits && cat $(ls "$STEM".z[0-9][0-9] 2>/dev/null | sort) "$STEM.zip") > joined.zip
unzip -p joined.zip > "$OUTPUT" 2>/dev/null || true
md5sum -c "$STEM.md5"
rm -rf splits joined.zip parts.list
ls -lh "$DST/$OUTPUT"
```

Always concatenate `.z01`, `.z02`, and later parts in order, followed by
`.zip`. Do not use `zip -s 0 --out`, `zip -F`, or direct unzip of the final
split part.

### 6. Verify the transferred artifact

Inspect the safetensors keys without loading tensor data:

```bash
python - <<'PY'
from pathlib import Path
from safetensors import safe_open

path = Path("<LOCAL_DEST>/lora_weights.safetensors")
with safe_open(path, framework="pt", device="cpu") as f:
    keys = list(f.keys())

assert keys, "LoRA weights file contains no changed tensors"
print(f"{path}: {len(keys)} changed tensors")
print("\n".join(keys[:20]))
PY
```

Confirm that the key count matches the remote `ckpt_delta.py diff` summary.
The local destination should contain `lora_weights.safetensors`, not the full
`model.safetensors`.

### 7. Cleanup

```bash
brix ssh <NODE> -- 'rm -rf /tmp/ckpt_lora_transfer'
```

Keep the uploaded `lora_weights.safetensors` by default. Remove the temporary
`lora_weights_split/` blob directory only after the local MD5 check succeeds.

## Output Semantics

- `lora_weights.safetensors` contains only changed tensors.
- Each value is the full trained tensor value, not a numerical delta.
- Keys follow baseline naming after normalization by `ckpt_delta.py`.
- The artifact excludes tokenizer, config, processor, Python modeling files,
  unchanged base weights, optimizer state, and trainer state.
- The artifact is intended for transfer, inspection, archival, or later
  baseline-based recovery. It is not directly loadable as a PEFT adapter.

## Gotchas

- The baseline must be the exact model from which the run started.
- Do not label extracted merged tensors as `adapter_model.safetensors`.
- Do not transfer the full `qwen_hf/` directory for a LoRA-only request.
- Both full `model.safetensors` files are downloaded only inside the in-region
  node; only the changed-tensor artifact crosses the WAN.
- `ckpt_delta.py diff` loads both full models into CPU memory.

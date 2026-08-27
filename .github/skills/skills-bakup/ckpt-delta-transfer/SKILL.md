---
name: ckpt-delta-transfer
description: "Compute a checkpoint delta between a new (verl RL) and baseline (fastllm) checkpoint on a remote brix node using ckpt_delta.py, fast-download the delta to local via blob-split, and reconstruct the full checkpoint locally with apply. Use when: transferring RL checkpoints efficiently, diffing verl checkpoints against baseline, creating delta files for checkpoint archival, recovering full checkpoints from baseline + delta. Triggers: 'ckpt delta', 'checkpoint diff', 'diff and recover', 'delta transfer', 'ckpt_delta diff', 'ckpt_delta apply'."
argument-hint: "<new_blob_path> <baseline_blob_path> <local_dest_dir> [remote_node] [delta_name]"
---

# Checkpoint Delta: Remote Diff + Fast-Download + Local Recovery

Compute a compact delta between a verl RL checkpoint and a baseline checkpoint on a remote brix node (where blob I/O is fast), then fast-download only the delta to local and reconstruct the full checkpoint from baseline + delta.

This avoids downloading the full ~18GB checkpoint over the WAN — only the delta (~600MB for ~40 changed LoRA-target tensors) needs to cross the network.

## When to Use

- A new verl RL checkpoint exists on orng blob and you need it locally.
- You already have (or can obtain) the baseline checkpoint locally.
- Direct `bbb cp` of the full ~18GB checkpoint is too slow.
- You want to archive/share checkpoints efficiently via deltas.

## Prerequisites

- `scripts/ckpt_delta.py` in the verl repo (pushed to the remote node).
- A Ready brix node in `prod-westus2-cw-*` (same region as orng blob).
- Baseline checkpoint available both on blob (for remote diff) and locally (for local apply).
- Both checkpoints must be single `.pt` files (typically `lora_merged_model_states.pt`).

## Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `NEW_BLOB` | Blob path to the new (verl RL) checkpoint directory | `az://orngwus2cresco/data/boren/outputs/.../global_step_113` |
| `BASE_BLOB` | Blob path to the baseline (fastllm) checkpoint directory | `az://orngwus2cresco/data/speech/projects/.../90000` |
| `LOCAL_DEST` | Local directory for the recovered checkpoint | `~/data/ckp/<run>/<step>` |
| `NODE` | Remote brix node name (default: pick any Ready `verl-n1-*`) | `verl-n1-i1` |
| `DELTA_NAME` | Name for the delta file (default: `delta_v1.pt`) | `delta_v1.pt` |
| `CKPT_FILE` | Checkpoint filename inside each directory (default: `lora_merged_model_states.pt`) | `lora_merged_model_states.pt` |
| `BASE_LOCAL` | Local path to baseline checkpoint (for apply step) | `~/data/ckp/fast-llm-.../90000/lora_merged_model_states.pt` |
| `BASE_HF_PATH` | Blob path to baseline HF model dir (has `config.json`) | `speech/projects/phi-fastllm-2605/amlt-results/.../90000/qwen_hf/` |

## Procedure

### Phase 1: Remote Diff

#### Step 1 — Push code + verify node

```bash
bpush <NODE>
brix ls '<NODE>' 2>&1
```

Confirm node is Ready. If paused, `brix resume <NODE>` and poll until Ready.

#### Step 2 — Download both checkpoints + run diff on remote

Run this as a single `brix ssh` command. All blob operations use `az storage blob --auth-mode login` (brix nodes have no `bbb` credentials or IMDS).

```bash
brix ssh <NODE> -- 'bash -l -c "
set -euo pipefail
WORK=/tmp/ckpt_delta
ACCT=orngwus2cresco
CONT=data
NEW_BLOB=<new_blob_path_under_container>/<CKPT_FILE>
BASE_BLOB=<base_blob_path_under_container>/<CKPT_FILE>

mkdir -p \$WORK && cd \$WORK

echo \"=== Download NEW checkpoint ===\"
az storage blob download --auth-mode login --account-name \$ACCT --container-name \$CONT \
  --name \"\$NEW_BLOB\" --file \$WORK/new.pt --max-connections 32 --no-progress -o none
ls -lh \$WORK/new.pt

echo \"=== Download BASELINE checkpoint ===\"
az storage blob download --auth-mode login --account-name \$ACCT --container-name \$CONT \
  --name \"\$BASE_BLOB\" --file \$WORK/base.pt --max-connections 32 --no-progress -o none
ls -lh \$WORK/base.pt

echo \"=== Run ckpt_delta diff ===\"
cd /root/code/verl
python scripts/ckpt_delta.py diff \
  --new \$WORK/new.pt \
  --baseline \$WORK/base.pt \
  --delta \$WORK/<DELTA_NAME>
ls -lh \$WORK/<DELTA_NAME>
"'
```

**Expected output**: diff stats showing ~40 changed tensors, delta file much smaller than full checkpoint (~600MB vs ~18GB).

#### Step 3 — Upload delta to blob

```bash
brix ssh <NODE> -- 'bash -l -c "
az storage blob upload --auth-mode login --account-name orngwus2cresco --container-name data \
  --name \"<new_blob_path_under_container>/<DELTA_NAME>\" \
  --file /tmp/ckpt_delta/<DELTA_NAME> --overwrite -o none
echo UPLOAD_DONE
"'
```

### Phase 2: Fast-Download Delta to Local (blob-split)

#### Step 4 — Remote: md5 + zip split + upload parts

On the same remote node, split the delta into ≤48MB parts and upload to blob:

```bash
brix ssh <NODE> -- 'bash -l -c "
set -euo pipefail
WORK=/tmp/ckpt_delta
ACCT=orngwus2cresco
CONT=data
SPLIT_DIR=<new_blob_path_under_container>/<DELTA_NAME_no_ext>_split

cd \$WORK
md5sum <DELTA_NAME> | tee <DELTA_NAME_no_ext>.md5
rm -rf splits && mkdir splits && cd splits
zip -s 48m -0 -q <DELTA_NAME_no_ext>.zip ../\$WORK/<DELTA_NAME>
cd ..
md5sum splits/* | sed \"s|splits/||\" > splits.md5
ls -lh splits/

echo \"=== Upload splits ===\"
az storage blob delete-batch --auth-mode login --account-name \$ACCT --source \$CONT \
  --pattern \"\$SPLIT_DIR/*\" -o none 2>/dev/null || true
az storage blob upload-batch --auth-mode login --account-name \$ACCT --destination \$CONT \
  --destination-path \"\$SPLIT_DIR\" --source splits --max-connections 16 --overwrite -o none
az storage blob upload --auth-mode login --account-name \$ACCT --container-name \$CONT \
  --name \"\$SPLIT_DIR/splits.md5\" --file splits.md5 --overwrite -o none
az storage blob upload --auth-mode login --account-name \$ACCT --container-name \$CONT \
  --name \"\$SPLIT_DIR/<DELTA_NAME_no_ext>.md5\" --file <DELTA_NAME_no_ext>.md5 --overwrite -o none
echo SPLIT_UPLOAD_DONE
"'
```

Note the split naming: `zip -s` produces `<name>.z01`, `<name>.z02`, ..., `<name>.zip` (the `.zip` file contains the central directory and is the last piece).

#### Step 5 — Local: parallel download parts + reassemble

```bash
DST=<LOCAL_DEST>
SRC=az://orngwus2cresco/data/<new_blob_path_under_container>/<DELTA_NAME_no_ext>_split
DELTA=<DELTA_NAME>
STEM=<DELTA_NAME_no_ext>

mkdir -p "$DST/splits" && cd "$DST"

# Download md5 files
bbb cp "$SRC/${STEM}.md5" "${STEM}.md5"
bbb cp "$SRC/splits.md5" splits/splits.md5

# List and download parts in parallel (P=4 is reliable; P=16 can overwhelm local bandwidth)
bbb ls "$SRC/" | grep -E "${STEM}\.(z[0-9]+|zip)$" > parts.list
N=$(wc -l < parts.list); echo "$N parts to download"

t0=$SECONDS
xargs -a parts.list -P 4 -I {} bbb cp -q "{}" splits/
echo "Download took $((SECONDS - t0))s"

# Verify part checksums
(cd splits && md5sum -c splits.md5)

# Reassemble: cat split parts in order, then unzip
(cd splits && cat $(ls ${STEM}.z[0-9][0-9] 2>/dev/null | sort) ${STEM}.zip) > joined.zip
unzip -p joined.zip > "$DELTA" 2>/dev/null || true

# Verify final file
md5sum -c "${STEM}.md5" && echo "MD5 OK"

# Cleanup
rm -rf splits joined.zip parts.list
ls -lh "$DST/$DELTA"
```

**Critical**: Always `cat` the parts in order (z01, z02, ..., then .zip) before `unzip -p`. Do NOT use `zip -s 0 --out` or `zip -F` — they only rebuild metadata and produce bogus output.

### Phase 3: Local Recovery (apply)

#### Step 6 — Check baseline availability

The baseline checkpoint must be available locally. Check:

```bash
ls -lh <BASE_LOCAL>
```

If not available, download it:
```bash
mkdir -p $(dirname <BASE_LOCAL>)
bbb cp az://orngwus2cresco/data/<base_blob_path_under_container>/<CKPT_FILE> <BASE_LOCAL>
```

For large baselines, consider using the `blob-split-fast-download` skill.

#### Step 7 — Run apply to reconstruct

```bash
cd ~/code/verl
python scripts/ckpt_delta.py apply \
  --baseline <BASE_LOCAL> \
  --delta <LOCAL_DEST>/<DELTA_NAME> \
  --new <LOCAL_DEST>/mp_rank_00_model_states.pt
```

This produces a checkpoint identical to the original new checkpoint:
- Same flat key naming (no `.base_layer.` infix)
- Same `{"module": OrderedDict[...]}` wrapping
- Same tensor values
- Named `mp_rank_00_model_states.pt` for compatibility with the vLLM/eval loading conventions

#### Step 7b — Copy config.json from local baseline model folder

The eval/vLLM loader expects `config.json` alongside the weights. Copy it from the local baseline folder:

```bash
cp $(dirname <BASE_LOCAL>)/config.json <LOCAL_DEST>/config.json
```

If `config.json` is not in the local baseline folder yet, download it first:
```bash
bbb cp "az://orngwus2cresco/data/<base_hf_path>/config.json" $(dirname <BASE_LOCAL>)/config.json
cp $(dirname <BASE_LOCAL>)/config.json <LOCAL_DEST>/config.json
```

#### Step 8 — Verify (optional)

If you have the original full checkpoint locally for comparison:

```bash
python -c "
import torch, sys
a = torch.load('<LOCAL_DEST>/original_<CKPT_FILE>', map_location='cpu')
b = torch.load('<LOCAL_DEST>/<CKPT_FILE>', map_location='cpu')
a_sd = a['module'] if 'module' in a else a
b_sd = b['module'] if 'module' in b else b
assert set(a_sd.keys()) == set(b_sd.keys()), 'Key mismatch'
mismatches = []
for k in a_sd:
    if not torch.equal(a_sd[k], b_sd[k]):
        mismatches.append(k)
if mismatches:
    print(f'FAIL: {len(mismatches)} tensors differ: {mismatches[:5]}')
    sys.exit(1)
print(f'OK: all {len(a_sd)} tensors match')
"
```

### Step 9 — Cleanup

```bash
# Remote: remove tmp files
brix ssh <NODE> -- 'rm -rf /tmp/ckpt_delta'

# Blob: optionally remove split directory
# bbb rmtree az://orngwus2cresco/data/<new_blob_path_under_container>/<DELTA_NAME_no_ext>_split/
```

## Key Details

### Checkpoint layouts

| Checkpoint | Wrapping | Key naming | Typical size |
|-----------|----------|------------|-------------|
| Baseline (fastllm merged) | `{"module": OrderedDict}` | `model.layers.{i}.<mod>.base_layer.weight` | ~18GB |
| New (verl RL merged) | `{"module": OrderedDict}` | `model.layers.{i}.<mod>.weight` (flat) | ~18GB |
| Delta | Plain `OrderedDict` | Baseline key naming (`.base_layer.`) | ~600MB |

### What the delta contains

- Only the tensor **values** from the new checkpoint that differ from baseline (NOT numerical differences).
- Uses baseline key naming so `apply` can match keys directly.
- Typically ~40 tensors for LoRA-target layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj across all transformer layers).

### Remote node requirements

- Must be in-region with the blob (e.g., `prod-westus2-cw-*` for `orngwus2cresco`).
- Use `az storage blob --auth-mode login` for all blob operations (no `bbb` creds, no IMDS on brix pods).
- Needs `torch` installed (standard on verl nodes).
- Work directory: `/tmp/ckpt_delta` (needs ~40GB free for both checkpoints + delta).

## Gotchas

- **Do not** use `bbb` on remote brix nodes for blob operations — it has no credentials. Use `az storage blob --auth-mode login`.
- **Do not** reassemble zip splits with `zip -s 0 --out`, `zip -F`, or `unzip` directly on the `.zip` file — always `cat` parts in order first.
- `unzip -p` will print warnings like "extra bytes at beginning" — these are expected for multi-part zips. Verify via md5.
- The `apply` output will have **flat key naming** (no `.base_layer.`) matching the new checkpoint format — this is correct and expected.
- If `/tmp` on the remote node doesn't have enough space (~40GB), use a different work directory.
- Both `diff` and `apply` load full checkpoints into CPU memory — the node needs ~40GB+ RAM available.

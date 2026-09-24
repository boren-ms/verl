---
name: ckpt-delta-transfer
description: 'Transfer a complete merged checkpoint efficiently by computing a changed-tensor delta against an exact baseline on Brix, downloading the delta, and reconstructing the full checkpoint locally with ckpt_delta.py. Use for checkpoint diff/apply, delta archival, or full-checkpoint recovery; not PEFT LoRA A/B extraction.'
argument-hint: '<new_blob_path> <baseline_blob_path> <local_dest_dir> [remote_node] [delta_name]'
---

# Checkpoint Delta Transfer

Use [scripts/ckpt_delta.py](../../../scripts/ckpt_delta.py) for a full-state
changed-tensor delta. A delta stores replacement tensor values, **not**
numerical differences or LoRA adapters. For actual A/B tensors use
[lora-weight-transfer](../lora-weight-transfer/SKILL.md); for HF export use
[verl-asr-run](../verl-asr-run/SKILL.md).

Read the [shared remote execution contract](../references/remote-execution.md).
Use an available in-region Brix node with enough CPU RAM and disk for both
full checkpoints plus the delta. Approximately 40 GB RAM/disk was sufficient
for some 18 GB checkpoints; measure the actual inputs and allow working space.
Delta size and changed-tensor count are model-dependent, not fixed guarantees.

## Inputs

| Input | Meaning |
|---|---|
| `NEW_BLOB`, `BASE_BLOB` | Direct checkpoint paths, or directories plus `CKPT_FILE`. |
| `CKPT_FILE` | Default `lora_merged_model_states.pt`; both inputs must be complete merged tensor checkpoints, not FSDP actor shards. |
| `BASE_LOCAL` | Exact same baseline locally for apply; verify its checksum against the remote diff input. |
| `LOCAL_DEST` | New local output directory, normally `~/data/ckp/<run>/<step>`. |
| `NODE` | User-specified or verified available in-region Ready pool. |
| `DELTA_NAME` | Filesystem-safe name, default `delta_v1.pt`. |
| `BASE_HF_PATH` | Compatible HF metadata source if downstream loading requires it. |

Resolve full `az://` paths before transfer. Verify source/destination identity
and do not overwrite unrelated artifacts.

## 1. Stage and Validate Inputs Remotely

Verify/sync the intended workspace. Use a unique per-run remote staging
directory, record its resolved path, and use it consistently across commands.

Use authenticated `bbb` when available. If it is unavailable on the selected
node, use Azure CLI with a verified tenant/account instead; authentication
capability is node-specific, not a universal Brix limitation. For example:

```bash
az storage blob download --auth-mode login \
  --account-name orngwus2cresco --container-name data \
  --name '<checkpoint-path-under-container>' \
  --file '<REMOTE_WORK>/new.pt' --max-connections 32 --no-progress -o none
```

Stage the baseline similarly as `<REMOTE_WORK>/base.pt`. Before diff:

- Normalize `.base_layer.` segments to `.` and reject key collisions.
- Require matching normalized tensor key sets, shapes, and dtypes.
- Reject non-finite tensors and unsupported non-tensor payloads.
- Record input checksums and model/config provenance.

These checks are required because the diff helper reports but skips missing
keys and shape mismatches. A zero exit code alone does not establish lossless
reconstruction.

## 2. Compute and Round-Trip the Delta

Run on the remote node from the repository root:

```bash
python scripts/ckpt_delta.py diff \
  --new '<REMOTE_WORK>/new.pt' \
  --baseline '<REMOTE_WORK>/base.pt' \
  --delta '<REMOTE_WORK>/delta_v1.pt' --tol 0
python scripts/ckpt_delta.py apply \
  --baseline '<REMOTE_WORK>/base.pt' \
  --delta '<REMOTE_WORK>/delta_v1.pt' \
  --new '<REMOTE_WORK>/recovered.pt'
```

Require zero skipped shape mismatches and zero keys unique to either input.
Compare the recovered state with the original new state, normalizing keys and
unwrapping `module` on either side: all keys, shapes, dtypes, and tensor values
must match exactly. `torch.equal` per tensor is appropriate after dtype checks.
Do not claim byte-identical PyTorch serialization; the invariant is tensor
identity. Only publish after the round trip passes.

Upload the delta, its checksum, and provenance containing both input hashes
and the successful verification result to a unique prefix below the new
checkpoint. Verify remote readability and byte size.

## 3. Download the Delta

Invoke `blob-split-fast-download` for the published delta when parallel split
transfer is needed. That skill owns splitting, ordered reassembly, checksum
verification, and temporary-part cleanup; do not maintain a second copy of
those commands here. For a small delta use a normal authenticated copy.

Require the local delta checksum to match the remote artifact. If the baseline
is not local, obtain it through the same transfer owner and verify its checksum
against the exact baseline used by remote diff.

## 4. Reconstruct Locally

From the local repository root:

```bash
python scripts/ckpt_delta.py apply \
  --baseline '<BASE_LOCAL>' \
  --delta '<LOCAL_DEST>/delta_v1.pt' \
  --new '<LOCAL_DEST>/mp_rank_00_model_states.pt'
```

The `.pt` output has a `module`-wrapped state dict with flat keys (no
`.base_layer.` segments). Verify it loads, has the validated normalized
key/shape/dtype set, and uses the checksummed baseline/delta pair. If the
original new checkpoint is local, also compare every tensor directly.

If the intended loader requires `config.json`, copy it from the matching base
model only after checking model compatibility. A single config file does not
turn this full-state `.pt` into a complete HF export; use the job runner's
export procedure when an HF directory is required.

## 5. Delivery and Cleanup

Report the local recovered checkpoint, delta, baseline identity, verification
results, and measured transfer savings. Remove only this run's explicitly
resolved temporary files after verification. Do not delete shared staging
directories, wildcard blob prefixes, source checkpoints, or archival deltas.
Retain failed artifacts/logs for diagnosis and report failures explicitly.

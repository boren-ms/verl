---
name: lora-weight-transfer
description: "Transfer actual PEFT LoRA A/B tensors from a complete verl model_world_size_*_rank_*.pt checkpoint set, merge them into the matching full Qwen3.5-audio baseline while preserving baseline MTP layers, and upload both outputs to corp Blob. Gather FSDP shards into lora_weights.pt, transfer splits with bbb, produce lora_merged_model_states.pt, then delegate local-to-corp uploads to /copy-to-corp-blob. Use when: transfer LoRA weights, reconstruct model state dictionaries with MTP, build a LoRA-merged checkpoint, or publish both artifacts to tsstd01safn."
argument-hint: "<remote_checkpoint_dir> <publish_blob_root> <local_dest_dir> [remote_node] [baseline]"
---

# LoRA Weight Transfer

Prepare and download the actual trained LoRA adapter matrices:

```text
*.lora_A.weight
*.lora_B.weight
```

`lora_weights.pt` is a PyTorch dictionary containing only these adapter
tensors. It is not a merged checkpoint, a changed-base-weight delta, or a PEFT
directory.

## Source of Truth

Use the complete actor model shard set:

```text
<global_step>/actor/model_world_size_<N>_rank_*.pt
```

All ranks from `0` through `N-1` are required. The extractor gathers DTensor
and legacy ShardedTensor values according to their shard metadata, and
concatenates ordinary FSDP tensor shards in rank order. Never use
`extra_state_world_size_*.pt`: those files contain only scheduler and RNG
state. Also never use optimizer state, merged model weights, or checkpoint
deltas. The merged update is proportional to `B @ A`; the original A/B
factorization cannot be uniquely recovered from that product.

## Inputs

| Input | Description | Default |
|---|---|---|
| `SOURCE` | Actor directory or quoted `model_world_size_<N>_rank_*.pt` pattern, local or `az://` | required |
| `ROOT` | Destination `az://` URI where the artifact and splits are published | required |
| `DST` | Local destination directory | required |
| `NODE` | Ready, in-region Brix node | any suitable `verl-n1-*` node |
| `BASELINE` | Matching local LoRA-merged baseline used as the merge base | legacy default below; use the run's actual baseline |
| `MTP_BASELINE` | Original full baseline checkpoint supplying unchanged `mtp.*` tensors if `BASELINE` was inference-pruned | not needed when `BASELINE` already retains all MTP layers |
| `CORP_ROOT` | Corp destination derived from `DST` by `/copy-to-corp-blob` | `https://tsstd01safn.blob.core.windows.net/data/users/boren/data/<DST-relative-to-~/data>` |

Example inputs:

```text
SOURCE=az://orngwus2cresco/data/boren/outputs/.../global_step_100/actor
ROOT=az://orngwus2cresco/data/boren/outputs/.../global_step_100
DST=~/data/ckp/run/global_step_100
NODE=verl-n1-i10
BASELINE=~/data/ckp/fast-llm-2607-qwen3-5-9b-s2-data-v3.4-sr-afteraudio-lexical-fix/50000/lora_merged_model_states.pt
```

Both the remote and local nodes must have an authenticated `bbb` command.
The remote Python environment must contain `torch`; set `PYTHON=/path/to/python`
when `python` is not the correct interpreter.

If the checkpoint and `lora_weights.pt` are already synced locally, reuse
them and start at Phase 4. Do not re-extract adapters or download shards.

## Phase 1: Locate All Model Shards

Resolve the complete shard set directly from the supplied checkpoint directory:

```bash
CHECKPOINT=<remote_checkpoint_dir_without_trailing_slash>
brix ssh <NODE> -- 'ls -lh '"$CHECKPOINT"'/actor/model_world_size_*_rank_*.pt'
```

The source may instead be an `az://` actor directory or quoted wildcard.
`prepare_remote.sh` discovers every matching file with `bbb ls` and downloads
each with `bbb cp`. Do not substitute `extra_state_world_size_*` or
`optim_world_size_*` files, and do not proceed with a partial rank set.

## Phase 2: Prepare and Upload on the Remote Node

Sync the repository so the scripts are available, then confirm `bbb` can read
the destination root:

```bash
rcall-brix sync <NODE>
brix ssh <NODE> -- 'bbb ls <publish_blob_root>'
```

Run [prepare_remote.sh](./scripts/prepare_remote.sh) on the remote node. It
calls [extract_lora_weights.py](./scripts/extract_lora_weights.py), creates
48 MiB zip splits and MD5 manifests, then uploads everything with `bbb cp`:

```bash
brix ssh <NODE> -- 'cd ~/code/verl && bash \
  .github/skills/lora-weight-transfer/scripts/prepare_remote.sh \
  <publish_blob_root> \
  <local_or_az_actor_directory>'
```

To perform extraction only, run the Python script directly:

```bash
python .github/skills/lora-weight-transfer/scripts/extract_lora_weights.py \
  model_world_size_*_rank_*.pt --output <lora_weights.pt>
```

The extractor accepts only LoRA A/B tensor keys and rejects empty or incomplete
pairs, mixed world sizes, missing ranks, and inconsistent shard keys before
writing the output.

## Phase 3: Download, Recover, and Verify Locally

Run [download_recover.sh](./scripts/download_recover.sh). It downloads all
files with `bbb cp`, verifies each split, concatenates numbered parts in version
order followed by the final `.zip`, extracts `lora_weights.pt`, verifies its
MD5, and checks every LoRA A tensor has a matching B tensor:

```bash
bash .github/skills/lora-weight-transfer/scripts/download_recover.sh \
  <publish_blob_root> <local_dest_dir> [parallel_downloads]
```

Do not reconstruct with `zip -s 0 --out`, `zip -F`, or direct unzip of the
final split part. The script uses sequential concatenation because the payload
is distributed across `.z01`, `.z02`, and later parts, with the central
directory in the trailing `.zip` file.

## Phase 4: Merge LoRA into the Full Baseline

Use the full baseline corresponding to the run's `actor_rollout_ref.model.path`,
not an unrelated model. The legacy default is:

```text
~/data/ckp/fast-llm-2607-qwen3-5-9b-s2-data-v3.4-sr-afteraudio-lexical-fix/50000/lora_merged_model_states.pt
```

The legacy training config in `recipe/phimm/config/base/dapo_asr.yaml` sets
`lora_alpha: 640` and `lora_rank: 320`, so the required merge is:

```text
W <- W + (640 / 320) * (B @ A)
```

For other runs, use their actual `lora_alpha` and `lora_rank` from the training
configuration or export provenance. For the 2609r2 step-80 checkpoint
`remax_2609r2_ml_verb_hint_s100_bs128_n8_r256_g32_flr1`, these are 512 and
256 (also scaling 2), and the matching baseline is
`~/data/ckp/fast-llm-2609-qwen9b-flashenc-mtp-s2-data-v3.5.4-r2/54000`.

Run [merge_lora_checkpoint.py](./scripts/merge_lora_checkpoint.py) after local
recovery. Write the new full checkpoint beside `lora_weights.pt` with the
filename `lora_merged_model_states.pt` for a non-MTP model, or
`lora_merged_model_states_mtp.pt` when retaining baseline MTP layers.
Keep any existing untagged non-MTP checkpoint separate. For the legacy model:

```bash
python .github/skills/lora-weight-transfer/scripts/merge_lora_checkpoint.py \
  --baseline ~/data/ckp/fast-llm-2607-qwen3-5-9b-s2-data-v3.4-sr-afteraudio-lexical-fix/50000/lora_merged_model_states.pt \
  --lora <local_dest_dir>/lora_weights.pt \
  --output <local_dest_dir>/lora_merged_model_states.pt \
  --lora-alpha 640 \
  --lora-rank 320
```

The merger follows the key mapping and fp32 merge math in
`plugins/qwen35_audio/src/hf_qwen35_audio/convert_verl_to_pt.py`. It strips
the leading `base_model.model.` from LoRA prefixes, normalizes baseline
`<prefix>.base_layer.weight` keys to `<prefix>.weight`, performs `B @ A` and
the addition in fp32, and casts back to the baseline dtype. The output uses
the reference checkpoint layout with the state dictionary under `module`.
It writes atomically and creates a matching `.md5` sidecar
(`lora_merged_model_states_mtp.md5` for an MTP-tagged output).

**Keep the baseline MTP layers.** Start from the full baseline state dictionary,
not the actor/HF inference key set: inference exports can omit all `mtp.*`
tensors. Never filter those tensors out during model-state reconstruction.
If a previously merged baseline was inference-pruned, supply its original full
checkpoint with `--mtp-baseline`. Only its `mtp.*` tensors are copied, unchanged;
its training LoRA adapters, optimizer state, and other weights are not copied or
merged a second time. Conflicting MTP tensors and RL LoRA updates targeting MTP
are rejected. The merger checks configured MTP layer coverage when a
`config.json` exists beside `--baseline`; also compare every MTP key, shape,
dtype, and value against the original full baseline.
MTP views backed by larger flat training buffers are cloned during serialization
so unrelated backing data is not included; their tensor values and dtypes stay
unchanged.

For the locally synced 2609r2 step-80 run:

```bash
BASE=~/data/ckp/fast-llm-2609-qwen9b-flashenc-mtp-s2-data-v3.5.4-r2/54000
DST=~/data/ckp/remax_2609r2_ml_verb_hint_s100_bs128_n8_r256_g32_flr1/global_step_80
python .github/skills/lora-weight-transfer/scripts/merge_lora_checkpoint.py \
  --baseline "$BASE/lora_merged_model_states.pt" \
  --mtp-baseline "$BASE/mp_rank_00_model_states.pt" \
  --lora "$DST/lora_weights.pt" \
  --output "$DST/lora_merged_model_states_mtp.pt" \
  --lora-alpha 512 --lora-rank 256
```

Verify the output has the normalized baseline key set plus any restored MTP
keys, no LoRA keys, exactly one changed non-MTP baseline weight per LoRA pair,
all original baseline MTP tensors unchanged, and a passing adjacent MD5 file.

## Phase 5: Upload Both Outputs to Corp Blob

After both local artifacts and their MD5 files pass verification, invoke the
`/copy-to-corp-blob` skill separately for each generated `.pt` file:

```text
/copy-to-corp-blob <local_dest_dir>/lora_weights.pt
/copy-to-corp-blob <local_dest_dir>/lora_merged_model_states.pt
```

Also upload their adjacent checksum files:

```text
/copy-to-corp-blob <local_dest_dir>/lora_weights.md5
/copy-to-corp-blob <local_dest_dir>/lora_merged_model_states.md5
```

For an MTP model, upload `lora_merged_model_states_mtp.pt` and
`lora_merged_model_states_mtp.md5` instead of the untagged merged pair.
Do not overwrite an existing remote non-MTP checkpoint with MTP weights.

Do not implement or duplicate SAS handling in this skill. Delegate it to
`/copy-to-corp-blob`, which must validate that the cached corp SAS is unexpired
and contains both write (`w`) and create (`c`) permissions before upload.

For a destination under `~/data`, preserve the default path mapping. Example:

```text
local:
  ~/data/ckp/<run>/<step>/lora_weights.pt
  ~/data/ckp/<run>/<step>/lora_merged_model_states.pt

corp:
  https://tsstd01safn.blob.core.windows.net/data/users/boren/data/ckp/<run>/<step>/lora_weights.pt
  https://tsstd01safn.blob.core.windows.net/data/users/boren/data/ckp/<run>/<step>/lora_merged_model_states.pt
```

Wait for each upload to complete, then use the verification step provided by
`/copy-to-corp-blob` to compare exact remote and local byte sizes. Never expose
the SAS token or a destination URL containing its query string.

## Safeguards

- Accept only keys matching `*.lora_A[.<adapter>].weight` and
  `*.lora_B[.<adapter>].weight`.
- Require every adapter prefix to have both A and B tensors.
- Do not include base-layer, embedding, optimizer, or scheduler tensors.
- Use `bbb` for every blob upload and download.
- Require all LoRA prefixes and `B @ A` shapes to match baseline weights.
- Preserve baseline key order, tensor dtypes, and unrelated values; write plain
  parameter names under the top-level `module` key.
- Preserve every baseline `mtp.*` tensor unchanged, restoring it from the
  original full baseline if an inference-pruned baseline is used. Do not
  reconstruct MTP from actor keys or initialize replacement MTP tensors.
- Use the run's actual `lora_alpha / lora_rank` (legacy default `640 / 320 = 2`);
  do not assume scaling 1.
- Upload both `.pt` outputs and checksum sidecars through `/copy-to-corp-blob`.
- Require exact remote/local byte-size matches after every corp upload.
- Never print or persist a SAS token outside the cache managed by the upload skill.
- Do not use `ckpt_delta.py` or infer A/B matrices from merged model weights.

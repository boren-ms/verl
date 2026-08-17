---
name: lora-weight-transfer
description: "Transfer actual PEFT LoRA A/B tensors from a complete verl model_world_size_*_rank_*.pt checkpoint set on a remote Brix node. Gather FSDP tensor shards into lora_weights.pt, publish split/checksum files to Azure Blob with bbb, download in parallel locally, reconstruct, and verify every lora_A tensor has a matching lora_B tensor. Reject missing ranks and extra-state files. Use when: transfer LoRA weights, extract lora_weights.pt from model rank shards, or fast-download LoRA A/B tensors."
argument-hint: "<remote_checkpoint_dir> <publish_blob_root> <local_dest_dir> [remote_node]"
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

Example inputs:

```text
SOURCE=az://orngwus2cresco/data/boren/outputs/.../global_step_100/actor
ROOT=az://orngwus2cresco/data/boren/outputs/.../global_step_100
DST=~/data/ckp/run/global_step_100
NODE=verl-n1-i10
```

Both the remote and local nodes must have an authenticated `bbb` command.
The remote Python environment must contain `torch`; set `PYTHON=/path/to/python`
when `python` is not the correct interpreter.

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

## Safeguards

- Accept only keys matching `*.lora_A[.<adapter>].weight` and
  `*.lora_B[.<adapter>].weight`.
- Require every adapter prefix to have both A and B tensors.
- Do not include base-layer, embedding, optimizer, or scheduler tensors.
- Use `bbb` for every blob upload and download.
- Do not use `ckpt_delta.py` or infer A/B matrices from merged model weights.

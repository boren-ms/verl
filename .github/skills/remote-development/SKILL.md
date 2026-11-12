---
name: remote-development
description: Create, resume, connect to, or develop on a remote Brix GPU node. Use for Brix devbox lifecycle, remote setup, dependency installation, and remote development or testing; use run-asr-exp for a managed ASR Ray job.
---

# Remote Development

Manage a Brix development node and run scoped development work on it. When
asked only to edit this skill, do not create, resume, connect to, or modify a
remote node.

## Context and boundaries

The usual context is cluster `prod-westus2-cw-6`, namespace `boren`, local repo
`/home/boren/code/verl`, and remote repo `/root/code/verl`. Verify current state
instead of relying on an old report.

- Use `brix ssh`; never raw `ssh`, `scp`, or `brix tmux`.
- Edit code locally and send it with `bpush`; avoid remote source edits.
- Use `bbb` through Azure Blob for non-code files.
- Do not pause, delete, restart, or otherwise alter unrelated pools.
- Preserve the requested cluster, topology, priority, team, image, packages, and test
  scope unless the user authorizes a change.

## Resolve or provision the node

Use the supplied node/pool name. For existing-node work without one, list
matching pools and ask only if the target remains ambiguous. For a requested
new node, these are defaults unless the user specifies otherwise:

| Parameter | Default |
| --- | --- |
| name | `verl-n1-i<I>`; smallest unused nonnegative `I` |
| cluster / namespace | `prod-westus2-cw-6` / `boren` |
| pods / GPUs per pod | `1` / `8` |
| priority / team | `team-critical` / `team-moonfire-speech` |
| image / launch config | Published verl/ReMax image from `docker/remax/README.md` / `docker/remax/image.rcall.py` |

For new verl devboxes, read `docker/remax/README.md` and the corresponding
`docker/remax/VERIFICATION.md` entry in the current checkout. Use the published,
verified immutable image reference (`repository@sha256:...`), unless the user
specifies another image. Do not select an arbitrary latest tag or rebuild the
image as part of ordinary devbox creation. The currently documented reference is:

```text
iridiumsdc.azurecr.io/boren/verl-remax@sha256:d3624316721afeb2cf072b803cecac5a01e45f691b84e58e2cb43fc70d565b4e
```

Recheck those files when using this skill; the reference above may be superseded.

Inspect all matching states, including suspended pools:

```bash
brix pools -a -c prod-westus2-cw-6 -n boren 'verl-n*'
```

If the exact authorized pool exists and is Paused/suspended, resume it and poll
until Ready; do not create a duplicate. For Assigning, Pending, or starting
pools, inspect scheduling/events and wait; do not repeatedly resume or recreate
them. Diagnose failed pools before changing them:

```bash
brix resume -c prod-westus2-cw-6 -n boren <POOL>
brix pools -a -c prod-westus2-cw-6 -n boren <POOL>
```

For a missing pool that the user asked to create:

```bash
# Run from the current verl checkout/worktree; use its documented image digest.
VERL_DOCKER_IMAGE=iridiumsdc.azurecr.io/boren/verl-remax@sha256:d3624316721afeb2cf072b803cecac5a01e45f691b84e58e2cb43fc70d565b4e \
twdev create-ray-devbox \
  cluster=prod-westus2-cw-6 num_pods=1 num_gpu=8 \
  job_name=<POOL> priority_class=team-critical team=team-moonfire-speech \
  skip_rcall_user_config=true \
  extra_rcall_configs="$(pwd)/docker/remax/image.rcall.py"
```

Keep both launch options: `image.rcall.py` loads the normal user network,
storage, and infrastructure settings first, then selects `VERL_DOCKER_IMAGE`.
Without `skip_rcall_user_config=true`, the later user config can overwrite the
custom image. The hook also recreates TorchFlow's setup marker and registers
its mounted source with `--no-deps`, preserving the baked dependencies and Ray
plugin. Do not change the shared user config or infrastructure image. This
launch config accepts the dedicated `iridiumsdc.azurecr.io/boren/verl-remax`
repository; do not pass an unrelated user-specified image through it.

Poll at a reasonable interval until Ready or a scheduling/error condition
requires user action. Report the exact pool, cluster, status, and GPU topology.
For a new pool, inspect the pod with `kubectl --context prod-westus2-cw-6 -n
boren get pod <POOL>-0 -o json`; find the `main` container by name and confirm
its configured image and actual `imageID` match the selected digest. Resuming
an existing pool retains its image; inspect and report it rather than silently
replacing or recreating the pool to adopt the new default.

## Sync and execute

Push the current repo or worktree, wait for completion, then run all remote
commands in a login shell:

```bash
BRIX_AVAILABLE_CLUSTERS=prod-westus2-cw-6 bpush <POOL>
brix ssh -c prod-westus2-cw-6 -n boren <POOL> -- \
  'bash -l -c "cd /root/code/verl && <COMMAND>"'
```

`bpush` wraps `brix git push`, preserves the canonical repo name for worktrees,
and targets `/root/code/<repo>`. It may be a shell function or an executable
(`~/.local/bin/bpush` here); check `command -v bpush` before assuming an
interactive shell is required. Scope `BRIX_AVAILABLE_CLUSTERS` to the requested
cluster so a matching pool elsewhere is not synced. It includes non-ignored
untracked files and dirty tracked changes; do not stage or commit unrelated
local changes just to sync. If a command is not on `PATH`, check
`~/.virtualenvs/openai/bin/{brix,bbb}`.

Wait for the pool to have a Ready pod before treating a successful `bpush` as
proof of synchronization: it can return success with `currentReplicas: 0`
while a new pool is still starting. Once connected, verify the remote
`git rev-parse HEAD` matches the snapshot commit reported by `bpush` (dirty
local changes produce a snapshot SHA distinct from local HEAD).

Use the repository's required installer and install only requested or necessary
dependencies. For the custom verl image, run `bash quick_install.sh` from
`/root/code/verl` after syncing; it checks and reuses the baked environment,
binds editable packages to the pushed checkout, and runs GPU kernel checks.
Inspect the current installer's marker/version requirements against the image
before invoking it: a mismatch triggers automatic reinstallation without a
confirmation prompt. Diagnose mismatches first rather than treating a full
reinstall as normal image reuse. Do not replace Ray in a running cluster.
Verify the observable result—for example, an import, version,
targeted test, or generated artifact—and surface errors immediately.

## Validate the image when requested

For a fresh-image validation, save pod UID, node, `main` imageID, readiness,
and restartCount before and after the checks. After syncing code, run the
verifier against the baked source **before** `quick_install.sh`, so setup does
not mask a broken image:

```bash
brix ssh -c prod-westus2-cw-6 -n boren <POOL> -- \
  'bash -l -c "cd /opt/verl-source && VERL_VERIFY_REPORT=/tmp/verl-image-verification.json timeout 600 python /root/code/verl/docker/remax/verify.py"'
brix ssh -c prod-westus2-cw-6 -n boren <POOL> -- \
  'bash -l -c "cd /root/code/verl && timeout 300 python -m torch.distributed.run --standalone --nproc-per-node=8 docker/remax/verify_nccl.py"'
```

Use the actual allocated GPU count for `--nproc-per-node`. Then run normal
`bash quick_install.sh` from the synced checkout and rerun `verify.py` there
with a separate report path. Inspect the installer log to confirm that main
dependencies were reused, and preserve verifier reports and relevant logs
locally through `bbb` before cleanup. GPU kernels, managed Ray worker, and
NCCL checks establish environment usability; they do not establish training
convergence or model quality. If a check fails, diagnose its first concrete
error and record any temporary package changes separately from image results.

## Network and file transfer

Brix nodes may lack public internet access: do not assume PyPI or external Git
is reachable. If needed, obtain a wheel or artifact locally, stage it through
Blob, then install or consume it on the node.

Use `bpush` for code. Use `bbb cp` or incremental `bbb sync` for data, wheels,
configs, checkpoints, logs, and results via:

```text
az://orngwus2cresco/data/boren/data/verl/
```

Examples:

```bash
# Local -> Blob -> remote
bbb sync /local/files/ az://orngwus2cresco/data/boren/data/verl/files/
brix ssh -c prod-westus2-cw-6 -n boren <POOL> -- \
  'bash -l -c "bbb sync az://orngwus2cresco/data/boren/data/verl/files/ /root/data/files/"'

# Remote -> Blob -> local
brix ssh -c prod-westus2-cw-6 -n boren <POOL> -- \
  'bash -l -c "bbb sync /root/data/results/ az://orngwus2cresco/data/boren/data/verl/results/"'
bbb sync az://orngwus2cresco/data/boren/data/verl/results/ /local/results/
```

Before a large transfer, verify source, destination, expected contents, and
available space. Do not use deletion/mirroring options unless explicitly
requested.

## Cleanup when requested

For a temporary node the user asked to delete, save evidence first, recheck
the exact pool's UID against the one recorded at creation, then delete only
that pool (no wildcard):

```bash
brix delete -c prod-westus2-cw-6 -n boren -y <POOL>
```

Confirm the Pool and its owned pods are gone; a successful delete request is
not proof that GPU resources were released. Pausing retains the Pool and is
not equivalent to deleting it. Do not delete the underlying shared physical
GPU host or any unrelated pool. If validation fails, still clean up this
temporary pool when deletion was requested, after saving failure evidence.

## Report

State the node/pool, cluster, readiness/topology, selected and actual image,
code revision or sync result,
commands run, dependency outcome, and verification result. Keep the report
concise and distinguish remote changes from local source changes.

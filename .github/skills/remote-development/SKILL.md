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
| image / launch config | Immutable digest and launch configuration from the selected pool or its verified build receipt |

The former `docker/remax` build helpers and root quick installer were removed.
For new devboxes, locate the current image build and launch configuration in
its owning checkout (for Orange builds, the dedicated bootstrapping branch).
Verify its source revision, immutable image digest and setup hooks before use.
Existing pools retain their configured image. Do not select an arbitrary latest
tag or rebuild an image as part of ordinary development.

Inspect all matching states, including suspended pools:

```bash
brix pools -a -c prod-westus2-cw-6 -n boren 'verl-n*'
```

Create or resume an authorized case-owned pool through normal scheduler
admission at the requested priority. Scheduler-managed preemption under existing
cluster policy is allowed; do not require a separate no-preemption audit,
cluster-wide Pod inventory, or physically idle GPU proof before the request.
Honor an explicit user no-preemption restriction when present. Keep the request
pending when capacity is unavailable, and inspect allocated pods before running
the workload. This does not authorize manually changing unrelated pools or
shared scheduling policies.

If the exact authorized pool exists and is Paused/suspended, resume it and poll
until Ready; do not create a duplicate. For Assigning, Pending, or starting
pools, inspect scheduling/events and wait; do not repeatedly resume or recreate
them. Diagnose failed pools before changing them:

```bash
brix resume -c prod-westus2-cw-6 -n boren <POOL>
brix pools -a -c prod-westus2-cw-6 -n boren <POOL>
```

For a missing pool that the user asked to create, inspect the installed
`twdev create-ray-devbox` interface and the selected image's launch configuration.
Use its supported image/launch options; do not invoke the deleted repository
hook or copy a historical digest. Preserve normal network, storage and
infrastructure settings. Confirm setup hooks reuse baked dependencies and
register any required TorchFlow Ray plugin without replacing live Ray.

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

Use an existing compatible GPU environment. On every participating GPU node,
register the synced source without replacing runtime dependencies:

```bash
uv pip install --system --no-deps --no-build-isolation -e . -e plugins/phimm -e plugins/qwen35_audio
```

Run from `/root/code/verl` in a login shell. Check the actual environment's
versions/imports first and diagnose missing dependencies separately. The ASR
submitter sets source paths for its driver and workers, but editable registration
is still needed for Hydra plugin discovery. Do not replace Ray in a running
cluster. Verify an import, targeted test or generated artifact as appropriate.

## Validate the image when requested

Save pod UID, node, main-container imageID, readiness and restart count before
and after validation. Run the current build owner's GPU verifier against baked
source before installing editable code, then repeat against the synced checkout.
Use the verifier and NCCL probe from that verified build checkout; the former
repository probes were removed. Test all allocated GPUs and cross-node
communication for a multi-node request. Save version, import, GPU kernel,
Ray-worker and NCCL results through `bbb` before cleanup. Source setup must not
mask a broken baked image, and environment checks do not prove model quality.

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

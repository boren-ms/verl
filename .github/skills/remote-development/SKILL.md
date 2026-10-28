---
name: remote-development
description: Create, resume, connect to, or develop on a remote Brix GPU node. Use for Brix devbox lifecycle, remote setup, dependency installation, and remote development or testing; use verl-asr-run for a managed ASR Ray job.
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
- Preserve the requested cluster, topology, priority, team, packages, and test
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

Inspect all matching states, including suspended pools:

```bash
brix pools -a -c prod-westus2-cw-6 -n boren 'verl-n*'
```

If the exact authorized pool exists but is not Ready, resume it and poll until
Ready; do not create a duplicate:

```bash
brix resume -c prod-westus2-cw-6 -n boren <POOL>
brix pools -a -c prod-westus2-cw-6 -n boren <POOL>
```

For a missing pool that the user asked to create:

```bash
twdev create-ray-devbox \
  cluster=prod-westus2-cw-6 num_pods=1 num_gpu=8 \
  job_name=<POOL> priority_class=team-critical team=team-moonfire-speech
```

Poll at a reasonable interval until Ready or a scheduling/error condition
requires user action. Report the exact pool, cluster, status, and GPU topology.

## Sync and execute

Push the current repo or worktree, wait for completion, then run all remote
commands in a login shell:

```bash
bpush <POOL>
brix ssh -c prod-westus2-cw-6 -n boren <POOL> -- \
  'bash -l -c "cd /root/code/verl && <COMMAND>"'
```

`bpush` is a zsh function that wraps `brix git push`, preserves the canonical
repo name for worktrees, and targets `/root/code/<repo>`. If a command is not on
`PATH`, check `~/.virtualenvs/openai/bin/{brix,bbb}`.

Use the repository's required installer and install only requested or necessary
dependencies. Verify the observable result—for example, an import, version,
targeted test, or generated artifact—and surface errors immediately.

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

## Report

State the node/pool, cluster, readiness/topology, code revision or sync result,
commands run, dependency outcome, and verification result. Keep the report
concise and distinguish remote changes from local source changes.

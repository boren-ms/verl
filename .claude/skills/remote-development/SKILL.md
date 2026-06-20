---
name: remote-development
description: 'Create, resume, or connect to a remote Brix node using brix CLI, switch workspace to ~/code/verl, install dependencies, and run development tasks. Use when: creating a new GPU node, resuming a paused node, setting up a remote dev environment, running code on a GPU node, installing required packages, or developing/testing on a remote machine. Triggers: "create node", "spin up a node", "new devbox", "resume node", "connect to remote", "set up remote", "install deps on node".'
argument-hint: 'Target node name, e.g. verl-n1-i0'
---

# Remote Development

Use this skill when the user needs to create a new Brix GPU node, resume a paused node, connect to a remote node, navigate to the workspace, install packages, or do development work on a remote node.

## When to Use
- User wants to **create** a new remote GPU node / devbox — "create node", "spin up a node", "new devbox"
- User wants to **resume** a paused or suspended node — "resume node", "start a node"
- User wants to set up or refresh a remote dev environment
- User needs to install required packages on a remote node
- User says "connect to remote", "set up the remote node", or "install deps on <node>"
- User wants to run, test, or develop code on a remote GPU node

## Procedure

### Step 0 — Create or resume node (if needed)

Skip this step if the user already has a running node.

#### Node creation defaults

| Parameter | Default |
|-----------|---------|
| prefix | `verl` |
| N (num_pods) | `1` |
| i (index) | auto-pick unused |
| cluster | `prod-westus2-cw-6` |
| num_gpu | `8` |
| priority_class | `team-critical` |
| team | `team-moonfire-speech` |

The full job name is `{PREFIX}-n{N}-i{I}`.

#### 0a. Check for existing pools

```bash
brix ls '{PREFIX}-n*' 2>&1
```

- If user provided a specific `i`, check if `{PREFIX}-n{N}-i{I}` exists.
- If user did NOT provide `i`, parse existing `{PREFIX}-n{N}-i*` names, extract used `i` values, and pick the smallest unused `i` starting from 0.

#### 0b. Resume if paused/suspended

If the pool exists but status is `Paused`, `Suspended`, or any non-Ready state:
```bash
brix resume {PREFIX}-n{N}-i{I}
```
Poll until Ready:
```bash
brix ls '{PREFIX}-n{N}-i{I}' 2>&1
```
Wait 15 seconds between polls. Report each status so user sees progress.

If already `Ready`, report and skip to Step 1.

#### 0c. Create new node

If the pool does NOT exist:
```bash
twdev create-ray-devbox \
  cluster={CLUSTER} \
  num_pods={N} \
  num_gpu={NUM_GPU} \
  job_name={PREFIX}-n{N}-i{I} \
  priority_class={PRIORITY} \
  team={TEAM}
```

#### 0d. Verify

Confirm the node is up:
```bash
brix ls '{PREFIX}-n{N}-i{I}' 2>&1
```

Report: job name, cluster, status, size (`{N} x {NUM_GPU} GPU`).

### Step 1 — Resolve the target node
If the user provides a node name (e.g. `cw-n1-i0`), use it directly.
If not, ask for the node name before proceeding.

### Step 2 — Push local code to the remote node
```bash
bpush <NODE>
```
- Pushes the current git repo (or worktree) to the remote node via `brix git push`.
- Works from both regular repos and git worktrees — automatically maps the worktree directory to the canonical repo name so the remote path stays `/root/code/<repo>`.
- Run this from anywhere inside the repo or worktree.
- Wait for the push to complete before proceeding.

### Step 3 — Run commands via brix ssh
Use `brix ssh` for all remote command execution — both one-off commands and multi-step workflows:
```bash
brix ssh <NODE> -- 'bash -l -c "<COMMAND>"'
```
- Use `brix ssh` for every remote interaction. Do **not** use `brix tmux`.
- For multi-step workflows, chain commands with `&&` or run each as a separate `brix ssh` invocation.
- The remote workspace is at `/root/code/verl`.

### Step 4 — Install dependencies
Install dependencies on the remote node:
```bash
brix ssh <NODE> -- 'bash -l -c "cd ~/code/verl && <install command for the target project>"'
```
- Use the install command specified by the repo or environment you are working in.
- Wait for the install to complete (watch for success/error output).
- If the user specifies packages, substitute them in the install command.

### Step 5 — Verify
After install, optionally verify the required packages or environment:
```bash
brix ssh <NODE> -- 'bash -l -c "python3 -c \"import <required_package>; print(\\\"OK\\\")\""'
```

## Network Constraints

Remote Brix nodes **may not have full internet access**. Keep this in mind:
- **Do not assume `pip install` from PyPI will work.** Packages may need to be pre-built locally or synced via blob storage.
- **Do not assume `git clone` from external repos will work.**
- Use `bbb` (blob storage) as the intermediary for transferring files, wheels, datasets, and other non-code artifacts.
- Code is pushed via `bpush` (which uses `brix git push` over the internal network, not the public internet).

## File Transfer with `bbb`

Use `bbb` to sync **non-code files** (data, wheels, configs, model checkpoints, etc.) between local machines, remote nodes, and Azure blob storage. Do **not** use `bbb` for code — use `bpush` for that.

### Transfer port (blob staging area)
```
az://orngwus2cresco/data/boren/data/verl/
```
Use this path as the intermediate staging area on Azure blob for files that need to move between local and remote nodes.

### Common patterns

**Local → Blob → Remote** (for files the remote node needs):
```bash
# 1. Upload from local to blob
bbb sync /local/path/to/files az://orngwus2cresco/data/boren/data/verl/files/

# 2. On remote node, download from blob
brix ssh <NODE> -- 'bbb sync az://orngwus2cresco/data/boren/data/verl/files/ /root/data/files/'
```

**Remote → Blob → Local** (for results, logs, checkpoints):
```bash
# 1. On remote node, upload to blob
brix ssh <NODE> -- 'bbb sync /root/data/results/ az://orngwus2cresco/data/boren/data/verl/results/'

# 2. Download from blob to local
bbb sync az://orngwus2cresco/data/boren/data/verl/results/ /local/path/to/results/
```

**Single file copy:**
```bash
bbb cp /local/file.whl az://orngwus2cresco/data/boren/data/verl/wheels/file.whl
bbb cp az://orngwus2cresco/data/boren/data/verl/wheels/file.whl /root/file.whl  # on remote
```

**List blob contents:**
```bash
bbb ls az://orngwus2cresco/data/boren/data/verl/
```

### Tips
- `bbb sync` is incremental — only changed/missing files are transferred.
- Use `bbb sync --delete` to mirror exactly (removes extra files at destination).
- Use `-x <regex>` to exclude files matching a pattern.
- Both local and remote nodes have `bbb` available.

## Development Workflow

When developing or testing on the remote node, **minimize the changes introduced**:
- **Edit locally, push remotely.** Make code changes on the local machine, then run `bpush <NODE>` to push them. This works from both regular repos and git worktrees. Avoid editing files directly on the remote node.
- **Always use `brix`** to access the remote node — use `brix ssh`. Do **not** use `brix tmux`. Use `bpush` for pushing code. Do not use raw `ssh` or `scp`.
- **Use `bbb` for non-code files.** Data, wheels, configs, checkpoints — sync via blob storage using the transfer port `az://orngwus2cresco/data/boren/data/verl/`.
- **Use the project's required installer** when installing or updating dependencies. If PyPI is unreachable, pre-download wheels locally, upload to blob with `bbb`, then `pip install` from the local path on the remote.
- **Run tests on remote** via `brix ssh <NODE> -- 'bash -l -c "cd ~/code/verl && <test command>"'`.
- **Keep changes small.** When iterating, push only what changed (`bpush` is incremental via git). Avoid reinstalling packages unless dependencies actually changed.

## Command Resolution
- `bpush` is a zsh function (defined in `~/.zshrc`) that wraps `brix git push`. It detects the canonical repo name from the main worktree and handles the symlink mapping automatically.
- `brix` may be at `~/.virtualenvs/openai/bin/brix` if not on PATH.
- `bbb` may be at `~/.virtualenvs/openai/bin/bbb` if not on PATH. Used for blob file operations (sync, cp, ls, rm).
- Always use `brix` for remote access — use `brix ssh`. Do **not** use `brix tmux`. Use `bpush` for pushing code. Use `bbb` for syncing non-code files. Do not use raw `ssh` or `scp`.
- Use the dependency installer required by the target project or environment.
- The remote workspace is typically at `/root/code/verl` (not `/home/boren/...`).
- The blob transfer port for non-code files is `az://orngwus2cresco/data/boren/data/verl/`.

## Response Style
- Confirm which node you connected to.
- Report install output (success or errors).
- Keep responses concise; surface errors immediately.

---
name: remote-development
description: 'Connect to a remote Brix node using rcall-brix, switch workspace to ~/code/verl, install dependencies, and run development tasks. Use when setting up a remote dev environment, running code on a GPU node, installing required packages, or developing/testing on a remote machine.'
argument-hint: 'Target node name, e.g. cw-n1-i0'
---

# Remote Development

Use this skill when the user needs to connect to a remote Brix node, navigate to the workspace, install packages, or do development work on a remote node.

## When to Use
- User wants to set up or refresh a remote dev environment
- User needs to install required packages on a remote node
- User says "connect to remote", "set up the remote node", or "install deps on <node>"
- User wants to run, test, or develop code on a remote GPU node

## Procedure

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

### Step 3 — Connect via rcall-brix tmux (dedicated session)
Use a **fixed session name per conversation thread** so subsequent invocations reuse the same tmux session:
```bash
rcall-brix tmux -s codex <NODE>
```
- Always use session name `codex` (or another fixed name) so the same tmux session is reattached within the same thread.
- This avoids colliding with the user's other tmux sessions (e.g. session `0`) while keeping state across multiple skill invocations in the same conversation.
- This is an interactive session — use `bash` with `mode="async"` to start it, then `write_bash` to send commands.
- Wait for the shell prompt to appear before sending further input.
- If the `codex` session already exists on the remote, `tmux -s codex` will reattach to it — this is the desired behavior.

### Step 4 — Switch to the workspace
Once connected, send:
```
cd ~/code/verl
```
Confirm the directory changed (prompt shows the path, or run `pwd`).

### Step 5 — Install dependencies
Install dependencies using the package manager required by the target project or environment.
```
<install command for the target project>
```
- Use the install command specified by the repo or environment you are working in.
- Wait for the install to complete (watch for success/error output).
- If the user specifies packages, substitute them in the install command.
- **Fallback:** If tmux output is hard to read, use `rcall-brix ssh <NODE> -- 'bash -l -c "cd ~/code/verl && <install command for the target project>"'` instead.

### Step 6 — Verify
After install, optionally verify the required packages or environment:
```
python3 -c "import <required_package>; print('OK')"
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
rcall-brix ssh <NODE> -- 'bbb sync az://orngwus2cresco/data/boren/data/verl/files/ /root/data/files/'
```

**Remote → Blob → Local** (for results, logs, checkpoints):
```bash
# 1. On remote node, upload to blob
rcall-brix ssh <NODE> -- 'bbb sync /root/data/results/ az://orngwus2cresco/data/boren/data/verl/results/'

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
- **Always use `rcall-brix`** to access the remote node — use `rcall-brix ssh` or `rcall-brix tmux`. Use `bpush` for pushing code. Do not use raw `ssh` or `scp`.
- **Use `bbb` for non-code files.** Data, wheels, configs, checkpoints — sync via blob storage using the transfer port `az://orngwus2cresco/data/boren/data/verl/`.
- **Use the project's required installer** when installing or updating dependencies. If PyPI is unreachable, pre-download wheels locally, upload to blob with `bbb`, then `pip install` from the local path on the remote.
- **Run tests on remote** via `rcall-brix ssh <NODE> -- 'bash -l -c "cd ~/code/verl && <test command>"'` for one-off commands, or inside the tmux session for interactive work.
- **Keep changes small.** When iterating, push only what changed (`bpush` is incremental via git). Avoid reinstalling packages unless dependencies actually changed.

## Command Resolution
- `bpush` is a zsh function (defined in `~/.zshrc`) that wraps `brix git push`. It detects the canonical repo name from the main worktree and handles the symlink mapping automatically.
- `rcall-brix` may be at `~/.virtualenvs/openai/bin/rcall-brix` if not on PATH.
- `bbb` may be at `~/.virtualenvs/openai/bin/bbb` if not on PATH. Used for blob file operations (sync, cp, ls, rm).
- Always use `rcall-brix` for remote access — `rcall-brix ssh`, `rcall-brix tmux`. Use `bpush` for pushing code. Use `bbb` for syncing non-code files. Do not use raw `ssh` or `scp`.
- Use the dependency installer required by the target project or environment.
- The remote workspace is typically at `/root/code/verl` (not `/home/boren/...`).
- The blob transfer port for non-code files is `az://orngwus2cresco/data/boren/data/verl/`.

## Response Style
- Confirm which node you connected to.
- Report install output (success or errors).
- Keep responses concise; surface errors immediately.

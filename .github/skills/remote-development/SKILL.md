---
name: remote-development
description: 'Connect to a remote Brix node using rcall-brix, switch workspace to ~/code/openai, install Python dependencies with oaipkg install, and run development tasks. Use when setting up a remote dev environment, running code on a GPU node, installing packages like speech and knight, or developing/testing on a remote machine.'
argument-hint: 'Target node name, e.g. cw-n1-i0'
---

# Remote Development

Use this skill when the user needs to connect to a remote Brix node, navigate to the workspace, install packages, or do development work on a remote node.

## When to Use
- User wants to set up or refresh a remote dev environment
- User needs to install `speech`, `knight`, or other packages on a remote node
- User says "connect to remote", "set up the remote node", or "install deps on <node>"
- User wants to run, test, or develop code on a remote GPU node

## Procedure

### Step 1 — Resolve the target node
If the user provides a node name (e.g. `cw-n1-i0`), use it directly.
If not, ask for the node name before proceeding.

### Step 2 — Sync local code to the remote node
```bash
rcall-brix sync <NODE>
```
- Syncs the current local workspace to the remote node before connecting.
- Run this from the local repo root (e.g. `~/code/openai`).
- Wait for sync to complete before proceeding.

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
cd ~/code/openai
```
Confirm the directory changed (prompt shows the path, or run `pwd`).

### Step 5 — Install dependencies
**Always use `oaipkg install`** to install packages. Never use `pip install` directly.
```
oaipkg install speech knight
```
- This installs the `speech` and `knight` packages using the internal oaipkg tool.
- Wait for the install to complete (watch for success/error output).
- If the user specifies different packages, substitute them here.
- **Fallback:** If tmux output is hard to read, use `rcall-brix ssh <NODE> -- 'bash -l -c "cd ~/code/openai && oaipkg install speech knight"'` instead. The `bash -l` login shell picks up Azure credentials needed by oaipkg.

### Step 6 — Verify
After install, optionally verify:
```
python3 -c "import speech; import knight; print('OK')"
```

## Development Workflow

When developing or testing on the remote node, **minimize the changes introduced**:
- **Edit locally, sync remotely.** Make code changes on the local machine, then run `rcall-brix sync <NODE>` to push them. Avoid editing files directly on the remote node.
- **Always use `rcall-brix`** to access the remote node — use `rcall-brix ssh`, `rcall-brix tmux`, or `rcall-brix sync`. Do not use raw `ssh` or `scp`.
- **Always use `oaipkg install`** when installing or updating dependent packages. Do not use `pip install` — oaipkg handles internal package resolution and Azure credentials.
- **Run tests on remote** via `rcall-brix ssh <NODE> -- 'bash -l -c "cd ~/code/openai && <test command>"'` for one-off commands, or inside the tmux session for interactive work.
- **Keep changes small.** When iterating, sync only what changed (rcall-brix sync is incremental). Avoid reinstalling packages unless dependencies actually changed.

## Command Resolution
- `rcall-brix` may be at `~/.virtualenvs/openai/bin/rcall-brix` if not on PATH.
- Always use `rcall-brix` for all remote access — `rcall-brix ssh`, `rcall-brix tmux`, `rcall-brix sync`. Do not use raw `ssh` or `scp`.
- `oaipkg` is the internal OpenAI package manager; always use it for installing packages. Do not fall back to `pip install`.
- The remote workspace is typically at `/root/code/openai` (not `/home/boren/...`).

## Response Style
- Confirm which node you connected to.
- Report install output (success or errors).
- Keep responses concise; surface errors immediately.

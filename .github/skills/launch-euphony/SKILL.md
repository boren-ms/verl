---
name: launch-euphony
description: 'Launch Euphony frontend+backend dev servers on an available remote Brix node. Use when: starting Euphony, launching Euphony remotely, running the Euphony dev server, deploying Euphony to a Brix node, or opening the Euphony UI. Finds an available Ready node (prefers team-critical queue), syncs code, pulls cached node_modules from Azure blob, starts both Vite and FastAPI servers, and reports the network URL.'
argument-hint: 'Optional node name, e.g. bus-ats-prod-westus2-19-i0'
---

# Launch Euphony

End-to-end skill that launches the Euphony frontend (Vite) and backend (FastAPI/uvicorn) dev servers on a remote Brix node, then reports the access URLs.

## When to Use
- User says "launch Euphony", "start Euphony", "run Euphony server"
- User wants to open the Euphony UI on a remote node
- User wants a one-command Euphony deployment to any available Brix node

## Prerequisites
- `rcall-brix` must be available (check `~/.virtualenvs/openai/bin/rcall-brix` if not on PATH)
- The node_modules cache must exist at `az://orngwus2cresco/data/boren/data/tools/euphony/node_modules.tar.gz` (push via `./setup_euphony_dev.sh --push` from a machine with internet)
- The setup script lives at `project/euphony/setup_euphony_dev.sh` in the repo

## Procedure

### Step 1 — Resolve the target node

**If the user provides a node name**, use it directly.

**If not**, find an available Ready node:

```bash
rcall-brix ls 2>&1 | sed 's/\x1b\[[0-9;]*m//g'
```

Parse the output and select a node using this priority order:
1. **Ready** nodes in the `team-critical` QUEUE — pick the first one
2. **Ready** nodes in any other queue — pick the first one
3. If no Ready nodes exist, stop and tell the user

Confirm the chosen node name with the user before proceeding.

### Step 2 — Sync code to the remote node

Use the `remote-development` skill's Step 2: sync the local workspace so the remote has the latest `setup_euphony_dev.sh`.

```bash
rcall-brix sync <NODE>
```

Wait for sync to complete. If the script doesn't appear on the remote (the sync only pushes git-tracked files), transfer it directly:

```bash
rcall-brix ssh <NODE> -- 'cat > ~/code/openai/project/euphony/setup_euphony_dev.sh' \
  < project/euphony/setup_euphony_dev.sh
rcall-brix ssh <NODE> -- 'chmod +x ~/code/openai/project/euphony/setup_euphony_dev.sh'
```

### Step 3 — Pull cached node_modules (if needed)

Check whether `node_modules` already exists on the remote:

```bash
rcall-brix ssh <NODE> -- 'test -d ~/code/openai/project/euphony/node_modules/.pnpm && echo EXISTS || echo MISSING'
```

If MISSING, pull the cache from Azure blob:

```bash
rcall-brix ssh <NODE> -- 'bash -l -c "cd ~/code/openai/project/euphony && bash setup_euphony_dev.sh --pull"'
```

This downloads a single ~34 MB tarball from `az://orngwus2cresco/data/boren/data/tools/euphony/node_modules.tar.gz` and extracts it. Takes only a few seconds.

### Step 4 — Start Euphony servers

Connect via tmux and launch both servers:

```bash
rcall-brix tmux -s euphony <NODE>
```

This is an interactive session — use `bash` with `mode="async"`, then `write_bash` to send commands:

```
cd ~/code/openai/project/euphony
bash setup_euphony_dev.sh --dev
```

Wait for the startup banner. The script:
1. Starts **backend** (uvicorn) on port 8020, waits for it to respond
2. Starts **frontend** (vite with `--host 0.0.0.0`) on port 3000
3. Extracts the Network IP from the vite log
4. Prints a banner like:
```
[euphony-setup] ════════════════════════════════════════════════════
[euphony-setup]   Euphony dev servers are running!
[euphony-setup]
[euphony-setup]   Frontend:  http://10.142.76.108:3000/
[euphony-setup]   Backend:   http://10.142.76.108:8020/
[euphony-setup]
[euphony-setup]   Press Ctrl+C to stop both servers.
[euphony-setup] ════════════════════════════════════════════════════
```

### Step 5 — Verify and report

After the banner appears, verify both services from the local devbox:

```bash
curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 http://<IP>:3000/
curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 http://<IP>:8020/docs
```

Report to the user:
- **Node**: which node the servers are running on
- **Frontend URL**: `http://<IP>:3000/`
- **Backend URL**: `http://<IP>:8020/`
- **tmux session**: `euphony` (reattach with `rcall-brix tmux -s euphony <NODE>`)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `node_modules` missing, `--pull` fails | Ensure cache was pushed: run `./setup_euphony_dev.sh --push` on a machine with internet |
| Backend deps missing (`import fastapi` fails) | The script auto-installs via `pip install -e project/euphony`. If oaipkg is required, run `oaipkg install` first per remote-development skill |
| Port already in use | The script auto-kills existing processes on ports 3000 and 8020 |
| `pnpm` not available on Brix node | Expected — the script uses vite directly from `node_modules/.bin/` |
| Vite shows "use --host to expose" instead of IP | The script invokes vite with `--host 0.0.0.0`; if this appears, the wrong script version is on the remote — re-sync |

## Important Details
- **Brix nodes have no internet** — all npm packages come from the cached tarball
- **SSH port forwarding doesn't work** through the Brix relay — use the pod's network IP directly
- **The tmux session persists** — Ctrl+C in the tmux session stops both servers; detaching (`Ctrl+B, D`) keeps them running
- **Cache blob**: `az://orngwus2cresco/data/boren/data/tools/euphony/node_modules.tar.gz`

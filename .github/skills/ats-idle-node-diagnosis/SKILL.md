---
name: ats-idle-node-diagnosis
description: 'Diagnose idle ATS nodes (bus-ats-prod) where GPUs show 0% utilization but nodes are Ready. Use when: ATS nodes are idle, bus-ats-prod GPUs unused, diagnosing why jobs died or are not running on ATS nodes, root-cause analysis of idle bus-ats-prod pools, investigating zombie jobs or crashed training on ATS.'
argument-hint: 'Optional pool pattern, e.g. bus-ats-prod-westus2-cw-6-*'
---

# ATS Idle Node Diagnosis

Diagnose why `bus-ats-prod` nodes show **Ready** but have **idle GPUs** (0% utilization). The skill SSHes into each idle node, collects multi-layered evidence (GPU state, logs, processes, disk, memory, dmesg), and classifies the root cause.

## When to Use
- ATS nodes are Ready but GPUs are at 0% — need to find out why
- User says "why are ATS nodes idle", "bus-ats-prod GPUs unused", "diagnose idle nodes"
- Jobs died or are not running on `bus-ats-prod` pools
- Need root-cause analysis after a training or eval job silently failed
- Zombie jobs suspected (GPU memory held but no compute)

## Scripts

### Full diagnosis — [diagnose_ats.py](./scripts/diagnose_ats.py)

Discovers `bus-ats-prod` Ready pools, SSHes into each, collects evidence from multiple sources, and classifies the root cause of idleness.

```bash
python3 .github/skills/ats-idle-node-diagnosis/scripts/diagnose_ats.py
```

Options:
- `--pattern GLOB` — filter pool names (default: `"bus-ats-prod*"`)
- `--output-dir DIR` — write `diagnosis_report.json` artifact
- `--timeout SECS` — SSH timeout per node (default 30)
- `--workers N` — parallel SSH threads (default 4)
- `--all` — diagnose all Ready nodes, not just idle ones
- `--restart-scheduled` — also restart any Scheduled pools matching the pattern

Example — diagnose all bus-ats-prod pools:
```bash
python3 .github/skills/ats-idle-node-diagnosis/scripts/diagnose_ats.py
```

Example — diagnose specific region and save report:
```bash
python3 .github/skills/ats-idle-node-diagnosis/scripts/diagnose_ats.py \
  --pattern 'bus-ats-prod-westus2-cw-6-*' \
  --output-dir tmp/ats-diagnosis
```

Example — include Scheduled pool restart:
```bash
python3 .github/skills/ats-idle-node-diagnosis/scripts/diagnose_ats.py \
  --restart-scheduled
```

## Procedure

1. **Run the diagnosis script** with `diagnose_ats.py`. By default it targets `bus-ats-prod*` pools. Pass `--pattern` for a narrower subset.
2. **Review the diagnosis output.** The script automatically:
   - Discovers Ready pools (and optionally restarts Scheduled ones)
   - Checks GPU utilization via `nvidia-smi`
   - Identifies idle nodes (0% GPU utilization)
   - For each idle node, collects evidence from 7 sources:
     - **GPU state** — utilization, memory, temperature, running processes (`nvidia-smi pmon`)
     - **bus.log** — last 50 lines for error patterns, job lifecycle events
     - **Running processes** — checks for engine, training, eval, harmony processes
     - **tmux sessions** — lists active tmux sessions for stale job sessions
     - **dmesg** — recent OOM kills, GPU errors, hardware faults
     - **Disk usage** — checks if `/` or `/tmp` are full
     - **systemd journal** — recent container/kubelet errors
   - Classifies the root cause into categories:
     - 🔴 **ZOMBIE** — job died but GPU memory is still held
     - 🔴 **CRASHED** — job process exited with errors in bus.log
     - 🟠 **OOM_KILLED** — process was killed by OOM killer
     - 🟠 **DISK_FULL** — disk usage > 95%, jobs cannot write
     - 🟡 **NO_JOBS** — no job processes found, node is genuinely idle
     - 🟡 **STALE_SESSION** — tmux session exists but job process is gone
     - ⏳ **PROVISIONING** — pod is still starting up
     - ❓ **UNKNOWN** — evidence inconclusive
3. **Act on findings.** Based on the root cause:
   - **ZOMBIE**: Kill the orphan process or restart the pool (`rcall-brix restart <POOL>`)
   - **CRASHED**: Inspect bus.log errors, fix config, resubmit the job
   - **OOM_KILLED**: Reduce batch size or request a larger node
   - **DISK_FULL**: Clean up `/tmp` or stale checkpoints
   - **NO_JOBS**: Submit new jobs or check the scheduler
   - **STALE_SESSION**: Clean up tmux sessions, restart pool if needed
4. **Optional — interactive deep-dive**: If the automated diagnosis is inconclusive, follow the remote-development workflow to SSH into the node:
   ```bash
   rcall-brix tmux -s codex <NODE>
   ```
   Then investigate manually (check logs, processes, GPU state interactively).

## Command Resolution
- If `rcall-brix` is not on PATH, check `~/.virtualenvs/openai/bin/rcall-brix`.
- The scripts call `rcall-brix` directly; aliases (`b`, `rb`) are not used.
- If `rcall-brix` is not available, stop and report the missing command.
- Remote workspace is at `/root/code/openai` (not `/home/boren/...`).

## Response Style
- Lead with a one-line summary: "N of M bus-ats-prod nodes are idle."
- For each idle node, show the root-cause classification with supporting evidence.
- Highlight actionable items — what the user should do to fix each node.
- If all nodes are healthy and busy, say so clearly.
- Keep responses concise; surface errors and recommendations prominently.

---
name: brix-node-gpu-check
description: 'Check Brix node status and GPU utilization on Ready nodes using rcall-brix. Use when: checking GPU usage, monitoring node health over time, inspecting nvidia-smi on remote nodes, listing Ready pools with GPU info, diagnosing idle or underutilized GPUs across Brix nodes.'
argument-hint: 'Optional pool name pattern to filter, e.g. bus-ats-*'
---

# Brix Node GPU Check

Use this skill to list Brix pools that are in `Ready` state, SSH into each to run `nvidia-smi`, and report GPU utilization — highlighting idle and underutilized nodes.

## When to Use
- User asks to check GPU utilization on remote nodes
- User wants to see which nodes are Ready and how busy their GPUs are
- User says "check GPUs", "GPU usage", "nvidia-smi on nodes", or "node status"
- User wants to find idle or underutilized nodes
- User wants to monitor GPU status over time (e.g. "monitor GPUs for 30 seconds")

## Scripts

### Full report — [check_gpu.py](./scripts/check_gpu.py)

Discovers Ready pools, SSHes into each, collects `nvidia-smi` data, and prints a summary table with idle-node highlighting.

```bash
python3 .github/skills/brix-node-gpu-check/scripts/check_gpu.py
```

Options:
- `--pattern GLOB` — filter pool names (e.g. `"bus-ats-*"`, `"cw-n*"`)
- `--output-dir DIR` — write a `gpu_report.json` artifact
- `--timeout SECS` — SSH timeout per node (default 30)
- `--workers N` — parallel SSH threads (default 4)
- `--monitor SECS` — monitor mode: re-check GPU status repeatedly for SECS seconds (default 0 = single check)
- `--interval SECS` — seconds between checks in monitor mode (default 10)

Example — check only westus2-cw-6 pools and save JSON:
```bash
python3 .github/skills/brix-node-gpu-check/scripts/check_gpu.py \
  --pattern 'bus-ats-prod-westus2-cw-6-*' \
  --output-dir tmp/gpu-check
```

Example — monitor all pools for 30 seconds with 10-second intervals:
```bash
python3 .github/skills/brix-node-gpu-check/scripts/check_gpu.py \
  --monitor 30 --interval 10
```

### List Ready pools only — [list_ready_pools.sh](./scripts/list_ready_pools.sh)

Quick helper that runs `rcall-brix ls` and outputs only Ready rows as TSV (`name\tcluster\tsize`).

```bash
bash .github/skills/brix-node-gpu-check/scripts/list_ready_pools.sh
```

With pattern filter:
```bash
bash .github/skills/brix-node-gpu-check/scripts/list_ready_pools.sh 'cw-n*'
```

## Procedure

1. **Run the check script** with `check_gpu.py`. Pass `--pattern` if the user specified a subset. Pass `--monitor SECS` if the user wants to watch GPU status over time (e.g. `--monitor 30` for 30 seconds).
2. **Review the output table**. The script automatically:
   - Filters to only `Ready` pools (skips Assigning, Scheduled, Unhealthy)
   - Marks 🔴 **FULLY IDLE** nodes (0% utilization on all GPUs)
   - Marks 🟡 nodes with half or more GPUs idle
   - Marks 🟢 heavily loaded nodes (≥90% avg)
   - Prints detailed per-GPU breakdown for idle/underutilized nodes
3. **Highlight idle nodes** to the user — these are candidates for new workloads or investigation.
4. **Optional**: If the user asks for cluster-wide capacity beyond their own pools:
   ```bash
   rcall-brix capacity --gpu
   ```

## Command Resolution
- If `rcall-brix` is not on PATH, check `~/.virtualenvs/openai/bin/rcall-brix`.
- The scripts call `rcall-brix` directly; aliases (`b`, `rb`) are not used.
- If `rcall-brix` is not available, stop and report the missing command.

## Response Style
- Lead with the count of Ready pools found and total GPU count.
- Present the summary table from the script output.
- Call out idle nodes explicitly — these are the most actionable finding.
- If no Ready nodes exist, say so clearly.
- Keep responses concise; surface errors immediately.

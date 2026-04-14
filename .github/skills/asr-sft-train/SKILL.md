---
name: asr-sft-train
description: 'Launch ASR SFT training on remote Brix cw-n5-* nodes via tmux. Use when: starting ASR fine-tuning, launching run_asr_sft.py with --selector, retrying failed training jobs, monitoring GPU utilization during training, running speech SFT experiments on remote GPU nodes.'
argument-hint: 'Selector pattern and optional overrides, e.g. egs_complete_mix_hc_pubmed_with_regular_lr1.0_bs256'
---

# ASR SFT Training

Launch `run_asr_sft.py --selector` on a remote Brix `cw-n5-*` node via tmux, with GPU monitoring in a side pane. Automatically checks node readiness and occupancy before launching.

## When to Use
- User wants to launch ASR SFT training on a remote node
- User says "train", "launch training", "run SFT", "start training", or "fine-tune"
- User wants to retry a failed/crashed training job
- User provides a `--selector` pattern for `run_asr_sft.py`

## Inputs

| Input | Required | Default | Example |
|-------|----------|---------|---------|
| Selector pattern | Yes | — | `egs_complete_mix_hc_pubmed_with_regular_lr1.0_bs256` |
| Node pattern | No | `cw-n5-*` | `cw-n5-i0`, `cw-n5-*` |
| Extra args | No | — | `--ts --cluster local` |

## Procedure

### Step 0 — Find Ready `cw-n5-*` nodes and check occupancy

Before launching any training, verify that a target node is Ready and not occupied by another training job. **Never launch training on an occupied node** — training saturates all GPUs.

1. **List Ready nodes** matching the `cw-n5-*` pattern:

   ```bash
   rcall-brix ls 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E '\bReady\b' | awk '{print $1}' | grep 'cw-n5'
   ```

   If the user specifies a particular node (e.g. `cw-n5-i0`), filter to just that node. Otherwise, discover all Ready `cw-n5-*` nodes.

2. **Check node occupancy** on each Ready node. For **each** candidate node, verify it is not running training:

   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "pgrep -fa \"knight\\|sft\\|run_asr_sft\\|train\" 2>/dev/null || echo No training running"'
   ```
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "tmux ls 2>/dev/null || echo No tmux sessions"'
   ```

   Also check GPU utilization to confirm GPUs are idle:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"'
   ```

   Classify each node as:
   - **Unoccupied**: No training processes, no active training tmux sessions, GPU utilization near 0%.
   - **Occupied**: Has running training processes OR active training tmux sessions OR high GPU utilization.

3. **Select the target node**. Pick the first unoccupied Ready `cw-n5-*` node. If the user specified a node, use that one but **warn if occupied**.

   If **no** unoccupied Ready `cw-n5-*` node is found, report all nodes and their status and ask the user how to proceed. Do not launch on an occupied node without user confirmation.

   Report the node selection:
   ```
   Node assignment:
     Training → {NODE} (Ready, unoccupied, GPUs idle)
   Occupied nodes: {NODE_X} (training session active), ...
   ```

### Step 1 — Sync code and verify environment

1. **Sync** the local workspace to the remote node:
   ```bash
   rcall-brix sync {NODE}
   ```

2. **Verify the remote environment**:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "cd ~/code/openai && python3 -c \"import speech; import knight; print(\\\"OK\\\")\""'
   ```

   If this fails, install packages:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "cd ~/code/openai && oaipkg install speech knight"'
   ```

   **Always use `oaipkg install`**, never `pip install`.

### Step 2 — Resolve the training command

Build the full training command from the user's selector and any extra args:

```bash
./run_asr_sft.py --selector "{SELECTOR}" --ts --cluster local
```

- `--selector` is the regex pattern matching experiment names from the sweep config (e.g. `egs_complete_mix_hc_pubmed_with_regular_lr1.0_bs256`).
- `--ts` adds a timestamp suffix to the run name.
- `--cluster local` runs training locally on the node (not on a remote cluster).
- Additional args from the user are appended as-is.

**Always list matching runs** before launching to verify the selector is correct:
```bash
rcall-brix ssh {NODE} -- 'bash -l -c "cd ~/code/openai/data/speech/speech/train && python3 run_asr_sft.py --list --selector \"{SELECTOR}\""'
```

Present the matching run names to the user and **confirm before launching**. If no runs match or unexpected runs appear, adjust the selector before proceeding. Do not skip this step.

### Step 3 — Launch training in tmux with GPU monitor (split panes)

Create a tmux session with a **left/right split**: training on the left pane, GPU monitor on the right pane.

Where `{SELECTOR_SHORT}` is a shortened version of the selector for log file naming.

Use a **multi-step approach** to avoid quoting issues:

1. Create the tmux session:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "tmux new-session -d -s train -n training"'
   ```

2. Send the training command to the left pane:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "tmux send-keys -t train:training \"cd ~/code/openai/data/speech/speech/train && ./run_asr_sft.py --selector \\\"'{SELECTOR}'\\\" --ts --cluster local {EXTRA_ARGS} 2>&1 | tee /tmp/train_{SELECTOR_SHORT}.log\" Enter"'
   ```

3. Split horizontally and start GPU monitor in the right pane:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "tmux split-window -h -t train:training"'
   rcall-brix ssh {NODE} -- 'bash -l -c "tmux send-keys -t train:training.1 \"cd ~/code/openai/data/speech/speech/train && python3 monitor_gpu.py --interval 5 --no-clear 2>&1 | tee /tmp/gpu_monitor.log\" Enter"'
   ```

Report the launched session:
```
Training launched:
  Node: {NODE}
  Tmux session: train
    Left pane  (.0): ./run_asr_sft.py --selector "{SELECTOR}" --ts --cluster local
    Right pane (.1): python3 monitor_gpu.py --interval 5
  Logs: /tmp/train_{SELECTOR_SHORT}.log, /tmp/gpu_monitor.log
```

### Step 4 — Monitor training progress

After launching, **automatically poll** the remote node every **10 minutes** to check training status.

1. **Check if the tmux training window is still active**:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "tmux has-session -t train 2>/dev/null && echo RUNNING || echo FINISHED"'
   ```

2. **If RUNNING**, show a tail of the training log:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "tail -30 /tmp/train_{SELECTOR_SHORT}.log"'
   ```

   Also check GPU utilization:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "tail -20 /tmp/gpu_monitor.log"'
   ```

3. **If FINISHED**, check the exit status by inspecting the log tail for success/error messages:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "tail -50 /tmp/train_{SELECTOR_SHORT}.log | grep -iE \"error|exception|complete|done|saved|checkpoint\""'
   ```

4. **Report status** to the user each cycle:
   ```
   Training status (10m check):
     ⏳ {SELECTOR} on {NODE} — RUNNING (step 1500/5000, loss: 0.23)
   GPU utilization: avg 85% across 8 GPUs
   ```

### Step 5 — Handle crashes and retries

If training fails or the node becomes unreachable:

1. **Detect failure**: SSH unreachable, tmux session lost, or training log shows errors/exceptions.

2. **Inform the user** and start automatic retry (up to **3 attempts**).

3. **Re-discover Ready `cw-n5-*` nodes** (Step 0). Re-check occupancy on all candidates. The crashed node may have recovered or a different node may be available.

4. **Re-sync code** to the new (or recovered) node:
   ```bash
   rcall-brix sync {NODE}
   ```

5. **Re-launch training** (Step 3). Knight/SFT training resumes from the latest checkpoint automatically if the checkpoint directory already has saved steps.

6. **Resume monitoring** (Step 4).

7. If **3 retries are exhausted**, mark as FAILED and report to the user with the list of nodes attempted.

## Important Notes

- **Always use `rcall-brix`** for remote access (ssh, tmux, sync). Never raw ssh.
- **Always use `oaipkg install`** for packages. Never `pip install`.
- **Always check node occupancy** before launching. Never launch on an occupied node without explicit user confirmation.
- The remote workspace is `/root/code/openai` (not `/home/boren/...`). `cd ~/code/openai` resolves correctly on remote.
- The training script is at `data/speech/speech/train/run_asr_sft.py`.
- The GPU monitor script is at `data/speech/speech/train/monitor_gpu.py`.
- Training configs are in `data/speech/speech/train/configs3/`.
- Checkpoints are saved to `az://orngwus2cresco/models/boren/audio_sft/gpt-4o-mini-asr-v1/`.
- `--cluster local` means the training runs directly on the remote node's GPUs, not dispatched to a cluster.
- `--selector` is a regex that filters experiment names from the sweep config in `run_asr_sft.py`.
- Use `--list` to preview which experiments match the selector before launching.

## Ad-hoc Commands

Attach to the training tmux session interactively:
```bash
rcall-brix tmux -s train {NODE}
```

View training log:
```bash
rcall-brix ssh {NODE} -- 'bash -l -c "tail -100 /tmp/train_{SELECTOR_SHORT}.log"'
```

View GPU monitor log:
```bash
rcall-brix ssh {NODE} -- 'bash -l -c "tail -30 /tmp/gpu_monitor.log"'
```

Check all tmux sessions on the node:
```bash
rcall-brix ssh {NODE} -- 'bash -l -c "tmux ls 2>/dev/null || echo No tmux sessions"'
```

Kill a training session (use with caution):
```bash
rcall-brix ssh {NODE} -- 'bash -l -c "tmux kill-session -t train"'
```

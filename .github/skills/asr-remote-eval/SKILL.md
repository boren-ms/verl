---
name: asr-remote-eval
description: 'Evaluate ASR models remotely on a Brix node. Use when running eval_audio.py for a model run, discovering available checkpoint steps from Azure blob storage, generating eval bash scripts, and executing evaluations in tmux. Use for: remote ASR evaluation, checkpoint discovery, eval script generation, model comparison evals.'
argument-hint: 'Run name and optional config, e.g. en_domain_hc_egs_cer05_lr1.0_bs256 on entity_raw.yaml'
---

# ASR Remote Evaluation

Discover model checkpoints on Azure, generate an evaluation bash script, run it remotely on a Brix node via tmux, and monitor progress.

## When to Use
- User wants to evaluate a trained ASR model (specified by run name) across datasets
- User says "evaluate", "run eval", "eval this model", or "check available steps"
- User wants to discover available checkpoint steps for a model run
- User wants to generate an eval bash script for a model at specific steps
- User wants to run evaluations remotely on a Brix GPU node

## Inputs

| Input | Required | Default | Example |
|-------|----------|---------|---------|
| Run name | Yes | — | `en_domain_hc_egs_cer05_lr1.0_bs256` |
| Eval config YAML | No | `entity_raw.yaml` | `openasr_entity.yaml`, `enus_dtest.yaml` |
| Seed model | No | `4o-mini-asr-v1` | `4o-mini-asr`, `4o-mini` |
| Step interval | No | `1000` | `5000`, `10000` (only used for listing; by default only the latest step is evaluated) |
| Brix node | No (auto) | First Ready `bus-eval-prod*` node | `bus-eval-prod-westus2-a1` |
| Parallelism (`-np`) | No | `8` | `4`, `16` |
| Extra args | No | — | `--tag seg120 --max_segment_seconds 120` |

## Procedure

### Step 0 — Find Ready Brix nodes, check occupancy, and ensure environment is ready

Automatically discover available eval nodes from the `bus-eval-prod*` pool and verify they are not occupied by existing evaluation jobs. Do **not** ask the user for node names unless auto-discovery fails.

For **multi-model evaluation**, you need one unoccupied node per model. For single-model evaluation, you need one node.

1. **List Ready nodes** matching the `bus-eval-prod*` prefix:

   ```bash
   rcall-brix ls 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E '\bReady\b' | awk '{print $1}' | grep '^bus-eval-prod'
   ```

   This filters `rcall-brix ls` output to only `Ready` rows whose pool name starts with `bus-eval-prod`.

2. **Check node occupancy**. For **each** Ready node, check for running `eval_audio` processes:

   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "pgrep -fa \"eval_audio.*\.py\" 2>/dev/null || echo No eval_audio running"'
   ```

   Classify each node:
   - **Occupied**: A running `eval_audio*.py` process exists (active evaluation in progress).
   - **Unoccupied**: No `eval_audio*.py` process. Any background engine still loaded on GPUs will be reused by the next `eval_audio.py` call (via `get_engine_ready`).

   Build a list of **unoccupied** Ready nodes. Also track occupied nodes for reporting.

3. **Assign nodes to models** — each model MUST go on a **different** node for parallel execution.
   - For **single-model** evaluation: pick the **first** unoccupied Ready node.
   - For **multi-model** evaluation: assign each model to a **distinct** unoccupied Ready node. **Never assign two models to the same node** — each evaluation saturates the node's GPUs, so co-locating models would cause resource contention and slow both down. Iterate through the unoccupied node list and pair models 1:1:
     - `RUN_1 → NODE_A`, `RUN_2 → NODE_B`, `RUN_3 → NODE_C`, etc.
   - If there are **fewer unoccupied nodes than models**, assign as many models as there are free nodes and **queue** the remaining models. Queued models will be launched automatically when a node becomes free (see Step 5 — "Handling queued models").
   - If the user explicitly provides node name(s), use those instead (still check occupancy and warn if occupied).
   - If **no** unoccupied Ready `bus-eval-prod*` node is found, report all nodes and their occupancy status and ask the user how to proceed.

   Present the assignment table to the user:
   ```
   Model → Node assignment:
     {RUN_1} → {NODE_A}
     {RUN_2} → {NODE_B}
     {RUN_3} → {NODE_C}
     {RUN_4} → (queued — no available node)
   Occupied nodes: {NODE_X} (eval_audio running)
   ```

4. **Verify the remote environment** on each assigned node. For each assigned `{NODE}`:

   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "cd ~/code/openai && python3 -c \"import speech; import knight; print(\\\"OK\\\")\""'
   ```

   If this fails, install packages first:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "cd ~/code/openai && oaipkg install speech knight"'
   ```

**Always use `oaipkg install`**, never `pip install`.

### Step 1 — Resolve inputs

Extract the **run name(s)** from the user's request. The user may provide:
- A single run name (e.g. `en_domain_hc_egs_cer05_lr1.0_bs256`)
- Multiple run names separated by newlines, commas, or as a list
If not provided, ask.

For **multi-model evaluation**, maintain an ordered list of all run names: `[RUN_1, RUN_2, ..., RUN_N]`.

Determine the **eval config**. Default is `configs/audio_sets/entity_raw.yaml`. The same config applies to all models unless the user specifies per-model configs. Available configs:
- `entity_raw.yaml` — 15 entity domain datasets (Gaming, Insurance, K12, Retail, etc.)
- `entity_raw_short.yaml` — short version of entity_raw
- `openasr_entity.yaml` — OpenASR + entity combined
- `openasr.yaml` — OpenASR datasets
- `enus_dtest.yaml` — en-US dev/test
- `enus_callcenter.yaml` — call center
- `cv.yaml` — CommonVoice
- `ls.yaml` — LibriSpeech

Determine the **seed** (default: `4o-mini-asr-v1`). The seed determines the model base directory:
- `4o-mini-asr-v1` → `az://orngcresco/models/boren/audio_sft/gpt-4o-mini-asr-v1/boren`
- `4o-mini-asr` → `az://orngcresco/models/boren/audio_sft/gpt-4o-mini-asr/boren`

The **Brix node(s)** are resolved automatically in Step 0 from the `bus-eval-prod*` Ready pool. Do not ask the user unless auto-discovery found no Ready nodes.

### Step 2 — Discover available checkpoint steps

List the available checkpoints for **each** run under the model directory. For multi-model evaluation, repeat this for every run name.

For each `{RUN}` in the run list:

**Primary method** — run locally with `bbb ls`:
```bash
bbb ls az://orngwus2cresco/models/boren/audio_sft/gpt-{SEED}/boren/{RUN}/
```

**Fallback** — if `bbb` is not available locally, use Python with blobfile:
```bash
python3 -c "import blobfile as bf; paths = list(bf.glob('az://orngwus2cresco/models/boren/audio_sft/gpt-{SEED}/boren/{RUN}/train_step_*')); [print(p.rstrip('/').split('_')[-1]) for p in sorted(paths)]"
```

**Remote fallback** — run on the Brix node if local tools aren't available:
```bash
rcall-brix ssh {NODE} -- 'bash -l -c "cd ~/code/openai && python3 -c \"import blobfile as bf; paths = list(bf.glob(\\\"az://orngwus2cresco/models/boren/audio_sft/gpt-{SEED}/boren/{RUN}/train_step_*\\\")); [print(p.rstrip(chr(47)).split(chr(95))[-1]) for p in sorted(paths)]\""'
```

This lists directories like `train_step_1000/`, `train_step_2000/`, etc.

Parse the output to extract step numbers. Select **only the latest (max) step** — this is the single checkpoint that will be evaluated. If the user explicitly requests evaluating specific steps or multiple steps, honour that request instead.

Present a **summary table** of all models and their selected step to the user and confirm:
```
Model → Step:
  {RUN_1} → 5000
  {RUN_2} → 6000
  {RUN_3} → 5000
  ...
```

### Step 3 — Generate the eval bash scripts

Generate **one bash script per model**. Create each script at `data/speech/speech/eval/eval_{RUN_SHORT}.sh` where `RUN_SHORT` is a shortened version of the run name (e.g. `en_domain_hc_egs_cer05_lr1.0_bs256` → `domain_hc_cer05`). If the user already has a script file open (like `eval_entity_filter.sh`), update that file instead of creating a new one. Each script follows this template:

```bash
#!/bin/bash

echo "Running evaluations for {RUN}..."
pushd $(dirname $0)

# Discovered steps: {STEP_LIST}
{EVAL_COMMANDS}

popd
```

Each eval command follows the pattern:
```bash
python eval_audio.py configs/audio_sets/{CONFIG}.yaml --run {RUN} -np {NP} --step {STEP}
```

Do **not** include a line without `--step` (latest checkpoint) unless the user explicitly requests it.

Include any extra args (e.g. `--tag`, `--max_segment_seconds`, `--force`) if the user specified them.

### Step 4 — Sync and run remotely in tmux

**Before proceeding**, show the user all generated scripts and their node assignments, and ask for confirmation to sync and run. Do not proceed until the user confirms.

1. **Sync** the local workspace to **each** assigned remote node. Only sync once per unique node (even if multiple models are assigned to the same node in sequential mode):
   ```bash
   rcall-brix sync {NODE}
   ```

2. **Launch each model's eval** on its own dedicated node. For **multi-model evaluation**, iterate through **all** `(RUN, NODE)` assignments and launch them **back-to-back** so all models start running in parallel across different nodes simultaneously:

   ```bash
   # Launch RUN_1 on NODE_A
   rcall-brix ssh {NODE_A} -- 'bash -l -c "cd ~/code/openai && tmux new-session -d -s eval_{RUN_SHORT_1} \"bash data/speech/speech/eval/eval_{RUN_SHORT_1}.sh 2>&1 | tee /tmp/eval_{RUN_SHORT_1}.log\""'
   # Launch RUN_2 on NODE_B
   rcall-brix ssh {NODE_B} -- 'bash -l -c "cd ~/code/openai && tmux new-session -d -s eval_{RUN_SHORT_2} \"bash data/speech/speech/eval/eval_{RUN_SHORT_2}.sh 2>&1 | tee /tmp/eval_{RUN_SHORT_2}.log\""'
   # ... repeat for each (RUN, NODE) pair
   ```

   Each model gets its own tmux session name (`eval_{RUN_SHORT}`) and log file (`/tmp/eval_{RUN_SHORT}.log`) on a **separate** node. This ensures all models evaluate in parallel without GPU contention.

   Report the launched sessions:
   ```
   Launched evaluations (parallel across nodes):
     eval_{RUN_SHORT_1} on {NODE_A}
     eval_{RUN_SHORT_2} on {NODE_B}
     eval_{RUN_SHORT_3} on {NODE_C}
     eval_{RUN_SHORT_4} — QUEUED (will start when a node finishes)
   ```

### Step 5 — Automatically monitor progress every 10 minutes

After launching all tmux sessions, **automatically** poll **all** remote nodes every **10 minutes** until all evaluations finish. Do not wait for the user to ask — start the polling loop immediately after Step 4 completes.

Maintain a tracking table of all `(RUN, RUN_SHORT, NODE, status)` tuples. Each entry starts as `RUNNING`.

**Polling loop** — repeat until **all** entries are FINISHED:

1. **For each `(RUN_SHORT, NODE)` pair still marked RUNNING**, check if the session is still active:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "tmux has-session -t eval_{RUN_SHORT} 2>/dev/null && echo RUNNING || echo FINISHED"'
   ```

2. **If the SSH command fails** for a node (connection refused, timeout, non-zero exit without RUNNING/FINISHED output, or `rcall-brix ssh` errors), the node has likely crashed or been reclaimed. Mark **all models assigned to that node** as needing retry, and trigger the automatic retry procedure described below.

3. **If RUNNING**, report the current dataset being evaluated and a short tail of the log:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "grep -E \"Starting evaluation|Src Path|Dst Path\" /tmp/eval_{RUN_SHORT}.log | tail -6"'
   ```
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "tail -20 /tmp/eval_{RUN_SHORT}.log"'
   ```

4. **If FINISHED**, mark that entry as FINISHED. If there are **queued models** waiting for a free node (see Step 0.3), assign the **next** queued model to the now-free node and launch it (see "Handling queued models" below).

5. **Report a combined status summary** to the user each polling cycle:
   ```
   Eval status (10m check):
     ✓ {RUN_SHORT_1} on {NODE_A} — FINISHED
     ⏳ {RUN_SHORT_2} on {NODE_B} — RUNNING (dataset: en-US-entity-v3-Gaming)
     ⏳ {RUN_SHORT_3} on {NODE_C} — RUNNING (dataset: en-US-entity-v3-Insurance)
     🕐 {RUN_SHORT_4} — QUEUED (waiting for a free node)
   ```

6. **Wait 10 minutes** before the next check. When all entries are FINISHED, proceed to Step 6.

#### Handling queued models

If some models could not be assigned to nodes in Step 0 (more models than available nodes), they are placed in a **queue**. After each polling cycle, check if any node has become free (its eval session finished). If so:

1. Pick the **next queued model**.
2. Assign it to the now-free node.
3. Verify the environment on that node (Step 0.4).
4. The workspace is already synced (from the earlier run on this node), but re-sync to pick up any local changes: `rcall-brix sync {NODE}`.
5. Launch the eval in tmux (Step 4.2).
6. Add the new `(RUN, RUN_SHORT, NODE, RUNNING)` entry to the tracking table.

#### Automatic retry on node crash

If a node failure is detected during monitoring (SSH unreachable, node no longer Ready, or tmux session lost unexpectedly), **automatically retry** the affected model(s) without asking the user. Allow up to **3 retry attempts per model** before giving up.

1. **Inform the user** that the node `{NODE}` appears to be down and a retry is starting for the affected model(s).
2. **Go back to Step 0** — re-run `rcall-brix ls` to discover new Ready `bus-eval-prod*` nodes. Exclude the failed node from candidates. If no Ready unoccupied node is found, wait 2 minutes and try `rcall-brix ls` again (up to 3 attempts). If still no Ready node, add the affected model(s) to the queue.
3. **Re-run the Step 0.2 occupancy check** on every candidate node immediately before assignment — nodes can become occupied between discovery and assignment. Only assign to a node that passes as unoccupied. If the chosen node turns out to be occupied, move to the next candidate.
4. **Assign the affected model(s)** to confirmed-unoccupied node(s).
5. **Re-run Step 0.4** — verify the environment on the new node (`import speech; import knight`), install if needed.
6. **Re-run Step 4** — sync the workspace (`rcall-brix sync {NODE}`) and launch the eval script in tmux on the new node. The eval script uses `eval_audio.py` which **skips datasets that already have results** in Azure blob storage, so previously completed datasets are not re-evaluated.
7. **Resume Step 5** — continue the 10-minute polling loop including the new node.
8. **Increment the retry counter** for the affected model(s). If 3 retries are exhausted for a model and the eval still cannot complete, mark it as FAILED and report the failure to the user with the list of nodes attempted.

**Additional manual commands** (for ad-hoc checks by the user):

To tail the full log for a specific model on its node:
```bash
rcall-brix ssh {NODE} -- 'bash -l -c "tail -50 /tmp/eval_{RUN_SHORT}.log"'
```

To list all running eval sessions on a specific node:
```bash
rcall-brix ssh {NODE} -- 'bash -l -c "tmux ls 2>/dev/null || echo No tmux sessions"'
```

To check status of all active evaluations across all nodes at once:
```bash
for NODE in {NODE_A} {NODE_B} {NODE_C}; do echo "=== $NODE ==="; rcall-brix ssh $NODE -- 'bash -l -c "tmux ls 2>/dev/null || echo No tmux sessions"'; done
```

### Step 6 — Fetch results and generate HTML report

Once **all** tmux sessions are FINISHED (or as each model finishes), fetch the evaluation results locally using `report_summary.py` and render them as an HTML table.

1. **Run `report_summary.py`** for each evaluated model. The model name for results is `{RUN}_step_{STEP}` for each step. Use the run name as a prefix to match all evaluated steps at once:

   ```bash
   cd ~/code/openai/data/speech/speech/eval
   python report_summary.py {RUN} --filter_expr "{EVAL_CONFIG_DATASET_PATTERN}"
   ```

   For **multi-model evaluation**, run this for each `{RUN}` in the run list. You can also pass all run names at once if `report_summary.py` supports multiple prefixes, or run them sequentially.

   - `{RUN}` is the run name (prefix-matches all `{RUN}_step_*` directories under the results root).
   - `--filter_expr` is an optional regex to scope to the eval config's dataset group (e.g. `"en-US-entity-v3"` for `entity_raw.yaml`, `"openasr"` for `openasr.yaml`). Omit if all datasets are wanted.
   - The script writes an Excel report to `~/data/results/report/{RUN}_{TIMESTAMP}.xlsx` and a markdown file alongside it.

2. **Read the markdown reports** and convert them to a standalone **HTML table** for the user. For multi-model evaluation, combine results from all models into a **unified comparison table** with datasets as rows, and columns for each model×step combination. Highlight the best (lowest) WER per dataset in bold across all models.

## Important Notes

- **Always use `rcall-brix`** for remote access (ssh, tmux, sync). Never raw ssh.
- **Always use `oaipkg install speech knight`** if packages need reinstalling on the remote node.
- The remote workspace is `/root/code/openai` (not `/home/boren/...`). The `cd ~/code/openai` resolves to this on remote.
- `bbb ls` uses `az://orngwus2cresco` prefix locally; `eval_audio.py` uses `to_orng()` internally to handle region mapping.
- The `--step` argument must match an existing `train_step_{N}` directory; omitting it uses the latest checkpoint.
- Typical parallelism is `-np 8` for single-node GPU evaluation.
- Results are written to `az://orngcresco/data/boren/data/results/gpt-{SEED}/{RUN}_step_{STEP}/{TEST_NAME}/`.
- **Multi-model evaluation** dispatches one model per unoccupied node. Each model gets its own tmux session and log file. Nodes are checked for running engine processes and eval tmux sessions before assignment to avoid conflicts.
- **Node occupancy** is determined by checking for running `eval_audio*.py` processes via `pgrep`. Any background engine still loaded on GPUs is reusable by the next eval and does not indicate occupancy.
- When a node finishes one model's eval and has queued models waiting, the freed node is automatically assigned the next queued model.

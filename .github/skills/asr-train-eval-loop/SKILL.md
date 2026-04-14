---
name: asr-train-eval-loop
description: 'Automate ASR SFT training with periodic evaluation at every 10% training progress. Use when: launching training with automatic eval checkpoints, train-and-eval loop, progressive evaluation during training, entity_raw.yaml eval at 10% intervals, automated ASR training pipeline with checkpoint evaluation.'
argument-hint: 'Selector pattern and optional eval config, e.g. en_hc_200k_htv1_cer05_mt_wd200_lr1.0_bs256 on entity_raw.yaml'
---

# ASR Train-Eval Loop

Launch ASR SFT training on a `cw-n5-*` node and automatically trigger evaluation on a `bus-eval-prod*` node at every 10% training progress milestone. Combines the `asr-sft-train` and `asr-remote-eval` workflows into a single automated pipeline.

## When to Use
- User wants to train and evaluate at regular intervals during training
- User says "train and eval", "train with eval loop", "progressive eval", or "eval at every 10%"
- User wants to monitor training quality at 10% milestones on `entity_raw.yaml`

## Inputs

| Input | Required | Default | Example |
|-------|----------|---------|---------|
| Selector pattern | Yes | — | `en_hc_200k_htv1_cer05_mt_wd200_lr1.0_bs256` |
| Eval config YAML | No | `entity_raw.yaml` | `openasr_entity.yaml` |
| Seed model | No | `4o-mini-asr-v1` | `4o-mini-asr` |
| Baseline model | No | `baseline` | `baseline`, `en_hc_v1_step_5000` |
| Milestone interval (%) | No | `10` | `5`, `20` |
| Eval parallelism (`-np`) | No | `8` | `4` |
| Train node | No (auto) | First Ready `cw-n5-*` | `cw-n5-i0` |
| Eval nodes | No (auto) | All Ready `bus-eval-prod*` | `bus-eval-prod-westus2-a1` |
| Extra train args | No | — | `--ts --cluster local` |
| Extra eval args | No | — | `--tag seg120` |

## Procedure

### Step 0 — Discover and assign nodes

Two separate node pools are needed: one for training, one for evaluation. Find and assign both before starting.

#### 0a — Find a training node (`cw-n5-*`)

Follow the same node discovery as the `asr-sft-train` skill:

1. List Ready `cw-n5-*` nodes:
   ```bash
   rcall-brix ls 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E '\bReady\b' | awk '{print $1}' | grep 'cw-n5'
   ```

2. Check occupancy on each candidate (look for training processes, tmux sessions, GPU utilization):
   ```bash
   rcall-brix ssh {TRAIN_NODE} -- 'bash -l -c "pgrep -fa \"knight\\|sft\\|run_asr_sft\\|train\" 2>/dev/null || echo No training running"'
   rcall-brix ssh {TRAIN_NODE} -- 'bash -l -c "tmux ls 2>/dev/null || echo No tmux sessions"'
   rcall-brix ssh {TRAIN_NODE} -- 'bash -l -c "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"'
   ```

3. Pick the first unoccupied Ready `cw-n5-*` node as `{TRAIN_NODE}`.

#### 0b — Find eval nodes (`bus-eval-prod*`)

Discover **all** available eval nodes. Multiple milestones may need to evaluate concurrently, so collect as many unoccupied nodes as possible.

1. List Ready `bus-eval-prod*` nodes:
   ```bash
   rcall-brix ls 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E '\bReady\b' | awk '{print $1}' | grep '^bus-eval-prod'
   ```

2. Check occupancy on **each** candidate (look for running `eval_audio` processes):
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "pgrep -fa \"eval_audio.*\.py\" 2>/dev/null || echo No eval_audio running"'
   ```

3. Build an **eval node pool** of all unoccupied Ready nodes (no `eval_audio*.py` process): `[EVAL_NODE_1, EVAL_NODE_2, ...]`. Any background engine still loaded on GPUs will be reused by the next eval. Milestones are assigned round-robin to available nodes. If all nodes are busy, queue the eval until one finishes.

#### 0c — Report assignment

```
Node assignment:
  Training → {TRAIN_NODE} (cw-n5-*, Ready, unoccupied)
  Eval pool → {EVAL_NODE_1}, {EVAL_NODE_2}, ... (bus-eval-prod*, Ready, unoccupied)
  Occupied eval nodes: {NODE_X} (eval_audio running), ...
```

If the training pool has no unoccupied Ready node, report status and ask the user. If the eval pool is empty, warn but proceed — eval nodes will be re-discovered when the first milestone is reached.

### Step 1 — Sync code and verify environments on both nodes

1. Sync to the training node and all eval nodes:
   ```bash
   rcall-brix sync {TRAIN_NODE}
   for NODE in {EVAL_NODE_1} {EVAL_NODE_2} ...; do
     rcall-brix sync $NODE
   done
   ```

2. Verify packages on the training node and each eval node:
   ```bash
   rcall-brix ssh {TRAIN_NODE} -- 'bash -l -c "cd ~/code/openai && python3 -c \"import speech; import knight; print(\\\"OK\\\")\""'
   ```
   ```bash
   for NODE in {EVAL_NODE_1} {EVAL_NODE_2} ...; do
     rcall-brix ssh $NODE -- 'bash -l -c "cd ~/code/openai && python3 -c \"import speech; import knight; print(\\\"OK\\\")\""'
   done
   ```

   If any fails, install with `oaipkg install speech knight`. **Always use `oaipkg install`**, never `pip install`.

### Step 2 — Resolve inputs and compute 10% step milestones

1. **Verify the selector** by listing matching runs:
   ```bash
   rcall-brix ssh {TRAIN_NODE} -- 'bash -l -c "cd ~/code/openai/data/speech/speech/train && python3 run_asr_sft.py --list --selector \"{SELECTOR}\""'
   ```

   Present matching run names and **ask the user to confirm before launching**. Do not proceed until confirmation is received.

2. **Determine total training steps**. The total steps are computed as `ceil(total_examples / batch_size)`. They are logged during training startup as `"Total steps: {N}"`. The total steps are **not known until training starts**, so the milestone table will be computed during Step 4 (monitoring).

3. **Determine milestone interval**. Default: `10` (%). The user may specify a different interval (e.g. `5`, `20`). Milestones will be at `interval%, 2*interval%, ..., 100%`.

4. **Determine eval config**. Default: `configs/audio_sets/entity_raw.yaml`. Resolve to the full config filename.

5. **Determine baseline model**. Default: `baseline`. This is the model name under the results root that all milestone checkpoints are compared against. The baseline must already have eval results at `az://orngwus2cresco/data/boren/data/results/gpt-{SEED}/baseline/`.

6. **Present the plan** to the user and **wait for confirmation**:
   ```
   Plan:
     Selector: {SELECTOR}
     Baseline: {BASELINE} (compared at every milestone)
     Training node: {TRAIN_NODE}
     Eval node pool: {EVAL_NODE_1}, {EVAL_NODE_2}, ...
     Eval config: {EVAL_CONFIG}
     Milestone interval: every {INTERVAL}%
     Milestones: {INTERVAL}%, {2*INTERVAL}%, ..., 100% (exact steps computed after training starts)
   
   Proceed? [y/n]
   ```

### Step 3 — Launch training in tmux

Create a tmux session on `{TRAIN_NODE}` with training on the left pane and GPU monitor on the right.

Where `{SEL_SHORT}` is a shortened version of the selector for log file naming.

1. Create the tmux session:
   ```bash
   rcall-brix ssh {TRAIN_NODE} -- 'bash -l -c "tmux new-session -d -s train -n training"'
   ```

2. Send the training command:
   ```bash
   rcall-brix ssh {TRAIN_NODE} -- 'bash -l -c "tmux send-keys -t train:training \"cd ~/code/openai/data/speech/speech/train && ./run_asr_sft.py --selector \\\"'{SELECTOR}'\\\" --ts --cluster local {EXTRA_TRAIN_ARGS} 2>&1 | tee /tmp/train_{SEL_SHORT}.log\" Enter"'
   ```

3. Split and start GPU monitor:
   ```bash
   rcall-brix ssh {TRAIN_NODE} -- 'bash -l -c "tmux split-window -h -t train:training"'
   rcall-brix ssh {TRAIN_NODE} -- 'bash -l -c "tmux send-keys -t train:training.1 \"cd ~/code/openai/data/speech/speech/train && python3 monitor_gpu.py --interval 5 --no-clear 2>&1 | tee /tmp/gpu_monitor.log\" Enter"'
   ```

Report:
```
Training launched:
  Node: {TRAIN_NODE}
  Tmux session: train
    Left pane (.0): ./run_asr_sft.py --selector "{SELECTOR}" --ts --cluster local
    Right pane (.1): monitor_gpu.py
  Logs: /tmp/train_{SEL_SHORT}.log
```

### Step 4 — Monitor training and trigger evaluation at 10% milestones

This is the **core loop** of the skill. Poll the training node every **5 minutes** and trigger evaluation when new 10% milestones are reached.

#### 4a — Parse total steps (first poll only)

On the first monitoring poll, extract total steps from the training log:

```bash
rcall-brix ssh {TRAIN_NODE} -- 'bash -l -c "grep -oP \"Total steps: \\K[0-9]+\" /tmp/train_{SEL_SHORT}.log | head -1"'
```

This gives `{TOTAL_STEPS}`. Compute the **10% milestone steps** as:

```
For pct in INTERVAL, 2*INTERVAL, 3*INTERVAL, ..., 100:
    milestone_step = round(TOTAL_STEPS * pct / 100 / 100) * 100
```

Each milestone is rounded to the **nearest multiple of 100** (the checkpoint save interval). Deduplicate the list (small runs may have colliding milestones).

Present the milestone table:
```
Total training steps: {TOTAL_STEPS}
10% milestone checkpoint steps: [M1, M2, M3, ..., M10]
```

If `"Total steps:"` is not yet in the log, wait and retry on the next poll cycle.

Also extract the **run name** (which includes the `--ts` timestamp suffix) from the training log:
```bash
rcall-brix ssh {TRAIN_NODE} -- 'bash -l -c "grep -oP \"run_id_base.*?=\\s*\\K\\S+\" /tmp/train_{SEL_SHORT}.log | head -1"'
```

Or detect it from the checkpoint directory listing once the first checkpoint is saved. The run name is needed for eval commands: `{RUN}` = the run_id_base value that appears in `az://orngwus2cresco/models/boren/audio_sft/gpt-{SEED}/boren/{RUN}/train_step_{STEP}/`.

#### 4b — Poll loop (every 5 minutes)

Maintain state:
- `milestones_triggered`: set of milestone steps for which eval has been launched
- `eval_sessions`: list of `(step, tmux_session_name, status)` for tracking running evals

Each cycle:

1. **Check training is still running**:
   ```bash
   rcall-brix ssh {TRAIN_NODE} -- 'bash -l -c "tmux has-session -t train 2>/dev/null && echo RUNNING || echo FINISHED"'
   ```

2. **Get current training step** from the log tail:
   ```bash
   rcall-brix ssh {TRAIN_NODE} -- 'bash -l -c "grep -oP \"step \\K[0-9]+\" /tmp/train_{SEL_SHORT}.log | tail -1"'
   ```

   Alternative patterns to look for in the log:
   ```bash
   rcall-brix ssh {TRAIN_NODE} -- 'bash -l -c "tail -30 /tmp/train_{SEL_SHORT}.log"'
   ```
   Parse the current step from log lines (typically shows `step {N}` or `Step {N}/{TOTAL}`).

3. **Check for newly reached milestones**. For each milestone step `M` in the milestone list:
   - If `M <= current_step` AND `M` not in `milestones_triggered`:
     - **Verify the checkpoint exists** on Azure:
       ```bash
       bbb ls az://orngwus2cresco/models/boren/audio_sft/gpt-{SEED}/boren/{RUN}/train_step_{M}/ 2>/dev/null | head -1
       ```
       If the directory exists, proceed to trigger eval. If not, the checkpoint may still be uploading — retry on next poll.
     - **Find an available eval node** from the pool. Check each node in `eval_node_pool`:
       ```bash
       rcall-brix ssh {NODE} -- 'bash -l -c "pgrep -fa \"eval_audio.*\.py\" 2>/dev/null || echo No eval_audio running"'
       ```
       Pick the first node with no running `eval_audio*.py` process. If all nodes are busy, **queue** the eval — it launches automatically when a node finishes (detected next cycle).
     - **Trigger evaluation** for step `M` on the chosen node (see Step 4c).
     - Add `M` to `milestones_triggered`.

4. **Check status of running evals** across all eval nodes:

   For each eval session in `eval_sessions` still marked RUNNING:
   ```bash
   rcall-brix ssh {ASSIGNED_EVAL_NODE} -- 'bash -l -c "tmux has-session -t eval_step_{STEP} 2>/dev/null && echo RUNNING || echo FINISHED"'
   ```

   If FINISHED:
   - Update status to FINISHED.
   - Show log tail:
     ```bash
     rcall-brix ssh {ASSIGNED_EVAL_NODE} -- 'bash -l -c "tail -20 /tmp/eval_step_{STEP}.log"'
     ```
   - **Run experiment report** for this milestone using the `asr-experiment-report` skill (see Step 4d).
   - **Check the eval queue**: if any milestone evals are queued, launch the next queued eval on this now-free node.
   - Mark the node as available in `eval_node_pool`.

5. **Report combined status**:
   ```
   Train-Eval Loop Status (5m check):
   Training: step {CURRENT}/{TOTAL_STEPS} ({PCT}%) on {TRAIN_NODE}
   
   Milestones:
     ✓ 10% (step {M1}) — eval FINISHED, report generated
     ✓ 20% (step {M2}) — eval FINISHED, report generated
     ⏳ 30% (step {M3}) — eval RUNNING on {EVAL_NODE_1}
     ⏳ 40% (step {M4}) — eval RUNNING on {EVAL_NODE_2}
     📋 50% (step {M5}) — eval QUEUED (all eval nodes busy)
     🕐 60% (step {M6}) — waiting (training at 55%)
     ...
   ```

6. **Termination**: When training is FINISHED **and** all triggered evals are FINISHED (and their reports generated), proceed to Step 5.

#### 4d — Generate experiment report for a finished milestone

Immediately after a milestone eval finishes, invoke the `asr-experiment-report` workflow to fetch metrics and generate analysis artifacts. This runs **locally** (not on a remote node).

The experiment name for each milestone is `{RUN}_step_{STEP}`.
The results root is `az://orngwus2cresco/data/boren/data/results/gpt-{SEED}/`.

1. **Discover datasets** for this milestone:
   ```bash
   bbb ls az://orngwus2cresco/data/boren/data/results/gpt-{SEED}/{RUN}_step_{STEP}/
   ```

2. **Fetch report summary**:
   ```bash
   cd ~/code/openai/data/speech/speech/eval
   python report_summary.py az://orngwus2cresco/data/boren/data/results/gpt-{SEED}/{RUN}_step_{STEP}/
   ```
   This produces `~/data/results/report/{RUN}_step_{STEP}_<timestamp>.xlsx` and `.md`.

3. **Run per-dataset word-error analysis** using the `asr-word-error-analysis` skill script:
   ```bash
   SCRIPT="$HOME/.github/skills/asr-word-error-analysis/scripts/analyze_word_errors.py"
   OUTBASE="$HOME/data/results/word_error_analysis/gpt-{SEED}"
   RESULTS_ROOT="az://orngwus2cresco/data/boren/data/results/gpt-{SEED}/"
   EXPERIMENT="{RUN}_step_{STEP}"

   for dataset in <discovered_datasets>; do
       ds_flat=$(echo "$dataset" | tr '/' '_')
       outdir="$OUTBASE/$EXPERIMENT/$ds_flat"
       [ -f "$outdir/report.html" ] && continue
       python "$SCRIPT" \
           --model "$EXPERIMENT" \
           --dataset "$dataset" \
           --results-root "$RESULTS_ROOT" \
           --output-dir "$outdir" \
           --write-html \
           --top-n 50 \
           --top-confusions 30
   done
   ```

4. **Reshape Excel report** into comparison tables using the `excel-metric-analysis` skill:
   ```bash
   python ~/.github/skills/excel-metric-analysis/scripts/build_metric_sheets.py \
       ~/data/results/report/{RUN}_step_{STEP}_<timestamp>.xlsx \
       --output-dir ~/data/results/report/analysis/
   ```

5. **Compare each dataset against the baseline** using the `asr-detail-compare` skill. For every dataset discovered in sub-step 1:
   ```bash
   COMPARE_SCRIPT="$HOME/.github/skills/asr-detail-compare/scripts/compare_result_details.py"
   RESULTS_ROOT="az://orngwus2cresco/data/boren/data/results/gpt-{SEED}/"
   EXPERIMENT="{RUN}_step_{STEP}"
   OUTBASE="$HOME/data/results/detail_compare/gpt-{SEED}/$EXPERIMENT"

   for dataset in <discovered_datasets>; do
       ds_flat=$(echo "$dataset" | tr '/' '_')
       outdir="$OUTBASE/$ds_flat"
       [ -f "$outdir"/*.topN.html ] && continue
       python "$COMPARE_SCRIPT" \
           --baseline-model {BASELINE} \
           --target-model "$EXPERIMENT" \
           --dataset "$dataset" \
           --results-root "$RESULTS_ROOT" \
           --output-dir "$outdir" \
           --write-html \
           --top-n 50 \
           --join-columns audio_file
   done
   ```

   Parse each `*.summary.json` to get baseline WER/EER and target WER/EER per dataset. Compute the **delta** (`target - baseline`). A positive delta means **regression**.

6. **Run entity error analysis on regression datasets**. For each dataset where the milestone WER or EER is **worse** (higher) than the baseline, run the `asr-entity-error-analysis` skill to diagnose which entities the model gets wrong:

   ```bash
   ENTITY_SCRIPT="$HOME/.github/skills/asr-entity-error-analysis/scripts/analyze_entity_errors.py"
   RESULTS_ROOT="az://orngwus2cresco/data/boren/data/results/gpt-{SEED}/"
   EXPERIMENT="{RUN}_step_{STEP}"
   OUTBASE="$HOME/data/results/entity_error_analysis/gpt-{SEED}/$EXPERIMENT"

   for dataset in <regression_datasets>; do
       ds_flat=$(echo "$dataset" | tr '/' '_')
       outdir="$OUTBASE/$ds_flat"
       [ -f "$outdir/report.html" ] && continue
       python "$ENTITY_SCRIPT" \
           --model "$EXPERIMENT" \
           --dataset "$dataset" \
           --results-root "$RESULTS_ROOT" \
           --output-dir "$outdir" \
           --write-html
   done
   ```

   This produces per-dataset entity error reports highlighting which entity spans (names, terms, etc.) the model transcribes incorrectly. Focus on datasets with the largest WER/EER regression delta.

7. **Present milestone metrics** to the user, always showing the baseline for comparison:
   ```
   Milestone {PCT}% (step {STEP}) — Report Ready:
     Summary: ~/data/results/report/{RUN}_step_{STEP}_<ts>.xlsx
     Baseline: {BASELINE}

     | Dataset        | Baseline WER | Step WER | Δ WER  | Baseline EER | Step EER | Δ EER  |
     |----------------|-------------|----------|--------|-------------|----------|--------|
     | Gaming         | 10.2        | 8.5      | -1.7   | 14.5        | 12.3     | -2.2   |
     | Insurance      | 8.8         | 9.1      | **+0.3** ⚠️ | 13.0  | 14.2     | **+1.2** ⚠️ |
     | ...            | ...         | ...      | ...    | ...         | ...      | ...    |

   Regressions (Δ > 0) — entity error analysis generated:
     ⚠️ Insurance: WER +0.3, EER +1.2 → ~/data/results/entity_error_analysis/gpt-{SEED}/{RUN}_step_{STEP}/Insurance/report.html
   ```

   Track these per-milestone metrics (including baseline) for the final comparison in Step 5.

#### 4c — Trigger evaluation for a milestone step

When a new 10% milestone checkpoint is detected, launch evaluation on `{EVAL_NODE}`.

**Important**: Only one eval can run per node (it saturates all GPUs). If all eval nodes are busy, **queue** the eval. The queued eval launches automatically when a node finishes (detected in the next poll cycle in Step 4b.4).

1. **Select an available eval node** from `eval_node_pool`. If none available, add to `eval_queue` and return.

2. **Re-sync the eval node** (to pick up any local changes since initial sync):
   ```bash
   rcall-brix sync {EVAL_NODE}
   ```

3. **Generate the eval command**:
   ```bash
   python eval_audio.py configs/audio_sets/{EVAL_CONFIG} --run {RUN} -np {NP} --step {STEP} {EXTRA_EVAL_ARGS}
   ```

4. **Launch in tmux on the eval node**:
   ```bash
   rcall-brix ssh {EVAL_NODE} -- 'bash -l -c "cd ~/code/openai/data/speech/speech/eval && tmux new-session -d -s eval_step_{STEP} \"python eval_audio.py configs/audio_sets/{EVAL_CONFIG} --run {RUN} -np {NP} --step {STEP} {EXTRA_EVAL_ARGS} 2>&1 | tee /tmp/eval_step_{STEP}.log\""'
   ```

5. Add `(STEP, eval_step_{STEP}, EVAL_NODE, RUNNING)` to `eval_sessions`. Mark the node as busy.

Report:
```
Eval triggered:
  Milestone: {PCT}% (step {STEP})
  Node: {EVAL_NODE}
  Tmux: eval_step_{STEP}
  Config: {EVAL_CONFIG}
  Log: /tmp/eval_step_{STEP}.log
```

### Step 5 — Final cross-milestone comparison report

Once training is complete and all milestone evals and their reports (Step 4d) are finished, generate a **unified comparison** across all milestones.

1. **Collect per-milestone metrics** from the Excel reports and `summary.json` files generated in Step 4d:
   ```python
   import json, os
   base = os.path.expanduser("~/data/results/word_error_analysis/gpt-{SEED}")
   milestones = {}
   for step in [M1, M2, ..., M10]:
       experiment = f"{RUN}_step_{step}"
       milestones[step] = {}
       for dataset_dir in sorted(os.listdir(os.path.join(base, experiment))):
           with open(os.path.join(base, experiment, dataset_dir, "summary.json")) as f:
               s = json.load(f)
           milestones[step][dataset_dir] = {"wer": s["wer"], "eer": s.get("eer")}
   ```

2. **Present a combined comparison table** with baseline as the first column:
   ```
   Training Progress vs. Evaluation Quality (WER %):
   
   | Dataset        | Baseline | 10% (M1) | 20% (M2) | ... | 100% (M10) | Best Δ vs BL |
   |----------------|----------|----------|----------|-----|------------|--------------|
   | Gaming         | 10.2     | 12.3     | 10.1     | ... | **8.5**    | -1.7         |
   | Insurance      | 8.8      | 15.2     | 13.4     | ... | 9.1        | **+0.3** ⚠️   |
   | ...            | ...      | ...      | ...      | ... | ...        | ...          |
   | **Average**    | 9.5      | 13.8     | 11.7     | ... | **8.8**    | -0.7         |
   ```
   Bold the best (lowest) value per dataset row. Mark with ⚠️ any dataset where the best milestone is still worse than baseline.

3. **Regression summary** — list all datasets that regressed at any milestone:
   ```
   Persistent regressions (best milestone still worse than baseline):
     ⚠️ Insurance: baseline 8.8 → best 9.1 (+0.3) at 100%
     Entity error reports: ~/data/results/entity_error_analysis/gpt-{SEED}/{RUN}_step_*/Insurance/
   
   Recovered regressions (regressed at some milestones but improved later):
     ✓ ScienceTech: regressed at 10% (+1.2) but recovered by 50% (-0.5)
   ```

4. **Highlight trends**:
   - Which datasets improved most vs baseline
   - Whether WER is still decreasing at 100% (suggesting more training may help)
   - Whether WER plateaus or worsens at any milestone (suggesting overfitting)
   - The milestone with the best average WER (may not be 100%)
   - **Regressions**: which datasets never recover from baseline, with pointers to entity error analysis reports

5. **Point to artifacts**:
   ```
   Per-milestone reports: ~/data/results/report/{RUN}_step_*_<ts>.xlsx
   Per-milestone HTML reports: ~/data/results/word_error_analysis/gpt-{SEED}/{RUN}_step_*/
   Baseline comparisons: ~/data/results/detail_compare/gpt-{SEED}/{RUN}_step_*/
   Entity error analysis (regressions): ~/data/results/entity_error_analysis/gpt-{SEED}/{RUN}_step_*/
   Metric comparison sheets: ~/data/results/report/analysis/
   ```

### Step 6 — Handle failures and retries

#### Training crash
If the training node becomes unreachable or the tmux session is lost:
1. Inform the user.
2. Re-discover Ready `cw-n5-*` nodes (Step 0a).
3. Re-sync and re-launch training (Steps 1, 3). Training resumes from the latest checkpoint automatically.
4. Resume the monitoring loop (Step 4).
5. Up to **3 retries** before marking as FAILED.

#### Eval node crash
If an eval node becomes unreachable or an eval tmux session is lost:
1. Remove the crashed node from `eval_node_pool`.
2. Re-discover Ready `bus-eval-prod*` nodes and add any new unoccupied nodes to the pool.
3. Re-assign the failed eval to an available node. Re-sync and re-launch. `eval_audio.py` **skips datasets with existing results**, so partial progress is preserved.
4. Resume monitoring.
5. Up to **3 retries per eval** before marking as FAILED.

## Important Notes

- **Always use `rcall-brix`** for remote access (ssh, tmux, sync). Never raw ssh.
- **Always use `oaipkg install`** for packages. Never `pip install`.
- **Checkpoints save every 100 steps** (`save_checkpoint_interval=100`). Milestones are rounded to multiples of 100.
- **Total steps** are auto-computed as `ceil(total_examples / batch_size)` and logged as `"Total steps: {N}"`.
- **One eval per node at a time**. Multiple eval nodes are used in parallel. Queue evals if all eval nodes are busy.
- The remote workspace is `~/code/openai` (resolves to `/root/code/openai` on remote nodes).
- Training script: `data/speech/speech/train/run_asr_sft.py`.
- Eval script: `data/speech/speech/eval/eval_audio.py`.
- GPU monitor: `data/speech/speech/train/monitor_gpu.py`.
- The `--ts` flag on training adds a timestamp suffix to the run name. The actual run name must be extracted from training logs for eval commands.
- Poll interval is **5 minutes** (faster than the 10-minute interval in standalone skills) to catch milestones promptly.
- **Baseline comparison** is run for every milestone. The baseline model (default: `baseline`) must have existing eval results under the results root. Datasets where the milestone WER/EER is worse than baseline trigger automatic entity error analysis.

## Dependent Skills

- **asr-sft-train**: Node discovery and training launch patterns (Steps 0a, 1, 3)
- **asr-remote-eval**: Node discovery and eval launch patterns (Steps 0b, 4c)
- **asr-experiment-report**: Per-milestone report generation (Step 4d) — fetches `report_summary.py`, runs `asr-word-error-analysis`, and reshapes with `excel-metric-analysis`
- **asr-detail-compare**: Baseline vs milestone utterance-level comparison (Step 4d.5) — generates HTML diff reports per dataset
- **asr-entity-error-analysis**: Entity error analysis on regression datasets (Step 4d.6) — diagnoses entity-level errors on datasets that regressed vs baseline
- **asr-word-error-analysis**: Per-dataset word error analysis HTML reports (invoked within Step 4d)
- **excel-metric-analysis**: Metric comparison table reshaping (invoked within Step 4d)

---
name: verl-asr-run
description: 'Submit and monitor ASR training or evaluation jobs on remote verl Brix nodes until completion, reporting metrics in structured tables. Use when: running RL training (ReMax, GRPO), evaluating checkpoints on LibriSpeech/OpenASR/entity datasets, submitting jobs via quick_run.sh, monitoring Ray job progress, tracking training metrics, pushing code and resubmitting after fixes, and analyzing per-dataset WER with word-level error breakdowns. Triggers: "submit job", "train on remote", "launch training", "run eval", "evaluate on librispeech", "eval_openasr", "check WER", "monitor job", "check training status", "push and submit", "run config on node".'
argument-hint: 'Config name and optional node, e.g. remax_ls_lr05 on verl-n1-i0, or eval_libri_h100'
---

# verl ASR Run

Submit a training or evaluation job on a remote verl Brix node via `submit_job.sh`, **continuously monitor until completion** reporting metrics in structured tables, and optionally perform word error analysis on validation output.

Refer to the **remote-development** skill for node connectivity, `rcall-brix`, `bpush`, `bbb`, and environment setup.

## When to Use

- User wants to **train** an ASR model with RL (ReMax, GRPO) — "run training", "submit job", "train on node"
- User wants to **evaluate** a checkpoint — "run eval", "evaluate on librispeech", "eval_openasr", "check WER"
- User wants to submit any verl ASR job to a remote node and monitor until completion
- User asks to monitor an existing job — "check status", "update", "how's the job"
- User needs to fix code, push, and resubmit after a failure
- User wants word-level error analysis on verl validation JSONL output
- User wants to run multiple jobs (see batch submission below)

## Inputs

| Parameter | Required | Default | Example |
|-----------|----------|---------|---------|
| **config** | Yes | — | `remax_ls_lr05`, `eval_libri_h100`, `gen_libri` |
| **node** | No (auto) | First Ready `verl-*` node | `verl-n1-i0`, `verl-n2-i1` |
| **model_path** | No | From config | `/data/boren/data/ckp/hf_models/Phi4-7b-STT-2603-SR2` |
| **word_error_sets** | No | `1` (first data source only) | `all` (all data sources) |

## Job Types

Determined by config name prefix:

| Prefix | Module | Type | Notes |
|--------|--------|------|-------|
| `remax_*`, `grpo_*` | `recipe.phimm.main_asr_dapo` | Training | RL training with validation at intervals |
| `eval_*` | `recipe.phimm.main_asr_dapo` | Eval-only | `val_only: True`, runs validation then exits |
| `gen_*` | `recipe.phimm.main_asr_gen` | Generation | Inference/generation only |

## Available Configs (examples)

### Training Configs
| Config | Algorithm | Notes |
|--------|-----------|-------|
| `remax_ls_*` | ReMax | LibriSpeech RL training |
| `grpo_*` | GRPO | GRPO RL training |

### Eval Configs
| Config | Datasets | Notes |
|--------|----------|-------|
| `eval_libri_h100` | LibriSpeech (h100 subset) | Fast eval |
| `eval_openasr` | OpenASR (ami, common_voice, earnings22, etc.) | Full OpenASR suite |

Configs live at `recipe/phimm/config/*.yaml`. Training configs compose from `recipe/phimm/config/base/dapo_asr.yaml`; eval configs compose from `recipe/phimm/config/base/eval_asr.yaml` (which sets `val_only: True` and `val_before_train: True`).

## Job Submission Pipeline

```
submit_jobs_repeat.sh  (batch wrapper — calls submit_job.sh N times)
  └─ submit_job.sh <node> <config> [dry_run] [cleanup] [sync_code]
       ├─ rcall-brix sync <node>          # sync code to remote (if sync_code=true)
       ├─ ray_job.py cleanup <config>     # cancel previous run of same config
       └─ rcall-brix ssh <node> "bash -l /root/code/verl/quick_run.sh <config>"
            └─ quick_run.sh <config>
                 ├─ ray_tool.py prepare_env   # install deps on all Ray nodes
                 └─ ray job submit ... python3 -m <module> --config-name <config>
```

## Procedure

### Step 0 — Find a Ready verl node and check occupancy

1. **List all verl nodes** and their status:
   ```bash
   rcall-brix ls 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep '^verl-'
   ```
   Separate into Ready nodes and Paused/Suspended nodes.

2. **Check occupancy** on each Ready node using two signals:

   **a) GPU utilization** — check if GPUs are actively in use:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"'
   ```
   A node is **GPU-busy** if any GPU shows utilization > 5% or memory used > 5000 MiB.

   **b) Ray jobs** — check for running Ray jobs:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "python /root/code/verl/ray_job.py list 2>/dev/null || echo No Ray jobs"'
   ```

   A node is considered **occupied** if either check is positive (GPU-busy OR running Ray jobs). A node is **free** only if GPUs are idle AND no Ray jobs are running.

3. **Assign node**: pick the first free (unoccupied) Ready `verl-*` node. If the user specified a node, use that even if it is busy (see below).

4. **If the chosen node is busy** (GPU-busy or has running Ray jobs):
   - **Do NOT kill or stop the existing job.** Report what is currently running (job ID, config name, runtime, GPU utilization).
   - **Wait for the node to become free.** Poll every **5 minutes** using the occupancy checks from step 2.
   - After each poll, report status: `"Node {NODE} still busy — GPU util: {UTIL}%, Ray jobs: {COUNT}. Waiting..."`.
   - Once the node is idle (GPUs idle AND no running Ray jobs), proceed to Step 1.

5. **If no unoccupied Ready node exists** and no specific node was requested:
   - Check if any `verl-*` nodes are in **Paused** or **Suspended** state:
     ```bash
     rcall-brix ls 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'Paused|Suspended' | awk '{print $1}' | grep '^verl-'
     ```
   - If a Paused/Suspended node is found, **automatically resume** the first one:
     ```bash
     rcall-brix resume {NODE}
     ```
   - **Poll until Ready**: check status every 15 seconds until the node reaches `Ready`:
     ```bash
     rcall-brix ls '{NODE}' 2>&1
     ```
     Report each poll: `"Resuming {NODE}... status: {STATUS}"`.
   - Once `Ready`, use this node and proceed to Step 1.
   - If **no Paused/Suspended nodes** exist either, report the status of all `verl-*` nodes and ask the user which busy node to wait on.

### Step 1 — Resolve inputs

- **config**: Extract from the user's request (required). The config name is the YAML filename without `.yaml` under `recipe/phimm/config/`.
- **Job type**: Determined automatically from config prefix:
  - `gen_*` → generation (uses `main_asr_gen` module)
  - `eval_*` → eval-only (uses `main_asr_dapo` with `val_only: True`)
  - Everything else → training (uses `main_asr_dapo`)
- **model_path**: Usually baked into the config. If the user specifies a custom model path, it will be passed as a hydra override.
- **word_error_sets**: Default `1` — only analyze the first data source. If user says "all", analyze all data sources.

### Step 2 — Push code and submit the job

**IMPORTANT**: Always push the latest code to the remote node before submitting any job. This ensures the remote node runs the same code as your local workspace.

1. **Push code** to the remote node using `bpush`:
   ```bash
   bpush {NODE}
   ```
   This commits and pushes the current workspace to the node's git checkout. Alternatively, use `submit_job.sh` which handles both push and submit (see step 3b).

2. **Clean up** any previous job with the same config name:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "python /root/code/verl/ray_job.py cleanup {CONFIG}"'
   ```

3. **Submit the job** using `quick_run.sh`:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l /root/code/verl/quick_run.sh recipe/phimm/config/{CONFIG}.yaml'
   ```

   **3b. Alternative (preferred)**: Use `submit_job.sh` which handles push + submit in one command:
   ```bash
   bash submit_job.sh {NODE} recipe/phimm/config/{CONFIG}.yaml false true false
   ```

   If a custom model path is specified, append a hydra override:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "cd /root/code/verl && python3 ray_tool.py prepare_env && ray job submit --working-dir=/root/code/verl --no-wait -- python3 -m recipe.phimm.main_asr_dapo --config-name {CONFIG} trainer.experiment_name={CONFIG} actor_rollout_ref.model.path={MODEL_PATH}"'
   ```

   Save the output — it contains the Ray job ID (e.g. `raysubmit_XXXX`).

4. **Capture local log**:
   ```bash
   mkdir -p logs/{NODE}
   ```
   The job output is also streamed to `{CONFIG}.log` on the node.

### Step 3 — Monitor until completion

Poll periodically until the job finishes. **Do NOT stop after a single check — keep monitoring until the job reaches SUCCEEDED or FAILED.**

- **Eval jobs**: Poll every **5 minutes**. Typically takes 10–30 minutes.
- **Training jobs**: Poll every **5 minutes** during startup, then adapt cadence based on step time. Training can take hours to days.

#### 3a. Check Ray job status
```bash
rcall-brix ssh {NODE} -- 'bash -l -c "python /root/code/verl/ray_job.py list"'
```

Or by job ID:
```bash
rcall-brix ssh {NODE} -- 'bash -l -c "ray job status {JOB_ID}"'
```

#### 3b. Tail the log for progress
```bash
rcall-brix ssh {NODE} -- 'bash -l -c "tail -30 /root/code/verl/{CONFIG}.log"'
```

Or via Ray logs:
```bash
rcall-brix ssh {NODE} -- 'bash -l -c "ray job logs {JOB_ID} | tail -n 30"'
```

#### 3c. Check GPU utilization
```bash
rcall-brix ssh {NODE} -- 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
```
Run this on every monitoring check. Parse the CSV output into the status header.

#### 3d. Monitoring phases (check in order)
1. **Startup**: reward functions loaded, dataset loaded, worker count
2. **Model sync**: `bbb sync` fetching model shards from blob
3. **Checkpoint loading**: `Loading checkpoint shards: X%`
4. **LoRA application**: `Applying LoRA to actor module`
5. **vLLM init**: LoRA warnings about vision encoder layers (expected)
6. **CUDA graph capture**: `Capturing CUDA graph shapes: X% | N/M`
7. **W&B init**: `wandb: Syncing run <NAME>` — capture the **run URL**
8. **Training**: `step:N` lines with metrics
9. **Validation**: `val-core/` and `val-aux/` metrics at validation steps
10. **Checkpointing**: `save_checkpoint` timing at save steps

#### 3e. Detect completion

**For eval jobs**, the job is done when:
- `ray_job.py list` shows no running jobs matching `{CONFIG}`, OR
- The log contains `"Initial validation metrics:"` followed by the metrics dict, OR
- The log contains a final WER summary line

**For training jobs**, the job is done when:
- Ray job status shows `SUCCEEDED` or `FAILED`, OR
- The log shows the final step completed and checkpoint saved

#### 3f. Handle failures

If the job status is FAILED or errors appear:
1. Get the error from logs: `ray job logs {JOB_ID} | tail -n 40`
2. Diagnose the root cause (read the traceback)
3. Fix the code locally
4. Push (`bpush {NODE}`) and resubmit (go back to Step 2)
5. Track all job IDs across resubmissions

Common failure patterns:
- **OOM**: reduce `gpu_memory_utilization`, `val_batch_size`, or `rollout_n` in config
- **Missing data**: check `DATA_PATH` env var on the node
- **Missing function**: reward module doesn't have the expected function → add it
- **Import error**: wrong module path or API mismatch → fix the call
- **Model incompatibility**: PEFT/vLLM doesn't support the model → revert model or patch
- **Checkpoint error**: model shard missing on blob → check blob path in config
- **"rcall-brix: command not found"**: Use full path `~/.virtualenvs/openai/bin/rcall-brix`
- **Sync fails**: Check node is Ready with `rcall-brix ls` or the brix-node-gpu-check skill

#### 3g. Extract and present metrics

**Status header** (show on every poll):
```
**Job**: `{JOB_ID}` | **Status**: RUNNING | **Progress**: step X/N (XX%)
**Node**: {NODE} | **Config**: {CONFIG}
**W&B**: [{RUN_NAME}](https://msaip.wandb.io/genai/{PROJECT}/runs/{RUN_ID})
**Ray**: http://{HEAD_IP}:9209
**GPU**: 0: 95% (65G/80G) | 1: 92% (64G/80G) | ...
```

- Extract the W&B URL from logs: look for `wandb: 🚀 View run at https://...`
- Extract the Ray dashboard from logs: look for the Ray dashboard URL in startup output
- GPU line: format each GPU as `{idx}: {util}% ({mem_used}G/{mem_total}G)` from `nvidia-smi` output

**Training metrics table** (for training jobs — accumulate all observed steps):

| Step | Progress | score/mean | entropy | pg_loss | grad_norm | lr | throughput | step_time |
|------|----------|------------|---------|---------|-----------|-----|-----------|-----------|
| 1    | 2.9%     | 0.486      | 1.07    | -0.0008 | 0.0014    | 5e-6| 724 tok/s | 24.6s     |
| 2    | 5.7%     | 0.501      | 1.05    | -0.0012 | 0.0018    | 5e-6| 731 tok/s | 24.2s     |

Extract from `step:N` log lines:
- `critic/score/mean` → score/mean
- `actor/entropy` → entropy
- `actor/pg_loss` → pg_loss
- `actor/grad_norm` → grad_norm
- `actor/lr` → lr (scientific notation)
- `perf/throughput` → throughput (append tok/s)
- `timing_s/step` → step_time (append s)
- `progress` → Progress (format as %)

**Validation metrics table** (for both training and eval — accumulate across val steps):

| Step | Dataset | WER (p_err) | p_ins_edge | Errors | Ref Words | reward/mean |
|------|---------|-------------|------------|--------|-----------|-------------|
| 0    | librispeech | 5.62%   | 5.57       | 165.8  | 29.5      | 0.486       |
| 10   | librispeech | 4.98%   | 4.91       | 148.2  | 27.1      | 0.521       |

Extract from validation log lines:
- `val-aux/{data_source}/p_err/mean@1` → WER (p_err)
- `val-aux/{data_source}/p_ins_edge/mean@1` → p_ins_edge
- `val-aux/{data_source}/n_err/sum@1` → Errors (or `n_err/mean@1`)
- `val-aux/{data_source}/n_ref/sum@1` → Ref Words (or `n_ref/mean@1`)
- `val-core/{data_source}/reward/mean@1` → reward/mean

**Compute WER** per data source: `WER = n_err / n_ref` (or use `p_err` directly).

**Show ALL steps observed — accumulate across monitoring checks to compare progression.**

### Step 4 — Final summary (when SUCCEEDED)

When the job completes, provide:
1. **Full validation metrics table** across ALL val steps (complete trajectory)
2. **Final training metrics** from the last step (for training jobs)
3. **Checkpoint save location** from the `save_checkpoint` log
4. **W&B run link** (clickable)
5. **Total training time** (from first step to last step)
6. **Trend summary**: did WER improve? By how much? Best val step?

#### W&B result check (optional post-job):
```bash
python ./wandb_result.py --metric val-aux search '{CONFIG}'
```
This queries W&B for the run's validation metrics and saves an Excel summary to `~/wandb_results/`.

### Step 5 — Word Error Analysis

After the job completes, perform word-level error analysis on the validation JSONL output.

The validation data is uploaded to Azure blob at:
```
az://orngwus2cresco/data/boren/outputs/{PROJECT}/{CONFIG}/val_data_gen/{DATA_SOURCE}/{STEP}.jsonl
```
where `{PROJECT}` is the `trainer.project_name` from config (default: `verl_repeat`) and `{CONFIG}` is `trainer.experiment_name`.

1. **List available validation JSONL files**:
   ```bash
   bbb ls az://orngwus2cresco/data/boren/outputs/{PROJECT}/{CONFIG}/val_data_gen/
   ```
   Then list per data source:
   ```bash
   bbb ls az://orngwus2cresco/data/boren/outputs/{PROJECT}/{CONFIG}/val_data_gen/{DATA_SOURCE}/
   ```

2. **Download the JSONL files** locally:
   ```bash
   mkdir -p /tmp/verl_eval/{DATA_SOURCE}
   bbb cp az://orngwus2cresco/data/boren/outputs/{PROJECT}/{CONFIG}/val_data_gen/{DATA_SOURCE}/{STEP}.jsonl /tmp/verl_eval/{DATA_SOURCE}/
   ```

3. **Run word error analysis** using the `asr-word-error-analysis` script. The verl validation JSONL uses `gts` for reference and `output` for hypothesis:

   ```bash
   python ~/.github/skills/asr-word-error-analysis/scripts/analyze_word_errors.py \
     --input-path /tmp/verl_eval/{DATA_SOURCE}/{STEP}.jsonl \
     --ref-column gts \
     --hyp-column output \
     --dataset {DATA_SOURCE} \
     --output-dir ~/data/results/verl_word_error/{CONFIG}/{DATA_SOURCE}/ \
     --write-html \
     --top-n 30
   ```

   By default, run analysis on **only the first data source** (e.g. just `librispeech`). If the user requested `all`, run for every data source found.

4. **Present word error analysis results**: Read `summary.json` and show:
   - Overall WER, substitution/deletion/insertion rates
   - Top 10 substitution confusion pairs from `substitutions.csv`
   - Top 5 deletions and insertions
   - Link to the HTML report for visual inspection

#### Step 5b — Fallback if no JSONL dump

If the blob path has no JSONL files (validation data dir not configured or upload failed):

1. The WER metrics are still available from the log (Step 3g). Present those.
2. Suggest re-running with `trainer.validation_data_dir` set in the config or as a hydra override.
3. If the user agrees, re-submit the job with the override and repeat from Step 2.

## Quality Checks

- **Step 2**: `ray job submit` output shows a job ID. If it errors, check `ray_tool.py prepare_env` ran.
- **Step 3 (eval)**: WER values are reasonable (0–100%). If WER > 50%, flag as suspicious.
- **Step 3 (training)**: Training metrics should show learning (score/mean trending up, pg_loss decreasing). If metrics are flat or diverging, flag.
- **Step 5**: `summary.json` exists and WER matches the log metrics (within rounding).
- **Word error HTML**: `report.html` is generated and not empty.

## Important Notes

- The verl configs use hydra. The config search path is `recipe/phimm/config/` with base configs in `recipe/phimm/config/base/`.
- `quick_run.sh` determines the module automatically: configs starting with `gen_*` use `main_asr_gen`, otherwise `main_asr_dapo`.
- Eval configs inherit from `base/eval_asr.yaml` which sets `val_only: True` — only validation runs, no training.
- Training configs inherit from `base/dapo_asr.yaml` (or `grpo_asr.yaml`, `grpo_asr_full.yaml`) — full RL training with periodic validation.
- The remote workspace is at `/root/code/verl` on Brix nodes.
- **Always push code first** before submitting jobs. Use `bpush {NODE}` or `submit_job.sh` (which pushes automatically). Never submit a job without syncing code first.
- For remote operations, use `rcall-brix ssh`, `rcall-brix scp`, or the convenience wrappers `bpush` and `submit_job.sh`.
- The word error analysis script (`analyze_word_errors.py`) supports custom column names via `--ref-column` and `--hyp-column`. verl JSONL uses `gts` and `output`.
- For long-running training jobs, use the **persistent-job-monitor** skill to poll every 5 minutes until a target step is reached.

## Command Reference

| Action | Command |
|--------|---------|
| Push code | `bpush {NODE}` |
| Submit job | `bash submit_job.sh {NODE} recipe/phimm/config/{CONFIG}.yaml false true false` |
| Job status | `rcall-brix ssh {NODE} -- 'bash -l -c "ray job status {JOB_ID}"'` |
| Job logs (tail) | `rcall-brix ssh {NODE} -- 'bash -l -c "ray job logs {JOB_ID} \| tail -n N"'` |
| Step progress | `rcall-brix ssh {NODE} -- 'bash -l -c "ray job logs {JOB_ID} \| grep \"step:\" \| tail -n 10"'` |
| Val metrics | `rcall-brix ssh {NODE} -- 'bash -l -c "ray job logs {JOB_ID} \| grep -E \"val-core\|val-aux\" \| tail -n 30"'` |
| Stop job | `rcall-brix ssh {NODE} -- 'bash -l -c "ray job stop {JOB_ID}"'` |
| Check errors | `rcall-brix ssh {NODE} -- 'bash -l -c "ray job logs {JOB_ID} \| grep -E \"Traceback\|Error\" \| tail -n 20"'` |
| W&B results | `python ./wandb_result.py --metric val-aux search '{CONFIG}'` |
| List jobs | `rcall-brix ssh {NODE} -- 'bash -l -c "python /root/code/verl/ray_job.py list"'` |

## Batch Submission

For submitting multiple jobs to different nodes, edit `submit_jobs_repeat.sh` to list the jobs, or call `submit_job.sh` multiple times:
```bash
bash submit_job.sh <node1> <config1> false true false
bash submit_job.sh <node2> <config2> false true false
```

Or use `submit_jobs_repeat.sh` which wraps multiple `submit_job.sh` calls:
```bash
bash submit_jobs_repeat.sh
```

## Response Style

- **Always show** the status header with job ID, W&B URL, and Ray dashboard URL
- **Use tables** for metrics — never dump raw log lines to the user
- **Accumulate** metrics across monitoring polls — show the full step-by-step progression
- Parse `step:N - key:val - key:val` format into structured table rows
- When monitoring, report the current phase and what to expect next
- On failure, show the error, diagnose, and proceed to fix without asking
- **Do not stop monitoring** until the job is SUCCEEDED or FAILED

## Dependent Skills

| Skill | Phase | Purpose |
|-------|-------|---------|
| **remote-development** | 0, 2 | Node discovery, sync, remote commands |
| **persistent-job-monitor** | 3 | Long-running training job monitoring |
| **asr-word-error-analysis** | 5 | Word-level error analysis on validation JSONL |

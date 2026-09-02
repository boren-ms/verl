---
name: verl-asr-run
description: 'Run the full ASR stack on remote verl Brix nodes: training -> best validation checkpoint HF export -> standard 2609 benchmark evaluation on the same training node -> consolidated report, while continuously monitoring until completion with structured metrics. Use when: running RL training (ReMax, GRPO), automatically running the post-training best-checkpoint 2609 benchmark, evaluating checkpoints on LibriSpeech/in-house/entity datasets, submitting jobs via quick_run.sh, monitoring Ray job progress, tracking training metrics, and pushing code and resubmitting after fixes. Triggers: "submit job", "train on remote", "launch training", "run eval", "evaluate best checkpoint", "evaluate on librispeech", "2609 benchmark report", "check WER", "monitor job", "check training status", "push and submit", "run config on node", "training evaluation report".'
argument-hint: 'Config name and optional node, e.g. remax_ls_lr05 on verl-n1-i0, or eval_libri_h100'
---

# verl ASR Run

Run a full ASR pipeline on remote verl Brix nodes: **training -> select and export the best validation checkpoint after training succeeds -> standard 2609 benchmark suite on the same training node -> final report -> persistent auto-monitor**. Submit jobs via `submit_job.sh`, continuously monitor until completion with structured metrics, and invoke the **eval-2609-benchmark-report** skill once for the best eligible checkpoint only after training has finished successfully and released the training node's GPUs. That skill owns the required in-house DTER, OpenASR-ML, and MixLang evaluations, matching reference baselines, and the final workbook. Persist the latest training and evaluation state in `recipe/phimm/config/verl_job.txt` throughout the pipeline. After submission, always install a `/every 5m update job status and autofix job` schedule (Step 5) so status updates and auto-fixes continue without manual re-prompting.

Refer to the **remote-development** skill for node connectivity, `brix`, `bpush`, `bbb`, and environment setup.

## When to Use

- User wants to **train** an ASR model with RL (ReMax, GRPO) — "run training", "submit job", "train on node"
- User wants to **evaluate** a checkpoint — "run eval", "evaluate on librispeech", "long_eval_inhouse", "check WER"
- User wants to submit any verl ASR job to a remote node and monitor until completion
- User asks to monitor an existing job — "check status", "update", "how's the job"
- User needs to fix code, push, and resubmit after a failure
- User wants full stack execution: training followed by the standard 2609 benchmark suite and consolidated report
- User wants to run multiple jobs (see batch submission below)

## Inputs

| Parameter | Required | Default | Example |
|-----------|----------|---------|---------|
| **config** | Yes | — | `remax_ls_lr05`, `eval_libri_h100`, `gen_libri` |
| **node** | No (auto) | First Ready `verl-*` node | `verl-n1-i0`, `verl-n2-i1` |
| **model_path** | No | From config | `/data/boren/data/ckp/hf_models/Phi4-7b-STT-2603-SR2` |
| **post_train_eval** | No | Run `eval-2609-benchmark-report` once after training succeeds | Best validation checkpoint: in-house DTER + OpenASR-ML + MixLang |
| **report** | No | Consolidated baseline-aware 2609 benchmark workbook | `tmp/eval_2609_reports/<model-label>.xlsx` |

## Job Types

Determined by config name prefix:

| Prefix | Module | Type | Notes |
|--------|--------|------|-------|
| `remax_*` | `recipe.phimm.main_asr_remax` | Training | RL training with validation at intervals |
| `grpo_*` and other training configs | `recipe.phimm.main_asr_dapo` | Training | RL training with validation at intervals |
| `long_eval_*` | `recipe.phimm.main_long_eval_asr` | Long-audio eval | Gen-style eval; SVAD-explode -> generate -> regroup -> TER/EER; writes `result_details.jsonl` + `measures.json` |
| `eval_*` | `recipe.phimm.main_asr_eval` | Eval-only | `val_only: True`, runs validation then exits |
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
| `long_eval_inhouse_2609_all_seg30` | In-house 2605 all locales (en-US, da-DK, hu-HU, nb-NO, nl-NL, cs-CZ; 30-second pre-segmented) | Long-audio gen-style eval; TER/EER measures; required component of the standard 2609 benchmark suite |
| `long_eval_inhouse_2605_enus_seg` | In-house 2605 en-US only (pre-segmented) | Long-audio gen-style eval; en-US TER/EER only |
| `eval_openasr` | OpenASR (ami, common_voice, earnings22, etc.) | Full OpenASR suite |
| `eval_openasr_ml` | OpenASR-ML (FLEURS, MCV, MLS by language) | Multilingual OpenASR suite; report per-language and overall averages |

Configs live under `recipe/phimm/config/` by job family, including `recipe/phimm/config/remax/`, `recipe/phimm/config/eval/`, and `recipe/phimm/config/gen/`. Training configs compose from `recipe/phimm/config/base/dapo_asr.yaml`; eval configs compose from `recipe/phimm/config/base/eval_asr.yaml` (which sets `val_only: True` and `val_before_train: True`).

## Job Submission Pipeline

```
submit_jobs_repeat.sh  (batch wrapper — calls submit_job.sh N times)
  └─ submit_job.sh <node> <config> [dry_run] [cleanup] [sync_code]
       ├─ bpush <node>                    # push code to remote (if sync_code=true)
       ├─ ray_job.py cleanup <config>     # cancel previous run of same config
       └─ brix ssh <node> -- "bash -l /root/code/verl/quick_run.sh <config>"
            └─ quick_run.sh <config>
                 ├─ ray_tool.py prepare_env   # install deps on all Ray nodes
                 └─ ray job submit ... python3 -m <module> --config-name <config>
```

## Procedure

### Step 0 — Find a Ready verl node and check occupancy

1. **List all verl nodes** and their status:
   ```bash
   brix pools 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep '^verl-'
   ```
   Separate into Ready nodes and Paused/Suspended nodes.

2. **Check occupancy** on each Ready node using two signals:

   **a) GPU utilization** — check if GPUs are actively in use:
   ```bash
   brix ssh {NODE} -- 'bash -l -c "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"'
   ```
   A node is **GPU-busy** if any GPU shows utilization > 5% or memory used > 5000 MiB.

   **b) Ray jobs** — check for running Ray jobs:
   ```bash
   brix ssh {NODE} -- 'bash -l -c "python /root/code/verl/ray_job.py list 2>/dev/null || echo No Ray jobs"'
   ```

   A node is considered **occupied** if either check is positive (GPU-busy OR running Ray jobs). A node is **free** only if GPUs are idle AND no Ray jobs are running.

3. **Assign node**: reserve the first free (unoccupied) Ready `verl-*` node for the new job. If the user specified a node and it is occupied, leave its jobs running, report what is currently running, and select another free Ready node. Never pause or stop a node that is occupied by a job.

4. **Pause unused idle nodes**: after reserving the submission node, pause every other Ready `verl-*` node that is free (GPUs idle AND no running Ray jobs):
   ```bash
   brix pause {IDLE_NODE}
   ```
   Do not pause the reserved submission node. Confirm each unused idle node reaches `Paused` or `Suspended` with `brix pools '{IDLE_NODE}' 2>&1` before continuing. A node with any Ray job or unexplained GPU activity is occupied and must remain running.

5. **If no unoccupied Ready node exists**:
   - Check if any `verl-*` nodes are in **Paused** or **Suspended** state:
     ```bash
     brix pools 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'Paused|Suspended' | awk '{print $1}' | grep '^verl-'
     ```
   - If a Paused/Suspended node is available and is known not to contain a job, **automatically resume** the first one and reserve it for submission:
     ```bash
     brix resume {NODE}
     ```
   - **Poll until Ready**: check status every 15 seconds until the node reaches `Ready`:
     ```bash
     brix pools '{NODE}' 2>&1
     ```
     Report each poll: `"Resuming {NODE}... status: {STATUS}"`.
   - Once `Ready`, use this node and proceed to Step 1.
   - If no eligible Paused/Suspended node exists, report the status of all `verl-*` nodes and the unresolved occupancy conflicts; do not pause occupied nodes or submit until a node is free.

### Step 1 — Resolve inputs

- **config**: Extract from the user's request (required). The config name is the YAML filename without `.yaml` under `recipe/phimm/config/`.
- **Job type**: Determined automatically from config prefix:
   - `gen_*` → generation (uses `main_asr_gen` module)
   - `eval_*` → eval-only (uses `main_asr_eval` with `val_only: True`)
   - `remax_*` → ReMax training (uses `main_asr_remax`)
   - Everything else → training (uses `main_asr_dapo`)
- **model_path**: Usually baked into the config. If the user specifies a custom model path, it will be passed as a hydra override.
- **post_train_eval**: For training jobs, wait until training reaches `SUCCEEDED`, confirm its processes released the GPUs, select the best complete saved checkpoint by the rule in Step 3h, export it, and invoke **eval-2609-benchmark-report** exactly once with model label `{TRAIN_CONFIG}@step{BEST_STEP}`, model path `{CHECKPOINT_PATH}`, and `{TRAIN_NODE}`. The benchmark skill runs in-house DTER, OpenASR-ML, and MixLang by default; include digits only when the user explicitly requests them.
- **evaluation node**: Reuse `{TRAIN_NODE}` for the best-checkpoint benchmark. Keep every candidate and reference child job on that node and run them sequentially.
- **report**: The **eval-2609-benchmark-report** skill owns baseline selection, benchmark execution, metric extraction, and the best-checkpoint `.xlsx` workbook. Do not separately run the legacy in-house-only report pipeline.

### Step 1a — Set evaluation topology

Before submitting a standalone `eval_*` or `long_eval_*` job, determine the number of Ready nodes that will participate in that evaluation and set `{EVAL_NNODES}` to that count. Always pass `trainer.nnodes={EVAL_NNODES}` as a Hydra override; do not rely on the inherited training default. For a training pipeline's best-checkpoint evaluation, reuse `{TRAIN_NODE}` and set `trainer.nnodes=1` for every child evaluation job.

After training succeeds, confirm `{TRAIN_NODE}` has no running Ray job and its GPUs are idle before launching evaluation. If training processes are still releasing resources, keep the selected best checkpoint queued and retry every 5 minutes on the same node. Do not select or resume another node.

### Step 1b — Maintain `verl_job.txt` as durable pipeline state

Use `recipe/phimm/config/verl_job.txt` as the canonical local status file for the latest training and evaluation job information. Create it when a pipeline is selected, before submission, and update it immediately after every submission, status poll, phase or progress change, best-checkpoint selection, queue transition, child benchmark launch/completion/failure, resubmission, artifact creation, and final completion. Also update it while the training node is releasing resources, so the recorded state explains what the pipeline is waiting for.

The file must always reflect the latest observed remote state rather than the state expected from a previous poll. Re-discover jobs with `ray_job.py list` before writing scheduled updates. Rewrite the file with `apply_patch`; do not append duplicate snapshots. Keep only the latest job state for each node; never retain superseded jobs or historical job IDs. Once an export or benchmark child starts on the training node, replace its completed training row. Never write credentials, SAS URLs, environment secrets, or raw logs.

The file must contain exactly three elements and no other sections:

1. `updated_at_utc: <ISO-8601 timestamp>`
2. The node/job table
3. The report bullets

Keep the current jobs in a Unicode box-drawing table at the top of the file. Use exactly these columns:

- `NODE`: short Brix node suffix such as `i0`; use `i12-recovery` when that label distinguishes a replacement.
- `RAY JOB ID`: the current Ray submission ID; replace stale IDs in place after resubmission.
- `STATUS`: `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, `STOPPED`, `LOST`, or `FREE`.
- `JOB / CONFIG`: the training config, benchmark config, export label, or report action.
- `PROGRESS / PHASE`: always identify the current activity. For a training Ray job, write `training X/N (P%)`. For an evaluation Ray job, write `evaluating <dataset> (<phase>)`, where `<dataset>` is the actual active dataset or benchmark component (for example, `inhouse_dter`, `openasr_ml`, or `mixlang`) and `<phase>` is concise, such as `model startup`, `decoding`, `uploading artifacts`, or `complete`. Never write a generic evaluation phase without the dataset name. For non-Ray work, write `selecting best checkpoint (<phase>)`, `exporting best checkpoint (<phase>)`, or `building report (<phase>)`.

Render at most one row per node. For each node, keep only its latest Ray submission belonging to the tracked pipeline, replacing that row in place whenever a newer job is submitted, discovered, or created by resubmission. After training succeeds, replace the completed training row with the current export, benchmark child, or report action on the same node. Track the pipeline's current end-to-end state through the `Reports:` status bullet, not additional node rows. Sort the table by the numeric node suffix in natural order (`i2` before `i10`), with an optional label such as `-recovery` sorted immediately after its base suffix. Keep column widths aligned and rewrite the whole table on every poll so it remains directly readable in a terminal. Example:

```text
updated_at_utc: <ISO-8601 timestamp>

┌──────────────┬──────────────────────────────┬───────────────┬──────────────────────────────────────────────────────────────────┬──────────────────────────────┐
│ NODE         │ RAY JOB ID                   │ STATUS        │ JOB / CONFIG                                                     │ PROGRESS / PHASE             │
├──────────────┼──────────────────────────────┼───────────────┼──────────────────────────────────────────────────────────────────┼──────────────────────────────┤
│ i0           │ raysubmit_yG3rtNwycM5RbCQD   │ RUNNING       │ remax_2609v1a_bad_mix13k_openml_verb_s200_bs256_lid05_sfl       │ training 99/200 (50%)        │
└──────────────┴──────────────────────────────┴───────────────┴──────────────────────────────────────────────────────────────────┴──────────────────────────────┘

Reports:
- `training-config` [`i0`]: training 99/200 (50%)
```

Under `Reports:`, use one concise status bullet per training model/run in the form `- model [node]: <status>`, where `node` is the short Brix node suffix used in the table (for example, `i0` or `i12-recovery`). Keep the node current if the pipeline moves to a replacement node. Rewrite the status in place whenever the pipeline advances; never append status history. Use exactly the most specific applicable form:

- `training <step>/<total> (<percent>%)`, or `training startup` before the first step
- `selecting best checkpoint`
- `waiting for best checkpoint step <step>`
- `exporting best checkpoint step <step>`
- `evaluating <dataset> step <step>`
- `building report step <step>`
- `complete step <step>`
- `failed <stage> step <step>`

Use the concrete benchmark dataset in evaluation statuses, such as `inhouse_dter`, `openasr_ml`, or `mixlang`. The bracketed node is required; do not include queue fields, free nodes, Excel files, workbook paths, or other metadata in the bullet. For standalone jobs without a training pipeline, leave `Reports:` empty.

For standalone `eval_*`, `long_eval_*`, or `gen_*` jobs, put the primary job directly in the node/job table. For training pipelines, keep one row for the training node. When candidate/reference benchmark children run sequentially there, each launch replaces the previous row. The file is not complete if the participating node is omitted or appears more than once. The node/job table and report bullets are the canonical representations; do not add metadata, notes, artifact blocks, or history lists.

For every active Ray job, derive `PROGRESS / PHASE` from the job type and current logs on every poll:

- Training: `training <step>/<total> (<percent>%)`; use `training startup` until a step is observed.
- Evaluation: `evaluating <dataset> (<phase>)`; derive `<dataset>` from the active config or data source, not from the parent training model label. For benchmark children, use the concrete component being decoded, such as `inhouse_dter`, `openasr_ml`, `mixlang`, `digits_enus`, or `digits_tier1`.
- Generation: `generating <dataset> (<phase>)`.

Terminal rows preserve the same activity and use `(complete)` or `(failed)` as the phase. The `STATUS` column remains the Ray lifecycle status; it does not replace the activity text.

### Step 2 — Push code and submit the job

**IMPORTANT**: Always push the latest code to the remote node before submitting any job. This ensures the remote node runs the same code as your local workspace.

For training jobs, compose the config locally before submission and record the effective `trainer.save_freq` in the first status update. Preserve that value during submission; post-training evaluation chooses only among complete checkpoints created by that save cadence and must not override it.

1. **Push code** to the remote node using `bpush`:
   ```bash
   bpush {NODE}
   ```
   This commits and pushes the current workspace to the node's git checkout. Alternatively, use `submit_job.sh` which handles both push and submit (see step 3b).

2. **Clean up** any previous job with the same config name:
   ```bash
   brix ssh {NODE} -- 'bash -l -c "python /root/code/verl/ray_job.py cleanup {CONFIG}"'
   ```

3. **Submit the job** using `quick_run.sh`:
   ```bash
   brix ssh {NODE} -- 'bash -l /root/code/verl/quick_run.sh recipe/phimm/config/{CONFIG}.yaml'
   ```

   **3b. Alternative (preferred)**: Use `submit_job.sh` which handles push + submit in one command:
   ```bash
   bash submit_job.sh {NODE} recipe/phimm/config/{CONFIG}.yaml false true true
   ```

   If a custom model path is specified, append a hydra override:
   ```bash
   brix ssh {NODE} -- 'bash -l -c "cd /root/code/verl && python3 ray_tool.py prepare_env && ray job submit --working-dir=/root/code/verl --no-wait -- python3 -m {MODULE} --config-name {CONFIG} trainer.experiment_name={CONFIG} actor_rollout_ref.model.path={MODEL_PATH}"'
   ```

   For `eval_*` and `long_eval_*` jobs requiring overrides, submit directly and include `trainer.nnodes={EVAL_NNODES}`. For the current one-node environment, use `trainer.nnodes=1`:
   ```bash
   brix ssh {NODE} -- 'bash -l -c "cd /root/code/verl && python3 ray_tool.py prepare_env && ray job submit --working-dir=/root/code/verl --no-wait -- python3 -m {MODULE} --config-name {CONFIG} trainer.experiment_name={CONFIG} trainer.nnodes={EVAL_NNODES}"'
   ```

   Save the output — it contains the Ray job ID (e.g. `raysubmit_XXXX`). Immediately record the node, config, Ray job ID, initial status, and job type in `recipe/phimm/config/verl_job.txt` according to Step 1b.

4. **Capture local log**:
   ```bash
   mkdir -p logs/{NODE}
   ```
   The job output is also streamed to `{CONFIG}.log` on the node.

### Step 3 — Monitor until completion

Poll periodically until the job finishes. **Do NOT stop after a single check — keep monitoring until the job reaches SUCCEEDED or FAILED.**

After collecting Ray status, log progress, and GPU utilization on every poll, update `recipe/phimm/config/verl_job.txt` before reporting status to the user. This applies to both training and evaluation jobs, including benchmark child jobs.

- **Eval jobs**: Poll every **5 minutes**. Typically takes 10–30 minutes.
- **Training jobs**: Poll every **5 minutes** during startup, then adapt cadence based on step time. Training can take hours to days.

#### 3a. Check Ray job status
```bash
brix ssh {NODE} -- 'bash -l -c "python /root/code/verl/ray_job.py list"'
```

Or by job ID:
```bash
brix ssh {NODE} -- 'bash -l -c "ray job status {JOB_ID}"'
```

#### 3b. Tail the log for progress
```bash
brix ssh {NODE} -- 'bash -l -c "tail -30 /root/code/verl/{CONFIG}.log"'
```

Or via Ray logs:
```bash
brix ssh {NODE} -- 'bash -l -c "ray job logs {JOB_ID} | tail -n 30"'
```

#### 3c. Check GPU utilization
```bash
brix ssh {NODE} -- 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
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
5. Replace the failed job's row with the replacement Ray ID in `recipe/phimm/config/verl_job.txt` immediately

Common failure patterns:
- **OOM**: reduce `gpu_memory_utilization`, `val_batch_size`, or `rollout_n` in config
- **Missing data**: check `DATA_PATH` env var on the node
- **Missing function**: reward module doesn't have the expected function → add it
- **Import error**: wrong module path or API mismatch → fix the call
- **Model incompatibility**: PEFT/vLLM doesn't support the model → revert model or patch
- **Checkpoint error**: model shard missing on blob → check blob path in config
- **"brix: command not found"**: Ensure `~/.openai/bin` is on PATH
- **Sync fails**: Check node is Ready with `brix pools`

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

**Validation metrics** (for both training and eval — accumulate across val steps):

Only report metrics with the `p_` prefix. Do not include raw count metrics (`n_err`, `n_ref`, `wer`, etc.) or `reward/mean`. All `p_` values are ratios (0–1 in logs); display as percentages (×100) with a `%` suffix on every number.

Extract from validation log lines:
- `val-aux/{data_source}/p_err/mean@1` → p_err
- `val-aux/{data_source}/p_edge/mean@1` → p_edge
- `val-aux/{data_source}/p_fmt/mean@1` → p_fmt
- `val-aux/{data_source}/p_lang/mean@1` → p_lang
- `val-aux/{data_source}/p_bracket/mean@1` → p_bracket

Report each `p_` metric in its own **separate table**:

**p_err table** (primary — always show):

| Step | Dataset | p_err |
|------|---------|-------|
| 0    | librispeech | 5.62% |
| 10   | librispeech | 4.98% |

**p_edge table** (separate — always show):

| Step | Dataset | p_edge |
|------|---------|--------|
| 0    | librispeech | 1.20% |
| 10   | librispeech | 0.98% |

**Quality metrics table** (separate — show only if any value deviates from ideal, i.e. p_fmt < 100%, p_lang < 100%, or p_bracket > 0%):

| Step | Dataset | p_fmt | p_lang | p_bracket |
|------|---------|-------|--------|-----------|
| 0    | librispeech | 99.50% | 99.80% | 0.10% |

If all quality metrics are at their ideal values (p_fmt = 100%, p_lang = 100%, p_bracket = 0%) across all datasets, omit the quality metrics table and note: "Quality metrics nominal (p_fmt=100%, p_lang=100%, p_bracket=0% for all datasets)."

**OpenASR-ML average rows**:
- For `eval_openasr_ml`, always include the original per-dataset rows and add derived average rows grouped by language plus one overall row.
- Infer language from dataset names: `de_*` → German, `fr_*` → French, `it_*` → Italian, `es_*` → Spanish, `pt_*` → Portuguese.
- Compute average p_err as a simple arithmetic mean of `p_err` across dataset rows in that language. Display as percent with `%` suffix.
- Compute average `p_edge` as a simple arithmetic mean across dataset rows in that language.
- The overall average is a simple arithmetic mean across all active OpenASR-ML dataset rows observed for that eval step. Include only datasets present in the metrics; do not invent rows for disabled datasets.

**Show ALL steps observed — accumulate across monitoring checks to compare progression.**

#### 3h. Select, evaluate, and report the best checkpoint

For training jobs, do not launch external checkpoint benchmarks while training is running. On every poll, monitor training only. After the training Ray job reaches `SUCCEEDED`:

1. Set the model's report bullet to `selecting best checkpoint`. Enumerate validation steps from the accumulated `val-aux/{data_source}/p_err/mean@1` metrics and complete saved `global_step_*` checkpoints from the observed `save_checkpoint` entries and configured output root. A checkpoint is eligible only when its save/upload has finished and all required actor shards are present.
2. For every eligible step with validation metrics, compute its selection score as the arithmetic mean of `p_err` over all configured validation datasets reported at that step. Select the lowest score; if scores tie, select the later step. Do not compare steps using different dataset sets: use the largest common configured dataset set present across eligible steps. If no eligible checkpoint has comparable `p_err`, fall back to the latest complete saved checkpoint and explicitly report that fallback.
3. Record `{BEST_STEP}`, its selection score, and the compared dataset set in the user-facing training summary. Set the model's report bullet to `waiting for best checkpoint step <step>`. Before launching work, reconcile that status with the current Ray job and existing export/report artifacts so monitor restarts cannot launch the best-checkpoint evaluation twice.
4. Check `{TRAIN_NODE}` with both Step 0 occupancy signals. If its training processes have not released the GPUs or Ray job slot, retain `waiting for best checkpoint step <step>` and report that the training node is not yet idle.
5. Export the selected best checkpoint using Step 4a, then invoke Step 4b on `{TRAIN_NODE}`.
6. Monitor all benchmark child jobs until the workbook passes quality gates. Set the report bullet to `complete step <step>`, then report its in-house DTER, OpenASR-ML, and MixLang summary metrics, workbook path, checkpoint path, training node, child Ray job IDs, W&B links, and artifact provenance.

Write each pipeline transition to the model's report status bullet. Use `selecting best checkpoint` during selection, `exporting best checkpoint step <step>` before export, `evaluating <dataset> step <step>` for each benchmark child, and `building report step <step>` while creating the workbook. Replace the completed training row with `selecting best checkpoint (<phase>)`, then `exporting best checkpoint (<phase>)` before export, then replace that row with `evaluating <dataset> (<phase>)` for each benchmark child at launch and on every poll, and with `building report (<phase>)` when building the workbook. Never retain more than one row for the training node.

### Step 4 — Training summary (when SUCCEEDED)

When the job completes, provide:
1. **Full validation metrics table** across ALL val steps (complete trajectory)
2. **Final training metrics** from the last step (for training jobs)
3. **Checkpoint save location** from the `save_checkpoint` log
4. **W&B run link** (clickable)
5. **Total training time** (from first step to last step)
6. **Trend summary**: did WER improve? By how much? Which saved validation step was selected, with what aggregate `p_err` and dataset set?

For training jobs, Step 4 is an interim training summary. Continue with the single best-checkpoint evaluation in Step 3h. Do not give the final response until that checkpoint has a successful benchmark workbook.

### Step 4a — Export checkpoint to HF safetensors format

Before invoking the benchmark skill, the selected verl FSDP checkpoint must be converted to HF-compatible safetensors format. Benchmark jobs load the model through `model.path` or `actor_rollout_ref.model.path` using an HF-format directory with `model.safetensors`, `config.json`, tokenizer files, and custom model code, not through `trainer.resume_from_path`.

#### 4a.1 Resolve the selected checkpoint

Derive `{TRAIN_OUTPUT_DIR}` from the training config's `trainer.default_hdfs_dir` or from the observed checkpoint path, usually:

```text
az://orngwus2cresco/data/boren/outputs/{PROJECT}/{TRAIN_CONFIG}
```

Confirm that the selected best `{STEP}` exists and is complete:

```bash
bbb ls {TRAIN_OUTPUT_DIR}/ | grep 'global_step_{STEP}'
```

Do not substitute the last checkpoint after `{BEST_STEP}` has been selected. If the selected checkpoint's upload is still in progress, keep the report status at `waiting for best checkpoint step <step>` and retry on the next monitor poll.

The checkpoint is also available locally on the node at:
```text
/root/checkpoints/{PROJECT}/{TRAIN_CONFIG}/global_step_{STEP}/
```

#### 4a.2 Convert FSDP shards to HF format

Run `convert_verl_to_pt.py` with `--match-lora-merged` to merge LoRA adapters into base weights:

```bash
brix ssh {TRAIN_NODE} -- 'bash -l -c "cd /root/code/verl && python3 plugins/qwen35_audio/hf_model/convert_verl_to_pt.py --input /root/checkpoints/{PROJECT}/{TRAIN_CONFIG}/global_step_{STEP} --output /root/checkpoints/{PROJECT}/{TRAIN_CONFIG}/global_step_{STEP}/qwen_hf/model.pt --match-lora-merged --lora-alpha 640 --lora-rank 320"'
```

**CRITICAL**: Always use `--match-lora-merged --lora-alpha 640 --lora-rank 320`. Without `--match-lora-merged`, LoRA adapters are NOT merged into base weights and the exported model produces ~100% DTER (garbage output). The `--lora-alpha` and `--lora-rank` arguments set the correct LoRA scaling factor (alpha/rank = 2.0) used during training.

#### 4a.3 Convert PyTorch to safetensors and strip `.base_layer.` keys

The `--match-lora-merged` flag wraps LoRA-target linear layer keys with `.base_layer.` (e.g. `model.layers.0.self_attn.q_proj.base_layer.weight`). These must be stripped for HF model loading to work correctly. Convert the PyTorch file to safetensors format and strip the keys in one step:

```bash
brix ssh {TRAIN_NODE} -- 'bash -l -c "cd /root/checkpoints/{PROJECT}/{TRAIN_CONFIG}/global_step_{STEP}/qwen_hf && python3 -c '\''
import torch; from safetensors.torch import save_file; import os
sd = torch.load(\"model.pt\", map_location=\"cpu\", weights_only=False)
new_sd = {k.replace(\".base_layer.\", \".\"): v for k, v in sd.items()}
save_file(new_sd, \"model.safetensors\")
os.remove(\"model.pt\")
print(\"Saved\", len(new_sd), \"tensors as safetensors\")
'\'' "' 
```

**CRITICAL**: If `.base_layer.` keys are NOT stripped, HF loads the model but all LoRA-target weights are randomly initialized, producing ~100% DTER.

#### 4a.4 Copy base model config and tokenizer files

The HF directory needs config, tokenizer, and custom model code files from the base model:

```bash
brix ssh {TRAIN_NODE} -- 'bash -l -c "
   CKPT_DIR=/root/checkpoints/{PROJECT}/{TRAIN_CONFIG}/global_step_{STEP}/qwen_hf
  BASE_MODEL={BASE_MODEL_PATH}
  bbb cpr \$BASE_MODEL /tmp/base_model_files
  for f in config.json tokenizer_config.json tokenizer.json merges.txt vocab.json preprocessor_config.json *.py; do
    cp /tmp/base_model_files/\$f \$CKPT_DIR/ 2>/dev/null || true
  done
  ls \$CKPT_DIR/
"'
```

Where `{BASE_MODEL_PATH}` is the `actor_rollout_ref.model.path` (or `model.path`) from the training config (e.g. `az://orngwus2cresco/.../qwen_hf/`).

#### 4a.5 Upload HF export to blob

```bash
brix ssh {TRAIN_NODE} -- 'bash -l -c "bbb sync /root/checkpoints/{PROJECT}/{TRAIN_CONFIG}/global_step_{STEP}/qwen_hf/ az://orngwus2cresco/data/boren/outputs/{PROJECT}/{TRAIN_CONFIG}/global_step_{STEP}/qwen_hf/"'
```

Set `{CHECKPOINT_PATH}` to `az://orngwus2cresco/data/boren/outputs/{PROJECT}/{TRAIN_CONFIG}/global_step_{STEP}/qwen_hf/` and report both the selected step and path before proceeding.

#### 4a.6 Clear blobfile cache on the node

If any prior failed attempt cached a bad model file, clear it so the eval downloads the fresh export:

```bash
brix ssh {TRAIN_NODE} -- 'bash -l -c "rm -rf /root/.blobfile/*/boren/outputs/{PROJECT}/{TRAIN_CONFIG}/global_step_{STEP}/qwen_hf/"'
```

### Step 4b — Mandatory best-checkpoint 2609 benchmark report

After Step 4a produces `{CHECKPOINT_PATH}` for `{STEP}`, invoke the **eval-2609-benchmark-report** skill on `{TRAIN_NODE}` and let it own evaluation and reporting:

```text
/eval-2609-benchmark-report "{TRAIN_CONFIG}@step{STEP}" "{CHECKPOINT_PATH}" --node "{TRAIN_NODE}" --out "tmp/eval_2609_reports/{TRAIN_CONFIG}_step{STEP}.xlsx"
```

Do not manually submit only `long_eval_inhouse_2609_all_seg30`, run `inhouse-dter-report`, or generate the legacy in-house HTML report. The benchmark skill must run its default required suite:

- in-house micro-DTER with `long_eval_inhouse_2609_all_seg30`
- OpenASR-ML WER / `p_err`
- MixLang DTER / TER
- matching config-defined reference evaluations when canonical compatible reference outputs are unavailable
- one consolidated baseline-aware Excel workbook

Pass `--include-digits-enus` and/or `--include-digits-tier1` only when the user explicitly requested those optional benchmarks. `{TRAIN_NODE}` must pass both idle-node occupancy checks immediately before launch. Every candidate and reference child job must run sequentially on `{TRAIN_NODE}` with `trainer.nnodes=1`.

When **eval-2609-benchmark-report** delegates individual remote jobs back to this skill, execute and monitor those jobs using Steps 0–3. Treat them as child benchmark jobs; do not recursively invoke the benchmark skill again for those child jobs. Continue until every required candidate/reference evaluation succeeds and the workbook passes that skill's quality gates.

Record each delegated candidate/reference job in the node/job table at launch, replacing the previous row for that node. Set `PROGRESS / PHASE` to `evaluating <dataset> (<phase>)`, set the report bullet to `evaluating <dataset> step <step>`, and update both the Ray lifecycle `STATUS` and phase on every poll. After the workbook is validated, set the report bullet to `complete step <step>`.

Per-checkpoint and final response requirements:

1. Report the selected best checkpoint when its workbook passes quality gates; include the training final summary and checkpoint-selection score from Step 4 in the final response.
2. State the evaluated checkpoint path and model label.
3. Include W&B and Ray links for training and benchmark jobs when available.
4. Show the consolidated workbook path and the benchmark skill's summary metrics for in-house DTER, OpenASR-ML, and MixLang, plus optional digits only when requested.
5. Include candidate and reference result locations needed to preserve baseline provenance.

#### W&B result check (optional post-job):
```bash
python ./wandb_result.py --metric val-aux search '{CONFIG}'
```
This queries W&B for the run's validation metrics and saves an Excel summary to `~/wandb_results/`.

### Step 5 — Persistent auto-monitor with `/every` (final step)

After the synchronous monitoring loop in Step 3 returns control to the user (job submitted and still RUNNING, or training job that will run for hours/days), **always** install a recurring background monitor as the last action of the turn so the user does not have to manually re-prompt for status updates.

Maintain exactly one monitor for the active pipeline on a node. Before installing a monitor, inspect active schedules. If the tracked config or Ray job has been stopped, replaced, or resubmitted, stop the stale schedule first and create a new schedule containing the current node, config, and Ray job ID. Never leave a replacement job attached only to a prompt naming an earlier job.

Emit this slash command as the final line of your response, on its own line, with the placeholders resolved and no surrounding code fence or extra text after it:

```
/every 5m update job status and autofix job for node {NODE}, config {CONFIG}, Ray job {JOB_ID}; re-discover and track any replacement job IDs for this pipeline
```

This schedules VS Code Copilot to re-invoke the agent every 5 minutes with an explicit pipeline identity. Each scheduled invocation must:

1. **Load durable state and re-discover the active job(s)**: read `recipe/phimm/config/verl_job.txt`, then run `ray_job.py list` for the tracked `{NODE}` / `{CONFIG}` pair. Treat `{JOB_ID}` as the initial job, not an immutable ID: after an autofix or user-requested replacement, update the schedule to name the replacement config and job ID. After training succeeds, select and track the best-checkpoint evaluation state and every candidate and reference job launched on the same node by **eval-2609-benchmark-report** for in-house DTER, OpenASR-ML, MixLang, and any explicitly requested digits benchmark. Reconcile and rewrite `verl_job.txt` with only the latest job state for the node before printing the user-facing update. Every active row must say `training ...`, `selecting best checkpoint (...)`, `exporting best checkpoint (...)`, `evaluating <dataset> (...)`, or `generating <dataset> (...)` as appropriate; never retain a generic phase from the previous job.
2. **Reprint the status header** from Step 3g (Job ID, status, progress, node, config, W&B URL, Ray URL, GPU utilization) for every active job.
3. **Append new rows** to the training-metrics, `p_err`, `p_edge`, and quality-metrics tables for any new steps observed since the previous poll. Do not re-print the entire historical table — only the new rows, plus a one-line trend summary ("score/mean +0.012 vs last poll, p_err 4.98% → 4.71%").
4. **Autofix on failure** without asking: if any tracked job is `FAILED`, pull the traceback (`ray job logs {JOB_ID} | tail -n 40`), diagnose using the failure patterns in Step 3f, edit the code locally, run `bpush {NODE}`, and resubmit via Step 2. Record the new job ID and continue monitoring it under the same `/every` schedule.
5. **Advance the pipeline** automatically when a stage completes:
   - While training is `RUNNING` → do not enqueue or evaluate any checkpoint.
   - Training `SUCCEEDED` → select the best complete saved checkpoint using Step 3h, reconcile the current report status and artifacts to avoid duplicate work, set `waiting for best checkpoint step <step>`, wait for the training node's GPUs and Ray job slot to become idle, then run Step 4a and Step 4b on that same node.
   - A child benchmark job `SUCCEEDED` → let the benchmark workflow launch its next missing candidate/reference evaluation or, when the best checkpoint's required jobs are complete, build and validate its workbook and report the result.
6. **Replace stale schedules immediately** when a user stops the tracked job, requests a different config, or an autofix creates a new job ID. Stop the old schedule, then install a monitor naming the current `{NODE}`, `{CONFIG}`, and `{JOB_ID}`.
7. **Stop the schedule** with `/every stop` (emitted as the final line) only after the full stack is complete: training `SUCCEEDED`; the best checkpoint was selected and exported; all required 2609 candidate/reference benchmark jobs `SUCCEEDED`; the final `.xlsx` workbook passed the benchmark skill's quality gates and was presented; and the report bullet is `complete step <step>`. Before stopping, write the terminal job row and final report status. Do not add workbook paths under `Reports:` in `recipe/phimm/config/verl_job.txt`. Until then, every scheduled response must end with the explicit `/every 5m ... node {NODE}, config {CONFIG}, Ray job {JOB_ID}` command to keep the loop alive.

**Rules**:
- `/every` lines must be the **very last line** of the assistant message — no trailing prose, no markdown blockquotes, no code fences around them.
- Resolve `{NODE}`, `{CONFIG}`, and `{JOB_ID}` to current values; never emit unresolved placeholders or reuse stale values from an earlier pipeline stage.
- Never install `/every` for a job that has already reached a terminal state at the time of response — emit `/every stop` instead.
- The persistent loop replaces manual user pings; do not ask the user "should I keep checking?" — just schedule it.

## Quality Checks

- **Step 2**: `ray job submit` output shows a job ID. If it errors, check `ray_tool.py prepare_env` ran.
- **Step 3 (eval)**: WER values are reasonable (0–100%). If WER > 50%, flag as suspicious.
- **Step 3 (training)**: Training metrics should show learning (score/mean trending up, pg_loss decreasing). If metrics are flat or diverging, flag.
- **Step 1b/3**: Reopen `recipe/phimm/config/verl_job.txt` and verify it contains only the timestamp, node/job table, and concise model status bullets. Confirm the training node appears exactly once and its row contains only the latest training, export, benchmark, or report job state; no superseded job ID remains. Confirm every Ray job's `PROGRESS / PHASE` starts with `training`, `evaluating <dataset>`, or `generating <dataset>` as appropriate, and that evaluation rows name the actual dataset rather than only a generic phase. Confirm each report bullet has exactly one current status, includes the bracketed short node suffix matching the pipeline's current table row, matches the latest remote state, contains no queue fields, and includes no free-node, Excel-file, or workbook-path metadata.
- **Step 3h**: Verify no external checkpoint evaluation launched before training succeeded; `{BEST_STEP}` is a complete saved checkpoint selected by the documented aggregate-`p_err` rule (or the fallback is explicitly recorded); the report status and artifacts prevent duplicate best-checkpoint work; and every evaluation child ran on the training node with `trainer.nnodes=1` after both occupancy checks showed it idle.
- **Step 4b**: Apply every quality gate from **eval-2609-benchmark-report** to the best-checkpoint workbook. At minimum, verify all required candidate/reference jobs succeeded, the workbook opens, the `summary`, `inhouse_dter`, `openasr_ml`, and `mixlang` sheets are present, and each sheet records matching baseline provenance. Optional digits sheets must appear only when requested.

## Important Notes

- The verl configs use hydra. The config search path is `recipe/phimm/config/` with base configs in `recipe/phimm/config/base/`.
- `quick_run.sh` determines the module automatically: configs starting with `long_eval_*` use `main_long_eval_asr`, configs starting with `gen_*` use `main_asr_gen`, configs starting with `eval_*` use `main_asr_eval`, configs starting with `remax_*` use `main_asr_remax`, and other training configs use `main_asr_dapo`.
- Eval configs inherit from `base/eval_asr.yaml` which sets `val_only: True` — only validation runs, no training.
- `long_eval_*` configs inherit from `base/long_eval_asr.yaml` (gen-style): SVAD-explode -> generate -> regroup per recording -> DisfluencyTolerant TER + entity EER, writing `result_details.jsonl` + `measures.json` under `data.output_path`. They load the model via `model.path` (HF export), not `trainer.resume_from_path`.
- Training configs inherit from `base/dapo_asr.yaml` (or `grpo_asr.yaml`, `grpo_asr_full.yaml`) — full RL training with periodic validation.
- The remote workspace is at `/root/code/verl` on Brix nodes.
- **Always push code first** before submitting jobs. Use `bpush {NODE}` or `submit_job.sh` (which pushes automatically). Never submit a job without syncing code first.
- For remote operations, use `brix ssh`, `brix scp`, or the convenience wrappers `bpush` and `submit_job.sh`.
- For long-running training jobs, use the **persistent-job-monitor** skill to poll every 5 minutes until a target step is reached.

## Command Reference

| Action | Command |
|--------|---------|
| Push code | `bpush {NODE}` |
| Submit job | `bash submit_job.sh {NODE} recipe/phimm/config/{CONFIG}.yaml false true true` |
| Job status | `brix ssh {NODE} -- 'bash -l -c "ray job status {JOB_ID}"'` |
| Job logs (tail) | `brix ssh {NODE} -- 'bash -l -c "ray job logs {JOB_ID} \| tail -n N"'` |
| Step progress | `brix ssh {NODE} -- 'bash -l -c "ray job logs {JOB_ID} \| grep \"step:\" \| tail -n 10"'` |
| Val metrics | `brix ssh {NODE} -- 'bash -l -c "ray job logs {JOB_ID} \| grep -E \"val-core\|val-aux\" \| tail -n 30"'` |
| Stop job | `brix ssh {NODE} -- 'bash -l -c "ray job stop {JOB_ID}"'` |
| Check errors | `brix ssh {NODE} -- 'bash -l -c "ray job logs {JOB_ID} \| grep -E \"Traceback\|Error\" \| tail -n 20"'` |
| Fetch TER/EER measures | `bbb cp {OUTPUT_PATH}/{DATA_SOURCE}/measures.json /tmp/verl_eval/measures.json && cat /tmp/verl_eval/measures.json` |
| W&B results | `python ./wandb_result.py --metric val-aux search '{CONFIG}'` |
| List jobs | `brix ssh {NODE} -- 'bash -l -c "python /root/code/verl/ray_job.py list"'` |

## Batch Submission

For submitting multiple jobs to different nodes, edit `submit_jobs_repeat.sh` to list the jobs, or call `submit_job.sh` multiple times:
```bash
bash submit_job.sh <node1> <config1> false true true
bash submit_job.sh <node2> <config2> false true true
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
- **Do not stop monitoring** until the full stack is complete: training succeeded, the best validation checkpoint was selected, exported, and reported on the same training node, all required candidate/reference jobs from **eval-2609-benchmark-report** succeeded there, and the final workbook plus benchmark summary were generated
- **End every non-terminal response with the explicit pipeline-specific `/every 5m ... node {NODE}, config {CONFIG}, Ray job {JOB_ID}` command** on its own final line (see Step 5). Replace with `/every stop` only once the full stack is complete.

## Dependent Skills

| Skill | Phase | Purpose |
|-------|-------|---------|
| **remote-development** | 0, 2 | Node discovery, sync, remote commands |
| **persistent-job-monitor** | 3 | Long-running training job monitoring |
| **eval-2609-benchmark-report** | 4b | Run the standard 2609 benchmark suite and build the consolidated baseline-aware workbook |

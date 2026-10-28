---
name: verl-asr-run
description: 'Run ASR jobs on remote verl Brix nodes: training, optional best-validation-checkpoint HF export and standard 2609 benchmark evaluation when explicitly requested, or standalone evaluation, while continuously monitoring until completion with structured metrics. Use when: running RL training (ReMax, GRPO), optionally running a requested post-training best-checkpoint 2609 benchmark, evaluating checkpoints on LibriSpeech/in-house/entity datasets, submitting jobs via quick_run.sh, monitoring Ray job progress, tracking training metrics, and pushing code and resubmitting after fixes. Triggers: "submit job", "train on remote", "launch training", "run eval", "evaluate best checkpoint", "evaluate on librispeech", "2609 benchmark report", "check WER", "monitor job", "check training status", "push and submit", "run config on node", "training evaluation report".'
argument-hint: 'Config name and optional node, e.g. remax_ls_lr05 on verl-new-n2-i0, or eval_libri_h100'
---

# verl ASR Run

Run ASR training or evaluation jobs on remote verl Brix nodes. The default training pipeline is **training + figures after each new validation result -> training summary -> persistent auto-monitor completion**. Only when the user explicitly requests post-training evaluation, continue with **select and export the best validation checkpoint -> standard 2609 benchmark suite on the same training node -> final report**. Submit jobs via `submit_job.sh` and continuously monitor until the requested work completes with structured metrics. When post-training evaluation is enabled, invoke the **eval-2609-benchmark-report** skill once for the best eligible checkpoint only after training has finished successfully and released the training node's GPUs. That skill owns the required in-house DTER, OpenASR-ML, and MixLang evaluations, matching reference baselines, and the final workbook. After submission, always install a `/every 30m update job status and autofix job` schedule (Step 5) so status updates and auto-fixes continue without manual re-prompting.

Refer to the **remote-development** skill for node connectivity, `brix`, `bpush`, `bbb`, and environment setup.

For evaluation, start audio precaching early in the background **only on the
main pod**, using Step 1c. It overlaps model startup/export and decoding;
prefetch preparation, completion, or failure must never block evaluation.
For standalone checkpoint evaluation, verify the requested checkpoint/HF
artifacts and provenance, not whether source training reached `SUCCEEDED`.
Source training may continue on a separate authorized pool; leave it untouched
and require only the evaluation target's resources to be idle. When invoked by
[run-openasr-exp](../run-openasr-exp/SKILL.md#readiness-aware-evaluation-queue),
follow its readiness-aware queue: defer missing/incomplete checkpoints and check
the next request without enabling this runner's automatic post-training pipeline.
When this skill is invoked only to edit its instructions, do not launch,
stop, or schedule remote jobs.

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
| **node** | No (auto) | First free Ready `verl-new-n2-i*` pool | `verl-new-n2-i0`, `verl-new-n2-i1` |
| **model_path** | No | From config | `/data/boren/data/ckp/hf_models/Phi4-7b-STT-2603-SR2` |
| **post_train_eval** | No | Disabled; enable only when explicitly requested | Best validation checkpoint: in-house DTER + OpenASR-ML + MixLang |
| **report** | No | Validation figures/comparison report after new validation results; benchmark workbook only when `post_train_eval` is enabled | `tmp/experiment_comparisons/<timestamp>/`; optional `tmp/eval_2609_reports/<model-label>.xlsx` |

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

The default pool family is **`verl-new-n2-i*`**, in cluster
`prod-westus2-cw-6`, namespace `boren`. These are two-pod pools with eight
GPUs per pod (16 GPUs per pool). Treat `{NODE}` as a pool name, not a
single GPU host. Discover current pool/pod identities and Ready GPU-bearing
Ray members before each launch; pool names and old status reports are not
proof of readiness. Respect an explicit user-selected pool; do not silently
fall back to legacy `verl-n1-*` / `verl-n2-*` pools or create new pools.
Scope Brix commands to this cluster and namespace (for example,
`brix ssh -c prod-westus2-cw-6 -n boren {NODE} -- ...`).

1. **List the new pools** and their status:
   ```bash
   brix pools 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E '^verl-new-n2-i[0-9]+[[:space:]]'
   ```
   Separate into Ready nodes and Paused/Suspended nodes.

2. **Check occupancy** on each Ready node using two signals:

   **a) GPU utilization** — resolve every pod in the pool and run this on
   each exact `{POD}` (both pods for `verl-new-n2-i*`), not only the first pod:

   ```bash
   brix ssh -c prod-westus2-cw-6 -n boren {POD} -- 'bash -l -c "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"'
   ```
   A node is **GPU-busy** if any GPU shows utilization > 5% or memory used > 5000 MiB.

   **b) Ray jobs** — check for running Ray jobs:
   ```bash
   brix ssh {NODE} -- 'bash -l -c "python /root/code/verl/ray_job.py list"'
   ```

   A pool is **occupied** if any pod is GPU-busy or Ray has a running/pending job. It is **free** only when every pod is verified idle, Ray has no active jobs, and there is no startup, validation, checkpointing, recovery intent, reservation, or planned launch. Missing occupancy evidence means unavailable, not free.

3. **Assign node**: reserve the first free (unoccupied) Ready `verl-new-n2-i*` pool in numeric index order for the new job. If the user specified a node and it is occupied, leave its jobs running, report what is currently running, and keep the request queued unless the user authorized an alternate pool. Never pause or stop a node that is occupied by a job.

4. **Preserve other pools**: selecting a submission pool does not authorize
   pausing other pools. Leave active workloads, quota queues, reservations,
   recovery intent, and planned launches intact. Pause pools only under a
   separate explicit cleanup instruction and after verifying they are unused.

5. **If no unoccupied Ready node exists**:
   - Check if any `verl-new-n2-i*` pools are in **Paused** or **Suspended** state:
     ```bash
     brix pools 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'Paused|Suspended' | awk '{print $1}' | grep -E '^verl-new-n2-i[0-9]+$'
     ```
   - If a Paused/Suspended node is available and is known not to contain a job, **automatically resume** the eligible one in numeric index order and reserve it for submission:
     ```bash
     brix resume {NODE}
     ```
   - **Poll until Ready**: check status every 15 seconds until the node reaches `Ready`:
     ```bash
     brix pools '{NODE}' 2>&1
     ```
     Report each poll: `"Resuming {NODE}... status: {STATUS}"`.
   - Once `Ready`, use this node and proceed to Step 1.
   - If no eligible Paused/Suspended node exists, report the status of the `verl-new-n2-i*` pools and the unresolved occupancy conflicts; do not pause occupied nodes or submit until a node is free.

### Step 1 — Resolve inputs

- **config**: Extract from the user's request (required). The config name is the YAML filename without `.yaml` under `recipe/phimm/config/`.
- **Job type**: Determined automatically from config prefix:
   - `gen_*` → generation (uses `main_asr_gen` module)
   - `eval_*` → eval-only (uses `main_asr_eval` with `val_only: True`)
   - `remax_*` → ReMax training (uses `main_asr_remax`)
   - Everything else → training (uses `main_asr_dapo`)
- **model_path**: Usually baked into the config. If the user specifies a custom model path, it will be passed as a hydra override.
- **post_train_eval**: Default to disabled. Enable it only when the user explicitly requests post-training evaluation, a 2609 benchmark, or a consolidated post-training report. Do not infer it from a general request to train, submit, or monitor a training job. When enabled, wait until training reaches `SUCCEEDED`, confirm its processes released the GPUs, select the best complete saved checkpoint by the rule in Step 3h, export it, and invoke **eval-2609-benchmark-report** exactly once with model label `{TRAIN_CONFIG}@step{BEST_STEP}`, model path `{CHECKPOINT_PATH}`, and `{TRAIN_NODE}`. The benchmark skill runs in-house DTER, OpenASR-ML, and MixLang by default; include digits only when the user explicitly requests them.
- **evaluation node**: When `post_train_eval` is enabled, reuse `{TRAIN_NODE}` for the best-checkpoint benchmark. Keep every candidate and reference child job on that node and run them sequentially.
- **report**: When `post_train_eval` is enabled, the **eval-2609-benchmark-report** skill owns baseline selection, benchmark execution, metric extraction, and the best-checkpoint `.xlsx` workbook. Do not separately run the legacy in-house-only report pipeline.

### Step 1a — Set training and evaluation topology

Count participating **GPU-bearing Ray members inside the selected pool**,
not the number of Brix pool names. Verify both pods are Ready and registered
with Ray, each exposing eight GPUs. For a normal `verl-new-n2-i*` launch,
`trainer.nnodes=2` and `trainer.n_gpus_per_node=8` (16 GPUs total).
Compose the training config locally and verify this topology before launch;
preserve any explicit user-requested topology and report capacity mismatches
instead of silently changing it. Do not submit a two-node job with only one
Ready Ray member.

For standalone `eval_*` or `long_eval_*`, set `{EVAL_NNODES}` to the verified
participating Ray member count (normally 2 on a new pool) and pass
`trainer.nnodes={EVAL_NNODES}` explicitly. An intentionally single-node
benchmark may retain `trainer.nnodes=1` on the same training pool; reserve
and check occupancy across the entire pool, and run child jobs sequentially.
The best-checkpoint benchmark workflow below retains that single-node policy.

When `post_train_eval` is enabled, after training succeeds confirm `{TRAIN_NODE}` has no running Ray job and its GPUs are idle before launching evaluation. If training processes are still releasing resources, keep the selected best checkpoint queued and retry every 30 minutes on the same node. Do not select or resume another node.

### Step 1c — Early, non-blocking main-pod evaluation audio precache

Use [cache_eval_audio.py](../../../recipe/phimm/cache_eval_audio.py), which
reuses [verl/audio_cache.py](../../../verl/audio_cache.py) for persistent
Orange caching, retries, locks, and atomic writes. Do not add another downloader.

- For standalone evaluation, start inventory preparation and prefetching
  asynchronously as soon as the target main pod and resolved datasets are
  known and current code is synced, ideally before model startup. Immediately
  continue job submission; never wait for inventory preparation or downloads.
- For explicitly requested post-training evaluation, start once training is
  verified running and the future evaluation dataset inventory is known.
  Audio does not depend on checkpoint weights; do not wait for checkpoint
  selection or export. Training-only requests do not enable an external
  evaluation suite or its precache. Delegated pipelines retain their own
  inventory and prefetch task; reconcile and reuse it instead of duplicating it.
- Follow the inventory-normalization procedure in
  [run-openasr-exp](../run-openasr-exp/SKILL.md#early-non-blocking-evaluation-audio-precache).
  The CLI accepts JSONL dataset YAMLs, not Hydra job YAMLs. Resolve the actual
  evaluation datasets and path-producing transforms, including `path_map` and
  `rename_fields`, before creating a cache-only inventory. For compatible
  dataset YAMLs, pass them directly. Unsupported inputs are explicit prefetch
  failures; do not guess paths or drop evaluation datasets.
- Verify the pool's main/head pod identity, evaluation interpreter, user/HOME,
  disk space, and `bbb` authentication. Use `brix ssh` to start exactly one
  background command there, from `/root/code/verl`:

  ```bash
  python -m recipe.phimm.cache_eval_audio \
    <verified-cache-dataset.yaml> --workers 16 --audio-fields audio_path
  ```

  Supply the inventory's effective audio fields if different. Use a verified
  remote background execution mechanism, capture its process identity, and
  verify startup/log activity without waiting for completion. Do not launch
  prefetchers on worker pods, change `HOME`, or claim their caches are populated.
- Store input hashes, main-pod identity, user/HOME/cache root, command, PID,
  logs, status, exit code, and summary under ignored
  `tmp/verl_asr_run/<config>/audio_precache/`, or reuse the parent pipeline's
  durable state. Keep this with the pipeline-specific prefetch state.
  Mark complete only after exit zero and verified inventory/summary counts.
- Monitor/retry independently alongside decoding; failures must not gate
  evaluation or trigger a Ray-job restart. The evaluator's existing on-demand
  cache handles missing audio. Lower concurrency if prefetch I/O harms decoding.
  On recovery, reconcile the main pod and input hashes before launching; never
  duplicate a running task. At pipeline completion, stop only its remaining
  prefetch process by verified exact identity and record cancellation. Optional
  prefetch must not keep evaluation/report completion or the monitor open.

### Step 1b — Track pipeline status

Track the current node, config, Ray job ID, activity, progress, and report state in
conversation context and the recurring monitor prompt. Re-discover live Ray state
on every poll and report changes to the user.

### Step 2 — Push code and submit the job

**IMPORTANT**: Always push the latest code to the remote node before submitting any job. This ensures the remote node runs the same code as your local workspace.

For evaluation, start or reconcile Step 1c's main-pod prefetch after syncing
code, then proceed with submission without waiting for it. For requested
post-training evaluation, start it once training is verified running.

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

   For `eval_*` and `long_eval_*` jobs requiring overrides, submit directly and include `trainer.nnodes={EVAL_NNODES}`. For a fully Ready new two-pod pool, use `{EVAL_NNODES}=2`:
   ```bash
   brix ssh {NODE} -- 'bash -l -c "cd /root/code/verl && python3 ray_tool.py prepare_env && ray job submit --working-dir=/root/code/verl --no-wait -- python3 -m {MODULE} --config-name {CONFIG} trainer.experiment_name={CONFIG} trainer.nnodes={EVAL_NNODES}"'
   ```

   Save the output — it contains the Ray job ID (e.g. `raysubmit_XXXX`). Include the node, config, Ray job ID, initial status, and job type in the user-facing update and recurring monitor prompt.

   When training topology needs explicit overrides, use `quick_run.sh`,
   which forwards trailing Hydra arguments (the wrapper does not expose them):
   ```bash
   brix ssh -c prod-westus2-cw-6 -n boren {NODE} -- 'bash -l /root/code/verl/quick_run.sh {CONFIG_PATH} trainer.nnodes=2 trainer.n_gpus_per_node=8'
   ```
   Resolve `{CONFIG_PATH}` to the actual config file, sync code first, and
   perform cleanup only for the verified previous run of this pipeline.

4. **Capture local log**:
   ```bash
   mkdir -p logs/{NODE}
   ```
   The job output is also streamed to `{CONFIG}.log` on the node.

### Step 3 — Monitor until completion

Poll periodically until the job finishes. **Do NOT stop after a single check — keep monitoring until the job reaches SUCCEEDED or FAILED.**

After collecting Ray status, log progress, and GPU utilization on every poll, report current status to the user. This applies to both training and evaluation jobs, including benchmark child jobs.

Also reconcile Step 1c's background main-pod prefetch when present. Report its
state separately from Ray progress; its failure is not a training/evaluation
failure and must not block decoding or pipeline advancement.

- **Eval jobs**: Poll every **30 minutes**. Typically takes 10–30 minutes.
- **Training jobs**: Poll every **30 minutes**, including during startup. Training can take hours to days.

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
brix ssh -c prod-westus2-cw-6 -n boren {POD} -- 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
```
Run this on every pod in the pool on every monitoring check. Include pod identity in GPU rows so both eight-GPU hosts are visible; do not report first-pod utilization as the whole pool.

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
5. Include the replacement Ray ID in the user-facing update and recurring monitor prompt immediately

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

#### 3g.1. Generate figures when new validation results are ready

On every monitoring check, including the first available validation and the
terminal check, invoke [compare-asr-experiments](../compare-asr-experiments/SKILL.md)
when new usable validation results are available. This is part of training and
standalone validation monitoring, independent of `post_train_eval`; do not wait
for training completion or a saved checkpoint. Use existing results only.

- Treat results as ready when finite, nonnegative dataset `p_err/mean@1`
  observations have been persisted in W&B or verified logs with the actual run,
  dataset, and validation step. An announced validation phase is not a result.
  If only some datasets are ready, plot the available observations and mark the
  remaining coverage pending; refresh when those datasets arrive.
- Pass the exact current run ID/URL, config/code provenance, full available
  histories, and validation-example artifacts to the comparison skill. Include
  the user's comparison/control runs when specified; otherwise show the current
  run's trajectory without inventing a control or selecting unrelated runs.
  With only one observation, show a labeled point and state that no trend is
  established. Missing reward histories or examples are explicit coverage gaps.
- Follow that skill's evidence and reporting requirements: per-dataset `p_err`
  figures, available training reward trajectories and mean-value figures, and
  supported bad-case analysis, with `report.pdf`, `report.md`, `evidence.json`,
  editable source, and reusable figures in a unique report directory. Keep
  datasets separate and running-run findings provisional.
- Persist a report ledger under ignored
  `tmp/verl_asr_run/<config>/validation_figures/`, and include its path in the
  pipeline state and recurring monitor prompt. Record run/segment identity,
  dataset/key/actual-step/value fingerprints, example-artifact versions,
  processed coverage, output paths, and generation status. Compare against this
  ledger to skip unchanged evidence and avoid duplicate reports after monitor
  restarts. Refresh for late or corrected observations and newly ready examples;
  never treat a retry's reused step number as the previous run's result.
- Link new figures and the PDF, Markdown, and evidence in the status update.
  Record retrieval/rendering failures and retry reporting on the next poll;
  report partial deliverables accurately. Figure-generation failures must not
  restart, stop, or delay the remote training/evaluation job. At completion,
  reconcile final validation coverage and disclose any unresolved report gaps.

#### 3h. Optionally select, evaluate, and report the best checkpoint

Run this section only when `post_train_eval` is enabled by an explicit user request. Otherwise, after the training Ray job reaches `SUCCEEDED`, provide the Step 4 training summary, report the terminal training state, and complete the pipeline without selecting, exporting, or externally evaluating a checkpoint.

When enabled, do not launch external checkpoint benchmarks while training is running. On every poll, monitor training only. After the training Ray job reaches `SUCCEEDED`:

1. Set the model's report bullet to `selecting best checkpoint`. Enumerate validation steps from the accumulated `val-aux/{data_source}/p_err/mean@1` metrics and complete saved `global_step_*` checkpoints from the observed `save_checkpoint` entries and configured output root. A checkpoint is eligible only when its save/upload has finished and all required actor shards are present.
2. For every eligible step with validation metrics, compute its selection score as the arithmetic mean of `p_err` over all configured validation datasets reported at that step. Select the lowest score; if scores tie, select the later step. Do not compare steps using different dataset sets: use the largest common configured dataset set present across eligible steps. If no eligible checkpoint has comparable `p_err`, fall back to the latest complete saved checkpoint and explicitly report that fallback.
3. Record `{BEST_STEP}`, its selection score, and the compared dataset set in the user-facing training summary. Set the model's report bullet to `waiting for best checkpoint step <step>`. Before launching work, reconcile that status with the current Ray job and existing export/report artifacts so monitor restarts cannot launch the best-checkpoint evaluation twice.
4. Check `{TRAIN_NODE}` with both Step 0 occupancy signals. If its training processes have not released the GPUs or Ray job slot, retain `waiting for best checkpoint step <step>` and report that the training node is not yet idle.
5. Export the selected best checkpoint using Step 4a, then invoke Step 4b on `{TRAIN_NODE}`.
6. Monitor all benchmark child jobs until the workbook passes quality gates. Set the report bullet to `complete step <step>`, then report its in-house DTER, OpenASR-ML, and MixLang summary metrics, workbook path, checkpoint path, training node, child Ray job IDs, W&B links, and artifact provenance.

Write each pipeline transition to the model's report status bullet. Use `selecting best checkpoint` during selection, `exporting best checkpoint step <step>` before export, `evaluating <dataset> step <step>` for each benchmark child, and `building report step <step>` while creating the workbook. Replace the completed training row with `selecting best checkpoint (<phase>)`, then `exporting best checkpoint (<phase>)` before export, then replace that row with `evaluating <dataset> (<phase>)` for each benchmark child at launch and on every poll, and with `building report (<phase>)` when building the workbook. Never retain more than one row for the training node.

### Step 4 — Training summary (when SUCCEEDED)

When the job completes, provide:
1. **Full validation metrics table** across ALL val steps (complete trajectory), plus the latest Step 3g.1 figures, comparison report, and evidence links
2. **Final training metrics** from the last step (for training jobs)
3. **Checkpoint save location** from the `save_checkpoint` log
4. **W&B run link** (clickable)
5. **Total training time** (from first step to last step)
6. **Trend summary**: did WER improve? By how much? Which saved validation step was selected, with what aggregate `p_err` and dataset set?

When `post_train_eval` is disabled, Step 4 is the final training summary and no benchmark workbook is required. When it is enabled, Step 4 is an interim training summary; continue with the single best-checkpoint evaluation in Step 3h and do not give the final response until that checkpoint has a successful benchmark workbook.

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

### Step 4b — Optional best-checkpoint 2609 benchmark report

Run this step only when `post_train_eval` is enabled. After Step 4a produces `{CHECKPOINT_PATH}` for `{STEP}`, invoke the **eval-2609-benchmark-report** skill on `{TRAIN_NODE}` and let it own evaluation and reporting:

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

Include the main-pod prefetch state path in this same monitor when applicable.
Do not install a separate prefetch schedule. Reconcile its process/logs and
retry independently; never wait for prefetch before advancing evaluation.
Clean up only this pipeline's remaining prefetch process at completion.

Emit this slash command as the final line of your response, on its own line, with the placeholders resolved and no surrounding code fence or extra text after it:

```
/every 30m update job status and autofix job for node {NODE}, config {CONFIG}, Ray job {JOB_ID}; re-discover and track any replacement job IDs for this pipeline
```

This schedules VS Code Copilot to re-invoke the agent every 30 minutes with an explicit pipeline identity. Each scheduled invocation must:

1. **Load durable state and re-discover the active job(s)**: use the conversation context and recurring monitor prompt, then run `ray_job.py list` for the tracked `{NODE}` / `{CONFIG}` pair. Treat `{JOB_ID}` as the initial job, not an immutable ID: after an autofix or user-requested replacement, update the schedule to name the replacement config and job ID. After training succeeds, select and track the best-checkpoint evaluation state and every candidate and reference job launched on the same node by **eval-2609-benchmark-report** only when `post_train_eval` is enabled. Report the latest activity and progress, naming the actual dataset for evaluation and generation jobs.
2. **Reprint the status header** from Step 3g (Job ID, status, progress, node, config, W&B URL, Ray URL, GPU utilization) for every active job.
3. **Refresh validation figures** via Step 3g.1 whenever new validation observations or example artifacts are ready, using the durable report ledger to skip unchanged evidence. **Append new rows** to the training-metrics, `p_err`, `p_edge`, and quality-metrics tables for any new steps observed since the previous poll. Do not re-print the entire historical table — only the new rows, plus a one-line trend summary ("score/mean +0.012 vs last poll, p_err 4.98% → 4.71%").
4. **Autofix on failure** without asking: if any tracked job is `FAILED`, pull the traceback (`ray job logs {JOB_ID} | tail -n 40`), diagnose using the failure patterns in Step 3f, edit the code locally, run `bpush {NODE}`, and resubmit via Step 2. Record the new job ID and continue monitoring it under the same `/every` schedule.
5. **Advance the pipeline** automatically when a stage completes:
   - While training is `RUNNING` → do not enqueue or evaluate any checkpoint.
   - Training `SUCCEEDED` with `post_train_eval` disabled → report the terminal training state, provide the Step 4 summary, and complete the pipeline without checkpoint export or benchmark submission.
   - Training `SUCCEEDED` with `post_train_eval` enabled → select the best complete saved checkpoint using Step 3h, reconcile the current report status and artifacts to avoid duplicate work, set `waiting for best checkpoint step <step>`, wait for the training node's GPUs and Ray job slot to become idle, then run Step 4a and Step 4b on that same node.
   - A child benchmark job `SUCCEEDED` → let the benchmark workflow launch its next missing candidate/reference evaluation or, when the best checkpoint's required jobs are complete, build and validate its workbook and report the result.
6. **Replace stale schedules immediately** when a user stops the tracked job, requests a different config, or an autofix creates a new job ID. Stop the old schedule, then install a monitor naming the current `{NODE}`, `{CONFIG}`, and `{JOB_ID}`.
7. **Stop the schedule** with `/every stop` (emitted as the final line) when all requested work is complete. With `post_train_eval` disabled, this means training reached `SUCCEEDED`, its final summary was presented, and the terminal training state was reported. With `post_train_eval` enabled, this additionally requires the best checkpoint to be selected and exported, all required 2609 candidate/reference benchmark jobs to reach `SUCCEEDED`, the final `.xlsx` workbook to pass the benchmark skill's quality gates and be presented. Until the applicable completion condition is met, every scheduled response must end with the explicit `/every 30m ... node {NODE}, config {CONFIG}, Ray job {JOB_ID}` command to keep the loop alive.

**Rules**:
- `/every` lines must be the **very last line** of the assistant message — no trailing prose, no markdown blockquotes, no code fences around them.
- Resolve `{NODE}`, `{CONFIG}`, and `{JOB_ID}` to current values; never emit unresolved placeholders or reuse stale values from an earlier pipeline stage.
- Never install `/every` for a job that has already reached a terminal state at the time of response — emit `/every stop` instead.
- The persistent loop replaces manual user pings; do not ask the user "should I keep checking?" — just schedule it.

## Quality Checks

- **Step 2**: `ray job submit` output shows a job ID. If it errors, check `ray_tool.py prepare_env` ran.
- **Step 3 (eval)**: WER values are reasonable (0–100%). If WER > 50%, flag as suspicious.
- **Step 3 (training)**: Training metrics should show learning (score/mean trending up, pg_loss decreasing). If metrics are flat or diverging, flag.
- **Step 3g.1**: Verify new validation evidence triggered **compare-asr-experiments**, unchanged evidence was skipped, per-dataset figures and report links are recorded, and final coverage or explicit reporting gaps are included in the terminal summary.
- **Step 1b/3**: Confirm the user-facing status and recurring monitor prompt identify the current node, config, and Ray job ID, with activity and progress matching live state.
- **Step 3h (only when `post_train_eval` is enabled)**: Verify no external checkpoint evaluation launched before training succeeded; `{BEST_STEP}` is a complete saved checkpoint selected by the documented aggregate-`p_err` rule (or the fallback is explicitly recorded); the report status and artifacts prevent duplicate best-checkpoint work; and every evaluation child ran on the training node with `trainer.nnodes=1` after both occupancy checks showed it idle.
- **Step 4b (only when `post_train_eval` is enabled)**: Apply every quality gate from **eval-2609-benchmark-report** to the best-checkpoint workbook. At minimum, verify all required candidate/reference jobs succeeded, the workbook opens, the `summary`, `inhouse_dter`, `openasr_ml`, and `mixlang` sheets are present, and each sheet records matching baseline provenance. Optional digits sheets must appear only when requested.

## Important Notes

- The verl configs use hydra. The config search path is `recipe/phimm/config/` with base configs in `recipe/phimm/config/base/`.
- `quick_run.sh` determines the module automatically: configs starting with `long_eval_*` use `main_long_eval_asr`, configs starting with `gen_*` use `main_asr_gen`, configs starting with `eval_*` use `main_asr_eval`, configs starting with `remax_*` use `main_asr_remax`, and other training configs use `main_asr_dapo`.
- Eval configs inherit from `base/eval_asr.yaml` which sets `val_only: True` — only validation runs, no training.
- `long_eval_*` configs inherit from `base/long_eval_asr.yaml` (gen-style): SVAD-explode -> generate -> regroup per recording -> DisfluencyTolerant TER + entity EER, writing `result_details.jsonl` + `measures.json` under `data.output_path`. They load the model via `model.path` (HF export), not `trainer.resume_from_path`.
- Training configs inherit from `base/dapo_asr.yaml` (or `grpo_asr.yaml`, `grpo_asr_full.yaml`) — full RL training with periodic validation.
- The remote workspace is at `/root/code/verl` on Brix nodes.
- **Always push code first** before submitting jobs. Use `bpush {NODE}` or `submit_job.sh` (which pushes automatically). Never submit a job without syncing code first.
- For remote operations, use `brix ssh`, `brix scp`, or the convenience wrappers `bpush` and `submit_job.sh`.
- For long-running training jobs, use the **persistent-job-monitor** skill to poll every 30 minutes until a target step is reached.

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
- **Do not stop monitoring** until all requested work is complete. For training-only runs, stop after training succeeds and the final training summary and durable terminal state are complete. When `post_train_eval` is enabled, continue until the best validation checkpoint is selected, exported, and reported on the same training node, all required candidate/reference jobs from **eval-2609-benchmark-report** succeed there, and the final workbook plus benchmark summary are generated.
- **End every non-terminal response with the explicit pipeline-specific `/every 30m ... node {NODE}, config {CONFIG}, Ray job {JOB_ID}` command** on its own final line (see Step 5). Replace with `/every stop` only once the full stack is complete.

## Dependent Skills

| Skill | Phase | Purpose |
|-------|-------|---------|
| **remote-development** | 0, 2 | Node discovery, sync, remote commands |
| **persistent-job-monitor** | 3 | Long-running training job monitoring |
| **compare-asr-experiments** | 3g.1, 4 | Refresh validation and reward figures plus evidence-backed comparison reports when new validation results are ready |
| **eval-2609-benchmark-report** | 4b | Run the standard 2609 benchmark suite and build the consolidated baseline-aware workbook |

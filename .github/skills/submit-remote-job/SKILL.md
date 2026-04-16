---
name: submit-remote-job
description: 'Submit and monitor a training or eval job on a remote Brix node until completion. Use when: submitting a job, launching training, running an experiment, starting a Ray job remotely, monitoring job progress, tracking training metrics, pushing code and resubmitting after fixes. Triggers: "submit job", "train on remote", "launch training", "monitor job", "check training status", "push and submit", "run config on node".'
argument-hint: '<node> <config> — e.g. verl-n1-i0 grpo_repeat_remax_ls_wer02_n4_remax_lr05'
---

# Submit Remote Job

Submit a training or eval job to a remote Brix node via `submit_job.sh`, then **continuously monitor until completion**, reporting metrics in structured tables that compare across steps.

Refer to the **remote-development** skill for node connectivity, `rcall-brix`, `bpush`, `bbb`, and environment setup.

## When to Use
- User wants to submit a job (training or eval) with a config on a remote node
- User says "submit job", "train on", "run config on", "launch training", "push and submit"
- User asks to monitor an existing job ("check status", "update", "how's the job")
- User needs to fix code, push, and resubmit after a failure
- User wants to track a job until it finishes
- User wants to run multiple jobs (see batch submission below)

## Prerequisites
- Config files live under `recipe/` (e.g. `recipe/phimm/config/`). The config name is the filename without extension.
- Configs starting with `gen_` use `recipe.phimm.main_asr_gen` module; all others use `recipe.phimm.main_asr_dapo`.

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

### Step 1 — Resolve inputs

Require two inputs:
- **Node name** (required): e.g. `verl-n1-i0`, `h-n2-hpe4`, `l-n1-hpe2`
- **Config name** (required): e.g. `grpo_repeat_remax_ls_wer02_n4_remax_lr05` (no extension)
- **Job ID** (optional): if monitoring an existing job, e.g. `raysubmit_s5pb7fxjMUccbJae` — skip to Step 4

Optional flags (positional after config):
| Arg | Default | Purpose |
|-----|---------|---------|
| `dry_run` | `false` | Print command without executing |
| `cleanup` | `true` | Cancel previous Ray jobs matching this config |
| `sync_code` | `true` | Sync code to remote via `rcall-brix sync` |

If the user doesn't provide a node or config, ask before proceeding.

### Step 2 — Push code to the remote node

Use `bpush` to push the current git state to the remote node:

```bash
bpush <NODE>
```

Wait for the "Pods checked-out" confirmation before proceeding.

### Step 3 — Submit the job

Run from the repo root (`/home/boren/code/verl`):
```bash
bash submit_job.sh <NODE> <CONFIG> false true false
```

- `cleanup=true` stops any previous job with the same config name
- `sync_code=false` because we already pushed with `bpush`
- Logs are saved to `logs/<node>/<config>.log`
- For dry run: `bash submit_job.sh <NODE> <CONFIG> true`

**Capture the job ID** from output: `Job 'raysubmit_XXXXX' submitted successfully`

### Step 4 — Monitor the job (continuous until done)

Poll the job status and logs periodically. **Do NOT stop after a single check — keep monitoring until the job reaches SUCCEEDED or FAILED.**

#### 4a. Check status
```bash
rcall-brix ssh <NODE> -- 'bash -l -c "ray job status <JOB_ID>"'
```

#### 4b. Get latest logs
```bash
rcall-brix ssh <NODE> -- 'bash -l -c "ray job logs <JOB_ID> | tail -n 30"'
```

#### 4c. Get step progression
```bash
rcall-brix ssh <NODE> -- 'bash -l -c "ray job logs <JOB_ID> | grep \"step:\" | tail -n 20"'
```

#### 4d. Get validation metrics
```bash
rcall-brix ssh <NODE> -- 'bash -l -c "ray job logs <JOB_ID> | grep -E \"val-core|val-aux\" | tail -n 30"'
```

#### 4e. Check for errors
```bash
rcall-brix ssh <NODE> -- 'bash -l -c "ray job logs <JOB_ID> | grep -E \"Traceback|Error|TypeError\" | tail -n 20"'
```

#### 4f. Check GPU utilization
```bash
rcall-brix ssh <NODE> -- 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
```
Run this on every monitoring check. Parse the CSV output into the status header.

#### Monitoring phases (check in order):
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

#### Monitoring cadence:
- During startup/loading phases: check every few minutes
- During training steps: check after expected step completion time
- **Always continue monitoring until status is SUCCEEDED or FAILED**

### Step 5 — Report metrics (after every monitoring check)

After each poll, present a **status header** and **accumulated metrics tables**.

#### Status header (always show):

```
**Job**: `<JOB_ID>` | **Status**: RUNNING | **Progress**: step X/N (XX%)
**Node**: <NODE> | **Config**: <CONFIG_NAME>
**W&B**: [<RUN_NAME>](https://msaip.wandb.io/genai/<PROJECT>/runs/<RUN_ID>)
**Ray**: http://<HEAD_IP>:9209
**GPU**: 0: 95% (65G/80G) | 1: 92% (64G/80G) | 2: 88% (63G/80G) | ...
```

- Extract the W&B URL from logs: look for `wandb: 🚀 View run at https://...`
- Extract the Ray dashboard from logs: look for the Ray dashboard URL in startup output
- GPU line: format each GPU as `<idx>: <util>% (<mem_used>G/<mem_total>G)` from `nvidia-smi` output

#### Training metrics table (all observed steps, comparing across time):

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

**Show ALL steps seen so far — accumulate across monitoring checks to compare progression.**

#### Validation metrics table (at val steps, accumulated):

| Step | p_err (WER) | p_ins_edge | n_err | n_ref | reward/mean |
|------|-------------|------------|-------|-------|-------------|
| 0    | 5.62        | 5.57       | 165.8 | 29.5  | 0.486       |
| 10   | 4.98        | 4.91       | 148.2 | 27.1  | 0.521       |
| 20   | 4.71        | 4.64       | 139.5 | 25.8  | 0.548       |

Extract from validation log lines:
- `val-aux/.../p_err/mean@1` → p_err (WER)
- `val-aux/.../p_ins_edge/mean@1` → p_ins_edge
- `val-aux/.../n_err/mean@1` → n_err
- `val-aux/.../n_ref/mean@1` → n_ref
- `val-core/.../reward/mean@1` → reward/mean

**Accumulate all validation steps — show the full trajectory so the user can see trends.**

### Step 6 — Handle failures

If the job status is FAILED or errors appear:
1. Get the error from logs: `ray job logs <JOB_ID> | tail -n 40`
2. Diagnose the root cause (read the traceback)
3. Fix the code locally
4. Push (`bpush <NODE>`) and resubmit (go back to Step 2)
5. Track all job IDs across resubmissions

Common failure patterns:
- **Missing function**: reward module doesn't have the expected function → add it
- **Import error**: wrong module path or API mismatch → fix the call
- **Model incompatibility**: PEFT/vLLM doesn't support the model → revert model or patch
- **OOM**: reduce batch size or rollout_n in config
- **Checkpoint error**: model shard missing on blob → check blob path in config
- **"rcall-brix: command not found"**: Use full path `~/.virtualenvs/openai/bin/rcall-brix`
- **Sync fails**: Check node is Ready with `rcall-brix ls` or the brix-node-gpu-check skill

### Step 7 — Final summary (when SUCCEEDED)

When the job completes, provide:
1. **Full validation metrics table** across ALL val steps (complete trajectory)
2. **Final training metrics** from the last step
3. **Checkpoint save location** from the save_checkpoint log
4. **W&B run link** (clickable)
5. **Total training time** (from first step to last step)
6. **Trend summary**: did WER improve? By how much? Best val step?

#### W&B result check (optional post-job):
```bash
python ./wandb_result.py --metric val-aux search '<CONFIG_NAME>'
```
This queries W&B for the run's validation metrics and saves an Excel summary to `~/wandb_results/`.

## Submitting Multiple Jobs

For batch submission, edit `submit_jobs_repeat.sh` to list the jobs, or call `submit_job.sh` multiple times:
```bash
bash submit_job.sh <node1> <config1>
bash submit_job.sh <node2> <config2>
```

## Command Reference

| Action | Command |
|--------|---------|
| Push code | `bpush <NODE>` |
| Submit job | `bash submit_job.sh <NODE> <CONFIG> false true false` |
| Job status | `rcall-brix ssh <NODE> -- 'bash -l -c "ray job status <JOB_ID>"'` |
| Job logs (tail) | `rcall-brix ssh <NODE> -- 'bash -l -c "ray job logs <JOB_ID> \| tail -n N"'` |
| Step progress | `rcall-brix ssh <NODE> -- 'bash -l -c "ray job logs <JOB_ID> \| grep \"step:\" \| tail -n 10"'` |
| Val metrics | `rcall-brix ssh <NODE> -- 'bash -l -c "ray job logs <JOB_ID> \| grep -E \"val-core\|val-aux\" \| tail -n 30"'` |
| Stop job | `rcall-brix ssh <NODE> -- 'bash -l -c "ray job stop <JOB_ID>"'` |
| Check errors | `rcall-brix ssh <NODE> -- 'bash -l -c "ray job logs <JOB_ID> \| grep -E \"Traceback\|Error\" \| tail -n 20"'` |
| W&B results | `python ./wandb_result.py --metric val-aux search '<CONFIG_NAME>'` |
| List jobs | `rcall-brix ssh <NODE> -- 'bash -l -c "python /root/code/verl/ray_job.py list"'` |

## Response Style
- **Always show** the status header with job ID, W&B URL, and Ray dashboard URL
- **Use tables** for metrics — never dump raw log lines to the user
- **Accumulate** metrics across monitoring polls — show the full step-by-step progression
- Parse `step:N - key:val - key:val` format into structured table rows
- When monitoring, report the current phase and what to expect next
- On failure, show the error, diagnose, and proceed to fix without asking
- **Do not stop monitoring** until the job is SUCCEEDED or FAILED

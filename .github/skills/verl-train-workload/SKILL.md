---
name: verl-train-workload
description: 'Recover, monitor, and continue the current verl 2607 training workload across the designated Brix pools. Use when: recover training workload, restore verl jobs, resume current training, rebuild the training monitor, show the training nodes table, recover after Copilot/session restart, or autofix the active 2607 ReMax/GDPO jobs.'
argument-hint: 'Optional action: status, recover, reinstall-monitor, or autofix'
---

# verl Train Workload

Recover and manage the current multi-node verl training workload after a session,
machine, schedule, or Ray-job interruption. Live Brix and Ray state is authoritative;
the job IDs below are initial provenance only and must never be treated as immutable.

Use the `verl-asr-run` skill for job execution, monitoring, checkpoint export, and
post-training benchmarks. Use `remote-development` for Brix node creation, resume,
code sync, and remote commands.

## Training pool

Always include every pool in the status table, even when idle or unavailable:

| Pool | Capacity | Expected workload |
|---|---:|---|
| `verl-n1-i0` | 8 GPUs | `remax_2607v1_openml_verb_s100_bs256_rn32_lid` |
| `verl-n1-i2` | 8 GPUs | `remax_2607v1_openml_verb_s100_bs256_lid_clr` |
| `verl-n1-i4` | 8 GPUs | `gdpo_2607v1_openml_verb_s100_bs256_lid` |
| `verl-n1-i5` | 8 GPUs | `remax_2607v1_mixcv15_openml3t_s200_bs256_scale2_lid0_swtich` |
| `verl-n1-i9` | 8 GPUs | `remax_2607v1_mix_openml_verb_s200_bs256_scale2_lid0_ilv3` |
| `verl-n1-i14` | 8 GPUs | `remax_2607v1_mixcv15_10k_openml_verb_s200_bs256_scale2_lid0_ilv` |
| `verl-n1-i15` | 8 GPUs | `remax_2607v1_mix_openml_verb_s200_bs256_scale2_lid0_ilv_sfl` |
| `verl-n2-i2` | 16 GPUs, 2 pods | Discover and preserve its live workload; never overwrite an unrelated job. |

All configs are under `recipe/phimm/config/ver_2607v1/`.

## Initial tracked submissions

These IDs identify the original pipeline but may be stale after an autofix:

| Pool | Config | Initial Ray job |
|---|---|---|
| `verl-n1-i4` | `gdpo_2607v1_openml_verb_s100_bs256_lid` | `raysubmit_afGMBC3Hys899XEp` |
| `verl-n1-i14` | `remax_2607v1_mixcv15_10k_openml_verb_s200_bs256_scale2_lid0_ilv` | `raysubmit_rpQzUqHfw9s8WGh9` |
| `verl-n1-i15` | `remax_2607v1_mix_openml_verb_s200_bs256_scale2_lid0_ilv_sfl` | `raysubmit_et15scEsdCVExhNQ` |

## Recovery procedure

### 1. Discover pool state

```bash
brix pools 2>&1 | sed 's/\x1b\[[0-9;]*m//g' |
  grep -E '^verl-(n1-i(0|2|4|5|9|14|15)|n2-i2)[[:space:]]'
```

For a missing pool, confirm it does not exist before creating it. Resume Paused or
Suspended pools with `brix resume POOL`, then poll `brix ls POOL` until Ready.
Do not recreate or replace a pool that is Assigning or Scheduled.

### 2. Rediscover live jobs

For every Ready pool:

```bash
brix ssh POOL -- 'bash -l -c "python /root/code/verl/ray_job.py list"'
brix ssh POOL -- 'bash -l -c "ray job list"'
brix ssh POOL -- 'bash -l -c "ray status"'
```

Match jobs by config name and experiment name, not only by Ray ID. Record replacement
IDs created by previous recovery attempts. A pool is free only when it has no running
Ray submission and GPUs are idle.

### 3. Print the workload table

Every status/recovery response must begin with one row per pool and these columns:

| Node/Pool | Pool State | Job Status | Config | Ray Job ID | Step/Total | Progress % | GPU Util | GPU Memory | W&B URL | Ray URL | Current Phase |
|---|---|---|---|---|---|---|---|---|---|---|---|

Summarize all 16 GPUs and both pods for `verl-n2-i2`. Extract steps and W&B links
from Ray logs. Mark startup phases such as model sync, checkpoint load, vLLM init,
CUDA graph capture, validation, checkpointing, or training.

### 4. Recover missing expected jobs

Before submitting, verify all of the following:

1. The expected config has no RUNNING or PENDING Ray submission on any tracked pool.
2. The assigned pool has no unrelated active job.
3. GPUs are idle and Ray reports enough available GPUs.
4. The config file exists locally.

Submit a single-node job:

```bash
bash submit_job.sh POOL recipe/phimm/config/ver_2607v1/CONFIG.yaml false true true
```

If a config inherits `trainer.nnodes=2` but is intentionally assigned to an `n1`
pool, submit directly with `trainer.nnodes=1`. Do not silently override topology for
`verl-n2-i2`.

Always sync code before submission. Never stop an unrelated evaluation or training
job to make room. Wait for it to finish or use another explicitly approved free pool.

### 5. Autofix failures

For a failed tracked job:

```bash
brix ssh POOL -- 'bash -l -c "ray job logs JOB_ID | tail -n 80"'
```

Diagnose the root cause, edit locally, run the smallest relevant validation, push with
`bpush POOL`, resubmit, and record the replacement Ray ID. Do not broad-catch errors,
hide failures, or repeatedly submit while stale Ray placement groups still reserve
GPUs. If hardware is idle but Ray has no GPUs, inspect `ray status`, live jobs, and
placement groups; wait for unrelated jobs rather than restarting or disrupting them.

### 6. Continue completed pipelines

When a tracked training job succeeds:

1. Find the selected `global_step_*` checkpoint.
2. Export it to HF safetensors using the exact `verl-asr-run` conversion procedure.
3. Invoke `eval-2607-benchmark-report`.
4. Preserve candidate/reference result paths and the consolidated workbook.

Do not stop monitoring merely because training ended; the pipeline completes only
after export, required benchmarks, and workbook validation.

## Reinstall the recurring monitor

Maintain exactly one five-minute schedule. Stop stale schedules before creating the
replacement. The schedule must:

- Monitor exactly the eight pools listed above.
- Begin every response with the complete workload table.
- Rediscover live and replacement Ray IDs.
- Show only newly observed metric rows after the table.
- Autofix tracked failures without disrupting unrelated jobs.
- Advance successful jobs through HF export and `eval-2607-benchmark-report`.
- Stop only after all tracked pipelines are complete.

## Safety rules

- Live Ray state is authoritative; initial IDs are provenance only.
- Never stop, clean up, or overwrite an unrelated job.
- Never submit duplicate copies of an expected config.
- Never assume low `nvidia-smi` utilization means Ray resources are free.
- Preserve local user changes and always push the current workspace before submission.
- Treat `verl-n2-i2` as a two-node, 16-GPU pool; inspect both pods.

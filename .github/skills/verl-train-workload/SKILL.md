---
name: verl-train-workload
description: 'Recover and coordinate the current verl 2607 training workload across all available verl-n1-i* Brix pools, delegating each job lifecycle to verl-asr-run. Use when: recover training workload, restore verl jobs, resume current training, rebuild the workload monitor, refresh the workload cache, or recover after Copilot/session restart.'
argument-hint: 'Optional action: status, recover, or reinstall-monitor'
---

# verl Train Workload

Recover and manage the current multi-node verl training workload after a session,
machine, schedule, or Ray-job interruption. Live Brix and Ray state is authoritative.

This skill owns workload-wide pool discovery, cache restoration, placement decisions,
and aggregate status reporting. It must invoke
the `verl-asr-run` skill for every
job's submission, monitoring, failure diagnosis, fix, resubmission, checkpoint export,
evaluation, report generation, and per-job recurring monitor. Do not reproduce those
procedures or execute their commands directly in this skill. Use `remote-development`
only for Brix node creation or resume when required by workload recovery.

## Training pools

Always discover every existing `verl-n1-i*` pool and include each one in the status
table, even when idle or unavailable. The skill contains no fixed config-to-pool
assignments. Any verified-free Ready `verl-n1-i*` pool is eligible for a cached job
that needs placement. Do not create or resume extra pools merely to expand capacity
when a Ready free pool exists.

## Recovery procedure

### 1. Discover pool state

```bash
brix pools 2>&1 | sed 's/\x1b\[[0-9;]*m//g' |
  grep -E '^verl-n1-i[^[:space:]]*[[:space:]]'
```

Discover all matching pools before deciding placement. If a cached node no longer
exists, select another verified-free `verl-n1-i*` pool rather than creating a
replacement. Resume a Paused or Suspended pool only when the cache identifies it as
hosting a tracked workload that must be recovered; then poll `brix ls POOL` until
Ready. Do not recreate or replace a pool that is Assigning or Scheduled.

### 2. Restore the node/job cache

Before querying remote nodes, read `recipe/phimm/config/verl_job.txt` when it exists.
Use its node, Ray job ID, config, progress/phase, and `Reports:` entries as recovery
hints so work can resume after a session restart. Expand abbreviated node names such
as `i14` to `verl-n1-i14`.

The cache defines the complete recovery set. Recover, submit, monitor, or continue
only jobs and report pipelines represented in this file. If the cache is absent or has
no valid job entries, report that there is no workload to recover and do not submit
anything. A live job not represented in the cache is unrelated: show it as occupying
its pool, but never adopt, stop, restart, resubmit, or add it to the recovery set.

The cache is never authoritative. Verify every cached node/job pair against current
Brix and Ray state before acting. A missing, malformed, stale, or contradictory row
must not reserve a node, trigger a submission, or override a live Ray job. Preserve
still-valid report queue/evaluating/reported state while reconciling it with live
export and evaluation processes.

### 3. Rediscover job placement

For every Ready pool, perform only the minimum Ray inspection needed to identify jobs
and determine whether the pool is occupied:

```bash
brix ssh POOL -- 'bash -l -c "ray job list"'
brix ssh POOL -- 'bash -l -c "ray status"'
```

Match jobs by config name and experiment name, not only by Ray ID. Record replacement
IDs created by previous recovery attempts. A pool is free only when it has no running
Ray submission and GPUs are idle. This inspection establishes placement and capacity;
delegate all log parsing, progress tracking, metrics, phase detection, and continued
polling for each discovered job to `verl-asr-run`.

### 4. Report the reconciled workload

After discovery and every delegated `verl-asr-run` transition, report the current
workload-wide node/job state to the user. Track replacement IDs, progress, phases,
and report state in conversation context and the recurring monitor prompt.
Do not create or update `recipe/phimm/config/verl_job.txt`; an existing file is
read-only recovery input.

### 5. Delegate missing cached jobs

Before submitting, verify all of the following:

1. The cached config has no RUNNING or PENDING Ray submission on any discovered pool.
2. The candidate `verl-n1-i*` pool has no unrelated active job.
3. GPUs are idle and Ray reports enough available GPUs on the candidate pool.
4. The cached config file exists locally.

After these checks pass, invoke `verl-asr-run` with the selected pool and config and
instruct it to submit and monitor the single-node job through the full pipeline. That
skill owns code sync, `trainer.nnodes=1` overrides when needed, command execution, Ray
job ID capture, and status reporting. Never stop an unrelated evaluation or training job
to make room. Wait for it to finish or choose another verified-free `verl-n1-i*` pool.

### 6. Delegate failures and resubmissions

For each failed tracked job, invoke `verl-asr-run` with its current pool, config, and
Ray job ID and instruct it to diagnose, fix, validate, sync, resubmit, record the
replacement ID, and continue monitoring. Never perform a direct resubmission from
this skill. If the delegated result reports blocked Ray resources, preserve the state
in the monitor context and wait rather than restarting or disrupting unrelated jobs.

### 7. Delegate completed pipelines

When training succeeds, keep that pipeline delegated to `verl-asr-run` until it has
exported all required checkpoints, completed the benchmark/report workflow, validated
the consolidated workbook, and reported the final state. Do not invoke checkpoint conversion
or benchmark execution directly from this skill.

## Reinstall the recurring monitor

Maintain exactly one 30-minute schedule. Stop stale schedules before creating the
replacement. The schedule must:

- Restore `recipe/phimm/config/verl_job.txt`, then rediscover and monitor every
  existing `verl-n1-i*` pool while delegating only cached jobs to `verl-asr-run`.
- Rediscover live and replacement Ray IDs.
- Report the aggregate node/job table, progress, phases, metrics, and report state
  in responses, and carry current pipeline identities in the monitor prompt.
- Invoke `verl-asr-run` to monitor each active pipeline and to handle every submission,
  failure, fix, resubmission, export, evaluation, and report transition.
- Reconcile delegated results into the aggregate status and monitor context without disrupting
  unrelated jobs.
- Stop only after all tracked pipelines are complete.

## Safety rules

- Live Ray state is authoritative.
- The cache is authoritative only for membership in the recovery set.
- Never stop, clean up, or overwrite an unrelated job.
- Never submit duplicate copies of a cached config.
- Never submit, monitor, fix, or resubmit a job directly; invoke `verl-asr-run`.
- Use only verified-free `verl-n1-i*` pools for fallback single-node placement.
- Never assume low `nvidia-smi` utilization means Ray resources are free.
- Preserve local user changes and let `verl-asr-run` sync the workspace before a
  submission or resubmission.
- Treat `recipe/phimm/config/verl_job.txt` as a cache only; live Brix and Ray state
  always wins.

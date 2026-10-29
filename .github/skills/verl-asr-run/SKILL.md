---
name: verl-asr-run
description: 'Execute one verl ASR Ray job or recurrently monitor, minimally recover when explicitly authorized, and report an explicitly scoped set of verl ASR experiments on authorized Brix pools: training, standalone evaluation, generation, or an explicitly requested checkpoint export. Use run-openasr-exp for a multi-stage train/checkpoint/OpenASR/report pipeline, and openasr-report for workbook-only requests.'
---

# verl ASR Run

Own one training, evaluation, or generation Ray job—and an explicitly requested
export—from preflight to a verified terminal state, or monitor a user-identified
set of experiments and recover them only when explicitly authorized. Use `run-openasr-exp` for a
train/export/evaluate/report pipeline or checkpoint queue, `openasr-report` for
workbook-only work, and `report-experiment` for requested run comparisons.
When editing this skill, do not operate remote work.

## Identity and authorization

Resolve the exact composed config, job type, pool/cluster/namespace, GPU-bearing
Ray nodes and topology, experiment/project/output names, code revision, launch
overrides, Ray submission ID, and W&B identity. Never substitute a similar
config, experiment, checkpoint, pool, or old submission. Preserve recipe
semantics, priority, topology, save cadence, resume policy, and model path
unless the user authorizes a change.

## Preflight

The usual context is cluster `prod-westus2-cw-6`, namespace `boren`, and remote
repo `/root/code/verl`; verify it live. Inspect the requested pool and every GPU
pod: Ray jobs/status, reservations, processes, and each GPU. Missing occupancy
evidence means unavailable, not free. Idle GPUs do not prove availability when
startup, validation, checkpointing, recovery, reservations, or planned work may
exist.

Respect an explicit pool. If occupied, queue the request unless another pool or
replacement of one exact job is authorized. Never pause, resume, stop, or clean
unrelated work. Resume an authorized paused pool only after confirming no
conflict, Brix readiness, and healthy Ray membership.

Useful checks:

```bash
brix pools
brix ssh -c prod-westus2-cw-6 -n boren <pool> -- \
  'bash -l -c "python /root/code/verl/ray_job.py list; ray status"'
brix ssh -c prod-westus2-cw-6 -n boren <pod> -- \
  'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
```

## Resolve config and artifacts

Compose Hydra before launch and capture material values. Count participating
GPU Ray nodes, not pool names. Verify `trainer.nnodes` and
`trainer.n_gpus_per_node` against live capacity; report rather than alter a
mismatch. For training, record the base model, LoRA rank/alpha, data mix,
algorithm, steps/epochs, validation/save cadence, resume mode, and local/blob
roots.

For standalone evaluation:

- require the exact requested checkpoint/HF export and verify completeness,
  provenance, files, and loader compatibility;
- treat source-training lifecycle as provenance, not a readiness gate;
- leave source training untouched when evaluation uses another authorized pool;
- pass the verified node count explicitly; use a unique experiment name; and
  disable training resume/new LoRA adapters for a merged model.

## Submit

Sync the exact local revision before submission. Prefer `quick_run.sh` when
Hydra overrides are needed because it prepares the Ray environment and forwards
the overrides.

```bash
bpush <pool>
brix ssh -c prod-westus2-cw-6 -n boren <pool> -- \
  'bash -l /root/code/verl/quick_run.sh <config.yaml> <hydra-overrides...>'
```

Capture the submission ID, then verify its live entrypoint, config, experiment,
resources, logs, and GPU activity; submission output alone is insufficient.
Never perform broad cleanup. Stop or replace only an explicitly authorized job
after matching its exact submission to the pool/config, then wait for terminal
state and resource release.

## Monitor

Use live Ray state as the job lifecycle authority. Current pod/process/GPU and
checkpoint evidence outrank helpers, W&B, old logs, and previous reports.

Each poll must rediscover the submission and Ray status; identify the actual
phase; inspect only new logs; check every participating pod/GPU; and validate
new checkpoints from complete rank shards and finished writes. Update durable
identity after an authorized replacement.

Count only completed optimizer steps for the current submission. Do not count
initialization, validation, generation-only activity, saves, old submissions,
or replayed log rows as new optimizer progress.

Report new training observations at their real step: available core metrics
(reward/score, loss, KL, entropy, gradient norm, learning rate), separate
per-dataset `val-core/*/p_err/mean@1` and `val-aux/*/p_err/mean@1`, and
checkpoint completeness/upload state. Finite metrics alone do not establish
stability; evidence any non-finite values, divergence, starvation, stalls, or
suspicious evaluation results.

### Recurring monitoring and authorized recovery

When the user requests periodic monitoring, use a chat-attached heartbeat so
the run retains this conversation's scope and evidence. Default to every 30
minutes when no cadence is given, and report on every scheduled pass, including
when a tracked job has no material change. Define the tracked set when creating
the heartbeat: include only jobs the user explicitly identified or asked this
invocation to launch and monitor. Save each tracked job's exact
experiment/config name, submission ID when known, pool, cluster, namespace,
lineage, and requested end condition in the heartbeat prompt. Live discovery
may relocate those tracked submissions but must never add other discovered or
user-owned jobs to the report. Record whether monitoring is read-only or may
recover failures and the recovery stopping condition. Before creating a
heartbeat, inspect existing schedules and job ownership so two automations do
not monitor or recover the same submission independently.

Each run is read-only unless the user separately authorized a specific recovery
action. Rediscover live Ray submissions rather than trusting an ID or state
saved by an earlier run, while preserving lineage when a submission was
replaced. Apply the normal monitoring evidence hierarchy and give one concise
row or section for every tracked job on every pass:

- experiment/config identity, live state and phase, pool, submission, and
  participating pods;
- completed optimizer step or evaluation progress, target, and delta since the
  prior report; explicitly say `no progress change` when unchanged;
- latest per-dataset validation and checkpoint state, distinguishing newly
  observed evidence from repeated values;
- current GPU/process health, abnormal metrics, changed risks, and any user
  action required.

Do not omit an unchanged tracked job or substitute a newly discovered job.
Highlight meaningful progress, phase/state changes, new validation or
checkpoints, failure, a stall supported by repeated observations, completion,
or required user input. Do not infer a stall from one quiet interval or infer
completion from an absent job. Stop the heartbeat after all tracked experiments
reach verified terminal states and the final status has been reported, or when
the user asks monitoring to stop. An explicit stop always overrides recovery
or continued monitoring.

When the user explicitly requests a scheduled monitor/fix/report loop, the
heartbeat may diagnose and recover the scoped job under `Recover` below. A
recovery pass must:

1. Establish a real failure from live Ray state and the complete current
   traceback, or establish a stall from repeated polls plus current
   process/GPU evidence. One quiet interval is not a stall.
2. Make only the smallest code or runtime-environment fix that preserves the
   effective recipe, data, reward, priority, pool, topology, checkpoint, and
   resume semantics. Never paper over a failure by silently changing them.
3. Verify the focused fix, sync the exact revision, confirm no competing live
   submission exists, and resubmit with a new submission ID while preserving
   the original failure lineage.
4. Attempt at most one repair/resubmission for a distinct root cause. If the
   same failure recurs, the required fix is outside this scope, ownership is
   ambiguous, or verification is unavailable, report the evidence and stop
   automated recovery pending user direction.

Continue monitoring a successfully recovered submission on the same cadence.
Keep it in the original tracked lineage and report it on every pass. State the
failure, fix, verification, old and new submission IDs, checkpoint/resume
boundary, and whether progress after recovery is genuinely new.

## Recover

Read the complete traceback and identify whether the failure is code, data,
environment, capacity, or recipe semantics. Apply and verify only a minimal
code/runtime fix within the requested job. Changes to data, algorithm, reward,
priority, pool, topology, checkpoint, or launch semantics require authorization.

Before resubmission, preserve the original lineage, record the failure and fix,
sync the new revision, and obtain a new submission ID. Never present the new
job's progress as continuation unless checkpoint resume evidence proves it.

## Complete

A training job is complete only when Ray reports `SUCCEEDED`, the intended
training target is satisfied, requested final validation is present, and any
required checkpoint has complete shards/uploads. A saved directory can exist
before it is complete. State separately the last completed optimizer step,
last validation step, and last complete checkpoint step.

Evaluation requires Ray `SUCCEEDED` plus final metrics and detailed artifacts
for every requested dataset; return exact paths. Do not build an OpenASR
workbook unless the report/pipeline workflow was also requested.

For an explicitly requested HF export, derive LoRA rank and alpha from the
effective producer config/checkpoint, never another run. Verify complete shards,
conversion, key normalization, tokenizer/config/custom-code assets, readability,
and upload before reporting the export.

## Reporting

Lead with experiment/config. For recurring updates, include one concise row or
section for every tracked job, its progress delta (including no change), the
latest evidence, and changed risks. Never include an untracked job merely
because live discovery found it:

| Field | Required evidence |
| --- | --- |
| State and phase | live Ray plus current logs |
| Pool/topology | exact pool and participating GPU pods |
| Progress | completed optimizer step or evaluation dataset |
| Resources | current per-pod GPU evidence |
| Validation | per-dataset values and real steps |
| Checkpoint | last complete step and shard/upload status |
| Links | verified W&B/Ray |

## Dependencies

- Use `remote-development` for generic Brix connection, code sync, and remote
  environment setup when needed.
- Use `report-experiment` only when the user requests a run comparison or
  experiment report.
- When invoked by `run-openasr-exp`, execute only its handed-off job; leave
  checkpoint selection, queue order, reporting, and shared monitoring to it.

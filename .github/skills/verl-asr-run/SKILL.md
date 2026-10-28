---
name: verl-asr-run
description: 'Execute and monitor one verl ASR Ray job on an authorized Brix pool: training, standalone evaluation, generation, or an explicitly requested checkpoint export. Use for a single job lifecycle. Use run-openasr-exp for a multi-stage train/checkpoint/OpenASR/report pipeline, and openasr-report for workbook-only requests.'
---

# verl ASR Run

Own one training, evaluation, or generation Ray job—and an explicitly requested
export—from preflight to a verified terminal state. Use `run-openasr-exp` for a
train/export/evaluate/report pipeline or checkpoint queue, `openasr-report` for
workbook-only work, and `report-experiment` for requested run comparisons.
When editing this skill, do not operate or schedule remote work.

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

Lead with experiment/config. For recurring updates, include only new rows, a
concise trend, and changed risks:

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

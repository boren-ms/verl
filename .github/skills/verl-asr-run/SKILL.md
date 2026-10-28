---
name: verl-asr-run
description: 'Execute and monitor one verl ASR Ray job on an authorized Brix pool: training, standalone evaluation, generation, or an explicitly requested checkpoint export. Use for a single job lifecycle. Use run-openasr-exp for a multi-stage train/checkpoint/OpenASR/report pipeline, and openasr-report for workbook-only requests.'
---

# verl ASR Run

Own one remote job from preflight through a verified terminal state. Keep job
execution separate from pipeline orchestration and report construction:

- **This skill:** one training, evaluation, or generation job; optional export.
- **`run-openasr-exp`:** a train/export/evaluate/report pipeline or checkpoint
  evaluation queue.
- **`openasr-report`:** workbook generation from completed evidence.
- **`report-experiment`:** run-level comparison reporting when explicitly requested.

When asked only to edit this skill, do not launch, stop, resume, or schedule
remote work.

## Inputs and job identity

Resolve and record:

- config path and composed config;
- Brix pool, cluster, namespace, participating Ray GPU nodes, and topology;
- experiment/project names and output root;
- Ray submission ID, W&B run ID/URL, code revision, and launch overrides;
- job type: `remax_*`, `grpo_*`/other training, `eval_*`, `long_eval_*`,
  or `gen_*`.

Do not silently substitute a similarly named config, experiment, checkpoint,
pool, or old submission. Preserve the user's recipe semantics, priority,
topology, save cadence, resume policy, and model path unless a change is
explicitly authorized.

## 1. Preflight the exact pool

The usual context is cluster `prod-westus2-cw-6`, namespace `boren`, with
pod-local repository `/root/code/verl`. Verify current identities rather than
relying on a prior report.

1. Inspect the requested pool and every GPU-bearing pod.
2. Check Ray jobs, placement/resource reservations, processes, and every GPU.
   A pool is free only when all of those signals are idle and there is no
   startup, validation, checkpoint, recovery, reservation, or planned launch.
3. Respect an explicit pool. If it is occupied, keep the request queued unless
   the user authorized another pool or replacement of one exact job.
4. Do not pause, resume, stop, or clean unrelated pools or jobs. Momentarily
   idle GPUs alone are not a cleanup signal.

Useful checks:

```bash
brix pools
brix ssh -c prod-westus2-cw-6 -n boren <pool> -- \
  'bash -l -c "python /root/code/verl/ray_job.py list; ray status"'
brix ssh -c prod-westus2-cw-6 -n boren <pod> -- \
  'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
```

Missing occupancy evidence means unavailable, not free. Resume a paused pool
only when it is the authorized target and has no conflicting work; wait for
both Brix readiness and healthy Ray membership.

## 2. Resolve config, topology, and artifacts

Compose the actual Hydra config and capture material resolved values before
launch. Count participating GPU-bearing Ray nodes, not pool names. Verify
`trainer.nnodes` and `trainer.n_gpus_per_node` against live capacity; report a
mismatch instead of silently changing either value.

For training, record at least the base model, LoRA rank/alpha, data mix,
algorithm, total steps/epochs, validation/save frequency, resume mode, and
local/blob output roots.

For standalone evaluation:

- require the exact checkpoint or HF export requested by the user;
- verify files, completeness, provenance, and loader compatibility;
- treat source-training lifecycle as provenance, not a readiness gate;
- leave source training untouched when evaluation uses another authorized pool;
- pass the verified evaluation node count explicitly;
- use a unique experiment name and disable training resume/new LoRA adapters
  for an already merged model.

For `long_eval_*`, remember that the model loads through `model.path` and final
evidence includes `result_details.jsonl` and `measures.json`. For `eval_*`, use
the recipe's validation-only behavior and collect every configured dataset, not
only the first metric line.

## 3. Sync and submit one job

Sync the exact local revision before submission. Prefer `quick_run.sh` when
Hydra overrides are needed because it prepares the Ray environment and forwards
the overrides.

```bash
bpush <pool>
brix ssh -c prod-westus2-cw-6 -n boren <pool> -- \
  'bash -l /root/code/verl/quick_run.sh <config.yaml> <hydra-overrides...>'
```

Capture the Ray submission ID from the command output, then verify the live
entrypoint, config, experiment name, resources, logs, and GPU activity.
Submission output alone does not prove that the job started.

Never run broad cleanup by default. Stop or replace a job only when explicitly
authorized, after proving the exact submission ID belongs to the intended
pool/config. Wait for its terminal state and resource release before launching
the replacement.

## 4. Monitor authoritative state

Use live Ray state as the job lifecycle authority. Current pod/process/GPU and
checkpoint evidence outrank helpers, W&B, old logs, and previous reports.

On each poll:

1. rediscover the current submission ID and Ray status;
2. identify the actual phase: environment setup, model load, rollout,
   optimization, validation, checkpoint save/upload, evaluation dataset, or
   shutdown;
3. inspect recent logs without replaying old rows as new progress;
4. check every participating pod and GPU;
5. verify new checkpoints from complete rank shards and finished writes;
6. update the durable identity if an authorized recovery created a replacement.

Count only completed optimizer steps for the current submission. Do not count
initialization, validation, generation-only activity, saves, old submissions,
or replayed log rows as new optimizer progress.

For training, present new observations with their real step:

- core optimization metrics such as reward/score, loss, KL, entropy, gradient
  norm, and learning rate when present;
- per-dataset `val-core/*/p_err/mean@1` and
  `val-aux/*/p_err/mean@1`, keeping each dataset separate;
- checkpoint step, completeness, and upload state.

Finite metrics do not establish stability. Flag non-finite values, divergence,
resource starvation, stalled phases, or suspicious evaluation values with the
supporting evidence.

## 5. Failure and recovery

Read the complete traceback and identify whether the failure is code, data,
environment, capacity, or recipe semantics. Apply and verify a minimal code or
runtime fix when it stays within the requested job. Do not change data,
algorithm, reward, priority, pool, topology, checkpoint, or launch semantics
without authorization.

Before resubmission, preserve the original lineage, record the failure and fix,
sync the new revision, and obtain a new submission ID. Never present the new
job's progress as continuation of the failed submission unless checkpoint
resume evidence proves it.

## 6. Completion

A training job is complete only when Ray reports `SUCCEEDED`, the intended
training target is satisfied, requested final validation is present, and any
required checkpoint has complete shards/uploads. A saved directory can exist
before it is complete. State separately the last completed optimizer step,
last validation step, and last complete checkpoint step.

An evaluation is complete only when Ray reports `SUCCEEDED` and every requested
dataset has final metrics and detailed artifacts. Return the exact output paths.
Do not build an OpenASR workbook unless the request also invokes the report or
pipeline workflow.

For an explicitly requested HF export, derive LoRA rank and alpha from the
effective producer config/checkpoint. Never use fixed rank/alpha values copied
from another run. Verify complete shards, conversion output, key normalization,
required tokenizer/config/custom-code assets, readability, and the uploaded
artifact before reporting the export path.

## 7. Long-running monitoring

When the requested job will outlive the current turn and the product exposes a
native scheduling tool, create or update one chat-attached monitor for this
job. Save a durable prompt containing the pool, config, current submission ID,
state path, checks above, recovery boundary, completion condition, and stop
condition. Keep it quiet while state is unchanged unless the user requested
every-poll updates. Notify on meaningful progress, completion, failure, or
required user action. Replace stale monitor identity after recovery and stop it
at the terminal condition.

Do not emit raw scheduling directives or edit automation storage files. If the
native scheduling capability is unavailable, continue synchronous monitoring
as far as practical and report that limitation accurately.

## Status format

Lead with experiment/config, then show:

| Field | Required evidence |
| --- | --- |
| State and phase | live Ray plus current logs |
| Pool/topology | exact pool and participating GPU pods |
| Progress | completed optimizer step or evaluation dataset |
| Resources | current per-pod GPU evidence |
| Validation | per-dataset values and real steps |
| Checkpoint | last complete step and shard/upload status |
| Links | W&B/Ray when verified |

For recurring updates, include only new rows plus a concise trend and any
changed risk. Never claim completion because a monitor was installed.

## Active dependencies

- Use `remote-development` for generic Brix connection, code sync, and remote
  environment setup when needed.
- Use `report-experiment` only when the user requests a run comparison or
  experiment report.
- When invoked by `run-openasr-exp`, execute only the single handed-off job and
  leave checkpoint selection, queue ordering, report building, and the shared
  pipeline monitor to that orchestrator.

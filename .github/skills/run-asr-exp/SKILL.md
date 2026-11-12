---
name: run-asr-exp
description: Run, monitor, and recover scoped verl ASR jobs on Brix. Training defaults to final-checkpoint export, full OpenASR evaluation and an Excel comparison unless explicitly disabled, with 30-minute monitoring and validation-triggered reports.
---

# Run ASR Exp

Own the requested ASR job or pipeline: training, standalone evaluation,
generation, checkpoint export, or an OpenASR train/export/evaluate/report
workflow or checkpoint queue. For training requests, include final-checkpoint
export, full OpenASR evaluation and an Excel comparison by default unless the
user explicitly disables them (including a training-only request). State this
pipeline at launch. This default does not expand status-only, generation-only,
export-only or standalone evaluation requests. During
authorized execution, monitor tracked work every 30 minutes and invoke
[report-experiment](../report-experiment/SKILL.md) when new or corrected
completed validation is ready, rather than on every monitor pass. Honor an
explicit different monitoring cadence, report opt-out or status-only boundary.
Use only identified comparison runs; never invent a baseline.
For training with the default final evaluation, a multi-stage workflow,
checkpoint queue, export, or audio
prefetch, read [references/openasr-pipeline.md](references/openasr-pipeline.md).
For training, periodic monitoring or scoped recovery, read
[references/monitor-recover.md](references/monitor-recover.md).

Editing this skill alone never authorizes remote operations. A status request is
read-only: do not resume or pause pools, submit, stop, schedule, or advance work.

## Identify the work

Resolve the exact composed config, job type, cluster/namespace, resource
restrictions, GPU-bearing Ray nodes and topology, experiment/project/output
names, code revision, launch overrides, submission ID and W&B identity. Never
substitute a similar config, experiment, checkpoint, pool, or old submission.
Preserve recipe, data, reward, priority, topology, save cadence, resume policy,
and model path unless a change is authorized. Explicit steps and paths win.

Compose Hydra before submission and capture material effective values. For
training include base model, LoRA rank/alpha, data mix, algorithm, steps/epochs,
validation/save cadence, resume mode and local/blob roots. Count participating
GPU Ray nodes rather than pool names; compare `trainer.nnodes` and
`trainer.n_gpus_per_node` with live capacity and report mismatches rather than
silently changing the recipe. For evaluation verify the exact checkpoint or HF
export, provenance, completeness and loader compatibility. Source training
state is provenance, not an evaluation readiness gate.

## Training target and continuity

Before launch, resolve and persist the desired final optimizer step `T` from
the user's explicit target, or the effective trainer config when none was given.
Record whether it is an absolute final step or an explicitly requested number
of additional updates; for additional updates, establish the verified starting
step before deriving `T`. Keep this target fixed across submissions and recovery.

Inspect the actual trainer's stopping logic and runtime dataloader length. In
the current PPO trainer, `trainer.total_training_steps` overrides the derived
step target, but the outer loop is still bounded by `trainer.total_epochs`.
Verify the available batches/epochs, restored dataloader position and all exit
flags can reach `T`; a training request must not run with `val_only` enabled.
Resolve conflicting limits before launch. Align them with an explicit user step
target only within the authorized recipe; surface any required semantic change
rather than silently lowering the target, changing data, or extending an
explicit epoch limit. Confirm the runtime prints the intended target.

Own training until the verified target is reached, including monitoring and
minimal recovery within the authorized boundary. A successful submission,
intermediate checkpoint, completed evaluation/report, or Ray `SUCCEEDED`
below `T` does not fulfill the training request. Keep unfinished training in the
tracked set and diagnose early exits under the recovery reference. Continue in
the current chat with one 30-minute monitor that tracks progress, applies
authorized minimal fixes, and invokes `report-experiment` when new or corrected
completed validation is ready. Keep monitoring and recovery
active if reporting fails. Honor an explicit different cadence or read-only
boundary. Follow the monitoring reference for persistence, report refreshes,
and stop conditions.
If the user stops work or recovery requires a decision outside scope, report
training as incomplete with the last verified step, remaining steps and blocker.

## Discover and prepare capacity

For an authorized execution, automatically discover Brix pools and compatible
nodes; the user need not name one. The usual cluster is `prod-westus2-cw-6`,
namespace `boren`, and remote repo `/root/code/verl`; verify live and respect
any explicit cluster, namespace, pool, priority, quota or resource restriction.
Within that scope, choose verified available capacity matching the effective
topology; prefer `team-critical` when no pool or priority was specified and
that choice does not change priority. Tell the user the selected pool.

Inspect all participating GPU pods, Ray status and jobs, reservations, processes,
each GPU, ownership and planned work. Missing or contradictory occupancy
evidence means unavailable. Idle GPUs do not prove availability during startup,
validation, checkpointing, recovery or reservations. Never disturb unrelated
work. An explicitly requested occupied pool stays queued unless an alternative
or replacement of the exact job is authorized.

An instruction to run the experiment authorizes resuming a suitable paused pool
within the permitted resource scope when needed. First verify ownership, absence
of conflicting work or plans, and that resuming is allowed; then resume the
selected pool and verify Brix readiness, Ray membership, all GPU pods and
capacity before submission. If ownership, permissions, availability or intended
use cannot be verified, do not resume; find another eligible pool or report the
blocker. Do not pause or resume unrelated pools. Respect read-only or narrower
user instructions.

Typical checks, adapted to the actual live environment:

```bash
brix pools -a
brix ssh -c prod-westus2-cw-6 -n boren <pool> -- \
  'bash -l -c "python /root/code/verl/ray_job.py list; ray status"'
brix ssh -c prod-westus2-cw-6 -n boren <pod> -- \
  'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
```

## Submit and verify

Sync the exact local revision to the selected pool. Prefer `quick_run.sh` for
Hydra overrides because it prepares the Ray environment and forwards them:

```bash
bpush <pool>
brix ssh -c prod-westus2-cw-6 -n boren <pool> -- \
  'bash -l /root/code/verl/quick_run.sh <config.yaml> <hydra-overrides...>'
```

Capture the submission ID; verify its live entrypoint, config, experiment,
resources, logs and GPU activity. Submission output alone is insufficient.
Stop or replace a tracked job only for authorized scoped recovery or an
explicit request, after matching its exact submission, pool and config; wait for
terminal state and resource release. Never use broad cleanup.

For standalone evaluation, use a unique experiment name and explicit verified
node count; disable training resume and new LoRA adapters for a merged model.
Leave source training untouched when evaluation uses other authorized capacity.

## Observe and complete

Live Ray state is the lifecycle authority; current pod/process/GPU and checkpoint
evidence outrank helpers, W&B, old logs and reports. At each poll rediscover the
submission and Ray status, identify the phase, inspect new logs and each
participating pod/GPU, and verify newly observed checkpoint shards and completed
writes/uploads. Count only completed optimizer steps for the current submission;
initialization, validation, generation, saves, previous submissions and replayed
rows are not new optimizer progress.

Report core metrics when available (reward/score, loss, KL, entropy, gradient
norm, learning rate), per-dataset `val-core/*/p_err/mean@1` and
`val-aux/*/p_err/mean@1` at their real steps, checkpoint state, resource health,
and non-finite values or other observed risks. Finite metrics alone do not
prove stability; one quiet interval does not prove a stall.

Training completes only when Ray reports `SUCCEEDED`, completed optimizer-step
evidence reaches the persisted target `T`, requested final validation is
satisfied, and required checkpoints have
complete shards/uploads. State last completed optimizer, validation and complete
checkpoint steps separately. Evaluation completes only with Ray `SUCCEEDED`,
final metrics and detail artifacts for every requested dataset; give exact
paths. A requested export requires producer-derived rank/alpha, complete shards,
verified conversion/assets/readability and upload. Use `report-experiment` for
report packages. Report exact identities, pool/topology, relevant links,
progress, validation, checkpoint state and material limitations.

## Pause completed resources

For an authorized execution, pause the node or Brix pool used by this job as
the default final resource action, unless the user explicitly requests keeping
it running. This does not authorize pausing resources during an instruction-only
edit or read-only monitoring. Record the exact cluster, namespace, pool and
participating nodes so completion cleanup cannot target a similarly named pool.

Pause only after the completion checks above pass and all requested work still
needing those resources is finished: training target and final validation,
checkpoint writes/uploads, export, evaluation and durable result collection as
applicable. Preserve required logs, metrics and artifacts outside ephemeral
node storage. Finish or stop only this pipeline's exact optional prefetch
process. An intermediate report, idle GPUs, failed job or early `SUCCEEDED`
below the target never triggers completion pausing.

Immediately before pausing, rediscover the pool and check every GPU pod, Ray
jobs, processes, reservations, ownership and planned/queued work. Verify the
pause operation's scope: if it pauses an entire pool, every affected node must
be eligible. Never pause source training for an evaluation-only request,
unrelated work, shared active capacity, or a pool needed by the next requested
stage or queued job. Missing or contradictory evidence defers the pause;
record the reason and keep tracking without stopping other work.

Use the supported Brix pause operation after checking its current CLI help;
do not guess a node-versus-pool command or use deletion as a substitute. Verify
the control plane reports the exact resource paused; command acceptance or SSH
failure alone is insufficient. Persist completion evidence, pause time/status
and any blocker or explicit keep-running/shared-use exception. If pausing fails,
retain a pending pause and retry with fresh checks on the next monitoring pass
or continue checking in this chat when no schedule exists. Report job success
and resource pause separately; do not claim the resource paused until verified.

Use `remote-development` for generic Brix setup when needed.

## Default final-checkpoint evaluation

After training reaches its persisted target `T` and final checkpoint writes and
uploads complete, export that exact final checkpoint and evaluate the full
OpenEN, OpenML, OpenLong and OpenExt (Chinese/Vietnamese) suites. Follow
[references/openasr-pipeline.md](references/openasr-pipeline.md), preserving
explicit dataset, resource and reference choices. Never substitute an earlier
checkpoint for the final one or change training to obtain evaluation evidence.

Use [openasr-report](../openasr-report/SKILL.md) to create the canonical Excel
comparison, including WER/CER, matched-coverage WERR, model provenance and exact
dataset result paths. Include the verified full Excel table figure in the final
report PDF and embed it inline in the final chat response beside the `.xlsx`
link, together with the verified validation trajectory figures from
`report-experiment`. Show both kinds of figures directly in chat, rather than
only linking their files. Split long/wide tables into readable images/pages while
retaining every canonical dataset and summary row; a workbook link or selected-row summary alone
does not satisfy final delivery. Use the
user's identified reference when supplied; otherwise use openasr-report's
documented supplied baseline and disclose its provenance limits. This supplied
baseline is not an invented comparison run. Reuse verified reference results;
record missing reference evidence rather than inventing scores or launching
unrequested reference training. Complete the report-experiment handoff and
resource pause checks only after all enabled stages finish. Persist explicit
opt-outs and the end condition in pipeline state.

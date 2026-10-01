---
name: verl-train-openasr-report
description: "Run a verl ASR training config on a specified Brix pool, optionally stop its previous job with explicit authorization, evaluate the last complete checkpoint with the requested eval config, and build an OpenASR Excel report. Use when: train then evaluate last checkpoint, replace a training job and report OpenASR, run eval_2609_openall_mix after training, or resume a train-to-OpenASR pipeline."
argument-hint: "<train.yaml> --node <pool> [--stop-previous | --stop-job <ray-id>] [--eval-config <eval.yaml>] [--out <report.xlsx>] [launch|resume|status]"
---

# Train, Evaluate Last Checkpoint, and Report OpenASR

Own one end-to-end pipeline:

**replace an explicitly authorized previous job -> train -> export the last
complete checkpoint -> run the requested evaluation -> OpenASR workbook**.

This skill is stored in the repository. Invoke `remote-development` for Brix
connectivity, `verl-asr-run` for submission and monitoring mechanics, and
`openasr-report` for the canonical report builder. Do the work directly;
subagents and a multi-agent workflow are not required.

## Contract and inputs

| Input | Behavior |
| --- | --- |
| Training config | Required existing YAML under `recipe/phimm/config/`; retain its experiment name. |
| Node | Required Brix pool; normalize `n4i0` or `n4-i0` to `verl-n4-i0`. |
| Stop previous | Disabled unless explicitly requested. `--stop-job` identifies one submission; `--stop-previous` requires one unambiguous previous active job. |
| Eval config | Default `recipe/phimm/config/v2609_asr/eval_2609_openall_mix.yaml`; honor any explicit replacement. |
| Checkpoint | Last complete checkpoint of this training run, by numeric step, not best validation score. |
| Report | Default `tmp/openasr_report/<train-stem>_step<step>.xlsx`. |
| Operation | `launch` starts the pipeline; `resume` reconciles existing work; `status` is read-only. |

Example:

```text
/verl-train-openasr-report recipe/phimm/config/v2609_asr/remax_2609v0a_earning_batches_tts_s1k_bs128_n4_r256_g32.yaml --node verl-n4-i0 --stop-previous --eval-config recipe/phimm/config/v2609_asr/eval_2609_openall_mix.yaml
```

When the user only requests creating or editing this skill, modify the skill
without stopping, submitting, or scheduling remote jobs.

### Differences from the general runner

Use `verl-asr-run` as a mechanics reference, not its default post-training
policy. In this pipeline:

- Select the **last complete** checkpoint, never the best validation checkpoint.
- Run the **specified eval YAML**, not the full in-house/reference benchmark
  suite. Do not invoke `eval-2609-benchmark-report` automatically.
- Keep training and evaluation on the requested pool. A Brix pool can contain
  multiple Ray GPU nodes: `verl-n4-i0` normally has four pods and 32 GPUs.
  Do not force `trainer.nnodes=1` merely because there is one pool name.
- Stop only the authorized previous Ray submission. Never call
  `ray_job.py cleanup`, including through `submit_job.sh`'s default cleanup.
- Do not pause, resume, stop jobs on, or otherwise manage unrelated pools.
- Derive LoRA rank and alpha from the effective training config/checkpoint;
  do not copy the general runner's fixed 320/640 export values.

## 1. Preflight and durable state

Create a todo list for preflight, replacement/submission, training, export,
evaluation, and report verification. Mark completion only from observed state.

Read the training and eval YAMLs and their defaults. Compose the training
config with the repository's Hydra resolvers; record the resolved model path,
LoRA rank/alpha, `trainer.nnodes`, GPUs per node, total steps/epochs,
`save_freq`, `resume_mode`, and local/blob output roots. Preserve save cadence
and training hyperparameters. Ensure epoch bounds permit the requested step
target. Surface conflicts rather than silently changing training semantics.

Use ignored `tmp/verl_train_openasr_report/<train-stem>/` for a `state.json`,
resolved config snapshots, submission output, full logs, metrics JSON, model
metadata, result-path mapping, and export provenance. Store:

- Normalized train/eval paths, pool, experiment names, overrides, code snapshot,
  output roots, checkpoint policy `last_complete`, and replacement authorization.
- Current phase, stopped job ID, training/export/eval job IDs and statuses,
  observed step/target, selected checkpoint path and step, W&B and Ray URLs.
- Report path, verified artifact paths, and current monitor identity.

Do not store credentials or signed URLs. Never treat files left by a previous
run with the same experiment name as proof this run succeeded. Reconcile live
jobs, run identity, logs and artifact provenance before reusing state.

Maintain `recipe/phimm/config/verl_job.txt` using the general runner's
timestamp/table/Reports format. Preserve unrelated entries, replace only this
pool's current row, and do not retain superseded job IDs. Use `n4-i0` rather
than ambiguous `i0` when multiple pool sizes share the index. Report phases as
training, exporting last checkpoint, evaluating the actual dataset, or
building report; adapt best-checkpoint wording to this last-checkpoint policy.
Do not update `recipe/phimm/config/ver_2607v1a/job.txt`.

## 2. Identify and replace the previous job

1. Inspect `brix pools <pool>`, `ray status`, Ray submissions and GPU activity
   on every participating GPU pod. Use `brix ssh` for all remote execution.
   Retrieve submission IDs, lifecycle status and entrypoints using
   `ray job list --format=json` or the Ray JobSubmissionClient API.
   `ray_job.py list` alone does not display submission IDs.
2. Inspect active schedules before replacement. Stop only a monitor belonging
   to the job being replaced; never remove unrelated monitoring.
3. Verify that `--stop-job` belongs to this pool and the intended previous job.
   For `--stop-previous`, identify the single previous active submission.
   Multiple plausible jobs require clarification; do not stop all of them.
   Without replacement authorization, an occupied pool is a blocker, not
   permission to evict jobs or choose another pool.
4. Stop by the exact observed submission ID:

   ```bash
   brix ssh <pool> -- 'bash -l -c "ray job stop <previous-submission-id>"'
   ```

5. Wait for a terminal status and release of resources across all GPU pods.
   Require no active/pending conflicting submission, no unexplained GPU
   activity, and no remaining placement-group GPU reservation. A pod is busy
   if any GPU utilization exceeds 5% or memory exceeds 5000 MiB. Do not use
   broad process-killing commands to make it idle.

If this pipeline is already running, monitor it rather than replacing it
again. If the pool must be resumed, wait for both Ready and successful
`ray status` before submission.

## 3. Submit and finish training

Sync the latest local code with `bpush <pool>` before submission. Verify the
effective config and checkpoint namespace on the remote pool. Preserve
`resume_mode` unless the user requested a fresh run; record any existing resume
checkpoint. A fresh rerun must not silently reuse stale step artifacts.

Use `quick_run.sh`, which prepares all Ray nodes and supplies the required
CUDA runtime environment, without invoking the unsafe cleanup wrapper:

```bash
brix ssh <pool> -- 'bash -l /root/code/verl/quick_run.sh <train.yaml>'
```

Capture the new Ray submission ID and immediately record it. Verify the
entrypoint, config, experiment name and requested topology; submission output
alone is not evidence training started. Inspect status, full logs, Ray
resources and GPU usage. Record effective `save_freq` in the first update.

Monitor startup, training steps, validation `p_err`/`p_edge`, checkpoint
saves and uploads. Accumulate training metrics and W&B/Ray links. Diagnose
failures from tracebacks, apply minimal local fixes, sync and resubmit under
the original training name. Record recovery overrides and replace stale IDs.
Do not copy unrelated prior jobs' runtime overrides without evidence.

Require Ray `SUCCEEDED`, the intended final training step, and finished
checkpoint writes before advancing. If epochs exhaust before the step target,
recover to the intended target rather than reporting full completion.
Do not launch checkpoint evaluation while training is running.

## 4. Resolve and export the last complete checkpoint

Enumerate this run's `global_step_<N>` directories numerically, using save
logs and local/blob listings. Verify all actor shards for one world size:
`model_world_size_<W>_rank_<R>.pt` for every rank `0..W-1`. Confirm uploads
are complete and belong to this run. Do not assume every pod contains every
shard, that a directory name proves completeness, or that an old HF export
with the same path is current.

Choose the largest complete step after successful training. Wait for the
final save if still uploading; never silently fall back past an incomplete
expected final save. Record the actual selected step (which need not equal
the target if the configured save behavior legitimately differs). Do not
substitute an earlier checkpoint because its validation metric is better.

Wait until the training pool is free. If an HF export verified against this
checkpoint already exists, reuse it. Otherwise use the converter at its
current repository location:

```bash
python3 plugins/qwen35_audio/src/hf_qwen35_audio/convert_verl_to_pt.py \
  --input <checkpoint-root> \
  --output <hf-dir>/model.pt \
  --match-lora-merged \
  --lora-alpha <effective-alpha> \
  --lora-rank <effective-rank>
```

Execute remotely through `brix ssh`; for an `az://` input, also pass
`--local-cache <dedicated-local-shard-cache>`. The converter checks complete
rank coverage. Confirm that the number of merged adapter pairs is sensible
and that alpha/rank matches training. The example config currently resolves
to rank 256 and alpha 512; recompute instead of relying on those numbers.

The merged converter output is a top-level tensor dictionary. Strip
`.base_layer.` from weight keys and save `model.safetensors`, checking for
key collisions and remaining adapter tensors. Copy config, tokenizer,
preprocessor and required custom model files from the actual training base,
not an unrelated baseline. Do not copy base-model weight files or a stale
sharded safetensors index into a single-file export.

Verify export readability, expected tensor keys/shapes and required model
assets before publishing to `<training-output>/global_step_<N>/qwen_hf/`.
Verify the uploaded files. Never reuse a cached failed export; invalidate
only specifically identified stale files, not broad cache directories.

## 5. Evaluate on the same pool

Compose the exact requested eval YAML and inventory its datasets. Set
`trainer.nnodes` explicitly to the number of participating healthy GPU Ray
nodes, not the number of Brix pool names or total Ray nodes (which may include
CPU-only nodes). Require the pool to be idle before launching.

Sync current code again. Use a candidate-specific experiment name, for example
`<eval-stem>__<train-stem>_step<N>`, so evaluation outputs cannot overwrite
another model's results. Pass the verified HF export as the model, disable
new LoRA adapters for the already-merged model, and prevent training resume:

```bash
brix ssh <pool> -- 'bash -l /root/code/verl/quick_run.sh <eval.yaml> \
  trainer.experiment_name=<candidate-eval-name> \
  trainer.nnodes=<gpu-node-count> \
  trainer.resume_mode=disable \
  actor_rollout_ref.model.path=<verified-hf-export> \
  actor_rollout_ref.model.lora_rank=0'
```

Check the composed config is validation-only and uses the intended candidate.
Record the exact evaluation output root, code snapshot, job ID and W&B URL.
If remote model caching would duplicate the export per worker, use a
verified identical shared pod-local model path on every participating pod;
keep the original exported model URI in provenance.

Monitor until Ray `SUCCEEDED` and all configured datasets have final metrics
and detailed artifacts. For `eval_2609_openall_mix.yaml`, this includes
MixLang, eight English datasets and seventeen ML datasets. Do not declare
success from one intermediate validation metric line.

Retain MixLang results even though the OpenASR workbook excludes them; report
their observed metrics separately without inserting extra canonical rows.

## 6. Build and verify the OpenASR report

Invoke `openasr-report` and use its supplied `2609v0` baseline, canonical
layout and builder. Do not substitute a new baseline evaluation or copy
candidate metrics from baseline values.

Extract only candidate `val-aux/<source>/p_err/mean@1` metrics as fractions,
not raw `wer` or percent-valued numbers. Map source IDs to canonical dataset
names only after verifying the composed dataset manifests:

| Verified source in the default eval config | Canonical report dataset |
| --- | --- |
| `ami` from the cleaned >=1s manifest | `ami_clean` |
| `gigaspeech` from the cleaned >=1s manifest | `gigaspeech_clean` |
| `ls_clean` | `librispeech-clean` |
| `ls_other` | `librispeech-other` |
| `earnings22_cleaned_aa_chunked` | `earnings22-cleaned-aa-chunked` |
| `voxpopuli_cleaned_aa` | `voxpopuli-cleaned-aa` |

Other canonical sources retain their names; use supported corpus-first ML
aliases where needed. The OpenASR builder does not automatically translate
these English source IDs. Write canonical `metrics.json` and
`dataset_results.json` explicitly; never globally alias an uncleaned
dataset to a cleaned one.

For every evaluated canonical dataset, discover and verify its exact detail
file, such as `<eval-output>/val_data_gen/<actual-source>/0.jsonl`. Keep the
actual source in the file path even when the report key is canonical. Do not
invent paths from naming conventions or supply a parent directory.

Write `model_info.json` with the exported model path, eval config, node,
evaluation Ray job ID, W&B URL and actual code snapshot. Leave unavailable
fields blank. Preserve training provenance in pipeline state.

Run the builder supplied by the loaded `openasr-report` skill:

```bash
/home/boren/.virtualenvs/openai/bin/python <openasr-report-dir>/scripts/build_openasr_xlsx.py \
  "<train-stem>@step<N>" \
  --metrics <artifact-dir>/metrics.json \
  --model-info <artifact-dir>/model_info.json \
  --dataset-results <artifact-dir>/dataset_results.json \
  --out <report.xlsx>
```

If that dependent skill or builder is unavailable, report the missing
dependency; do not claim workbook completion. Use `--extend-xlsx` only when
requested, retaining earlier columns and their metadata.

Reopen the workbook and verify:

- `openasr`, `model_info` and `dataset_results` exist with the correct model
  identity, checkpoint and exact supplied result paths.
- All expected canonical metrics are present. Missing expected results mean
  incomplete evaluation, not permission to fill from the baseline.
- Every positive-baseline dataset pair has `1 - model / baseline` as an Excel
  formula, including valid 100% WERR for a zero candidate. Missing pairs and
  zero baselines remain blank.
- Average WERRs are numeric and use matched dataset coverage, not averaged
  per-dataset WERRs. Displayed means use each column's available datasets.
- Positive and negative deltas, percent formatting, centered cells, row
  colors and zero-centered conditional formatting remain intact.
- The canonical sheet has no extra notes, comments, banners or MixLang rows.
  Existing metadata remains unchanged when extending a workbook.

## Monitoring, recovery and completion

After verifying a live submission, install one recurring five-minute monitor
using the scheduling tool. Its prompt must identify this skill, the durable
state path, node, train/eval configs and current Ray job ID, and explicitly
retain the **last complete checkpoint -> specified eval -> OpenASR report**
policy. Inspect existing schedules and replace only this pipeline's stale
monitor after resubmission or phase changes. Plain text claiming monitoring
is installed is not sufficient.

On each tick, rediscover current jobs and reconcile artifacts before taking
action; update durable state and the pool's status row. Resume from the
first incomplete phase. Never double-submit an export/evaluation, restart
training after it succeeded, or stop unrelated work encountered on recovery.
`status` only observes and reports; it does not schedule or advance work.

Keep monitoring through training, export, evaluation and workbook validation.
Stop the schedule only when all requested work is complete or the user
cancels it. If genuinely blocked, record the exact blocker and do not claim
success or retry destructively. Pause a pool resumed by this pipeline only
after it is idle and has no remaining scheduled work.

The final response includes the workbook path, actual selected last step and
checkpoint URI, English and ML mean `p_err` and WERR, separate MixLang results
when applicable, and training/evaluation IDs and links. Report a running
pipeline as running, not complete merely because a monitor was installed.

---
name: run-openasr-exp
description: "Use verl-asr-run to run a training config YAML on a specified Brix pool, start early background ZIP evaluation-audio precaching only on the main pod without blocking decoding, and evaluate its last complete checkpoint with recipe/phimm/config/v2609_asr/eval/eval_2609r2_openall_mix_zip.yaml, then use openasr-report to build the results workbook. Optionally replace an explicitly authorized previous job or honor an explicit eval config override. Use when: train then evaluate last checkpoint, replace a training job and report OpenASR, run eval_2609r2_openall_mix_zip after training, or resume a train-to-OpenASR pipeline."
argument-hint: "<train.yaml> --node <pool> [--stop-previous | --stop-job <ray-id>] [--eval-config <eval.yaml>] [--out <report.xlsx>] [launch|resume|status]"
---

# Run OpenASR Experiment

Own one end-to-end pipeline:

**`/verl-asr-run` training config YAML -> export the last complete checkpoint
-> `/verl-asr-run` with `recipe/phimm/config/v2609_asr/eval/eval_2609r2_openall_mix_zip.yaml`
-> `/openasr-report` results workbook**.

Start audio precaching only on the main pod as an early independent background task; it runs
alongside training/export and decoding, never as a gate in this pipeline.

This skill is stored in the repository and orchestrates the dependent skills.
Invoke `verl-asr-run` to execute and monitor both training and evaluation,
not merely as a mechanics reference. Invoke `remote-development` when needed
for Brix connectivity and `openasr-report` after evaluation for the canonical
report builder. Skill invocation does not require subagents; do the work
directly with the loaded skills and the policy below.

## Contract and inputs

| Input | Behavior |
| --- | --- |
| Training config | Required existing YAML under `recipe/phimm/config/`; retain its experiment name. |
| Node | Required Brix pool; normalize `n4i0` or `n4-i0` to `verl-n4-i0`. |
| Eval node count | Must equal the successful training run's effective `trainer.nnodes`, including training overrides; use the same pool. |
| Stop previous | Disabled unless explicitly requested. `--stop-job` identifies one submission; `--stop-previous` requires one unambiguous previous active job. |
| Eval config | Default `recipe/phimm/config/v2609_asr/eval/eval_2609r2_openall_mix_zip.yaml`; honor any explicit replacement. |
| Eval audio cache | Start `recipe.phimm.cache_eval_audio` early in the background only on the main pod to cache physical ZIP archives for the resolved suite; never wait for it before submitting or decoding evaluation. |
| Checkpoint | Last complete checkpoint of this training run, by numeric step, not best validation score. |
| Report | Default `tmp/openasr_report/<train-stem>_step<step>.xlsx`. |
| Operation | `launch` starts the pipeline; `resume` reconciles existing work; `status` is read-only. |

Example:

```text
/run-openasr-exp recipe/phimm/config/v2609_asr/remax_2609v0a_earning_s1k_bs128_n4_r256_g32.yaml --node verl-n4-i0 --stop-previous --eval-config recipe/phimm/config/v2609_asr/eval/eval_2609r2_openall_mix_zip.yaml
```

When the user only requests creating or editing this skill, modify the skill
without stopping, submitting, or scheduling remote jobs.

## Required skill handoffs

1. Invoke `/verl-asr-run` with the supplied training config YAML and requested
   pool. Request **training only** from that invocation; leave its automatic
   best-checkpoint/standard-benchmark post-training pipeline disabled.
   Pass the replacement authorization, if any, and the safety constraints
   below. Once training is verified running, start section 5's background audio
   precache workflow without waiting for training to finish. Continue monitoring
   training until verified completion.
2. Resolve and export this run's **last complete checkpoint** as described
   in section 4. This skill owns checkpoint selection; do not let the general
   runner replace it with a best-validation checkpoint.
3. Invoke `/verl-asr-run` again for a **standalone evaluation** using
   `recipe/phimm/config/v2609_asr/eval/eval_2609r2_openall_mix_zip.yaml` unless the user
   explicitly supplied another eval YAML. Keep section 5's early background
   audio precaching running alongside decoding; do not wait for it. Supply the verified last-checkpoint
   HF export, same pool and node count as training, unique evaluation
   experiment name, and section 5's overrides. Wait for all configured
   evaluation datasets to finish.
4. Invoke `/openasr-report` with the candidate label, canonical metrics,
   model metadata, exact dataset result files, and workbook output path from
   section 6. Verify the workbook before completing the pipeline.

Example handoff requests (natural-language skill inputs, not shell commands):

```text
/verl-asr-run Run <train.yaml> on <pool>; training only, no automatic post-training benchmark. Follow run-openasr-exp safety constraints and shared pipeline monitor.
/verl-asr-run Run standalone evaluation recipe/phimm/config/v2609_asr/eval/eval_2609r2_openall_mix_zip.yaml on <pool> with model <verified-last-checkpoint-hf-export>, trainer.experiment_name=<candidate-eval-name>, trainer.nnodes=<training-nnodes>, trainer.resume_mode=disable, actor_rollout_ref.model.lora_rank=0. Use the successful training run's effective node count, not the eval YAML default or currently available node count. Keep the early recipe.phimm.cache_eval_audio task caching physical ZIP archives only on the main pod alongside decoding; its completion or failure must not block evaluation submission. Do not launch prefetchers on worker pods. Follow the same pipeline constraints and monitor.
/openasr-report <train-stem>@step<N> --metrics <artifact-dir>/metrics.json --model-info <artifact-dir>/model_info.json --dataset-results <artifact-dir>/dataset_results.json --out <report.xlsx>
```

### Overrides to the general runner

The following pipeline constraints take precedence over `verl-asr-run`'s
generic defaults during both handoffs:

- Select the **last complete** checkpoint, never the best validation checkpoint.
- Run the **specified eval YAML**, not the full in-house/reference benchmark
  suite. Do not invoke `eval-2609-benchmark-report` automatically.
- Precache evaluation audio with the existing
  [cache_eval_audio.py](../../../recipe/phimm/cache_eval_audio.py), which reuses
  [verl/audio_cache.py](../../../verl/audio_cache.py). Do not implement another
  downloader or claim completed precaching from process startup. Downloads run
  only on the main pod in the background; incomplete or failed prefetching must not gate evaluation,
  whose existing on-demand cache handles audio not yet prefetched.
- Keep training and evaluation on the requested pool. A Brix pool can contain
  multiple Ray GPU nodes: `verl-n4-i0` normally has four pods and 32 GPUs.
  Set evaluation `trainer.nnodes` to the successful training run's effective
  value. Do not force `trainer.nnodes=1` merely because there is one pool name,
  or change the count to match currently available nodes.
- Stop only the authorized previous Ray submission. Never call
  `ray_job.py cleanup`, including through `submit_job.sh`'s default cleanup.
- Do not pause, resume, stop jobs on, or otherwise manage unrelated pools.
- Derive LoRA rank and alpha from the effective training config/checkpoint;
  do not copy the general runner's fixed 320/640 export values.

## 1. Preflight and durable state

Create a todo list for preflight, replacement/submission, training, export,
evaluation audio precache, evaluation, and report verification. Mark completion
only from observed state. Track audio precache independently, not as a
prerequisite for evaluation; its launch and monitoring overlap other phases.

Read the training and eval YAMLs and their defaults. Compose the training
config with the repository's Hydra resolvers; record the resolved model path,
LoRA rank/alpha, `trainer.nnodes`, GPUs per node, total steps/epochs,
`save_freq`, `resume_mode`, and local/blob output roots. Preserve save cadence
and training hyperparameters. Ensure epoch bounds permit the requested step
target. Surface conflicts rather than silently changing training semantics.

Use ignored `tmp/run_openasr_exp/<train-stem>/` for a `state.json`,
resolved config snapshots, submission output, full logs, metrics JSON, model
metadata, result-path mapping, and export provenance. On resume, reuse any
existing state path recorded by this pipeline's monitor rather than starting
a duplicate run because the skill was renamed. Store:

- Normalized train/eval paths, pool, experiment names, overrides, code snapshot,
  output roots, checkpoint policy `last_complete`, and replacement authorization.
- `training_nnodes` from the successful training run's resolved config and
  runtime logs, and `eval_nnodes`, which must equal `training_nnodes`. Update
  training topology provenance if an authorized recovery changes it.
- Current phase, stopped job ID, training/export/eval job IDs and statuses,
  observed step/target, selected checkpoint path and step, W&B and Ray URLs.
- Audio precache input paths and hashes, resolved dataset/manifest inventory,
  code snapshot, download worker count, and main-pod identity, user/HOME/cache root,
  command, remote PID/background execution identity, status, exit code,
  log/summary paths, counts, and completion time. Keep this separate from the
  main training/export/evaluation phase, including explicit prefetch failures.
- Report path, verified artifact paths, and current monitor identity.

Do not store credentials or signed URLs. Never treat files left by a previous
run with the same experiment name as proof this run succeeded. Reconcile live
jobs, run identity, logs and artifact provenance before reusing state.

Maintain `recipe/phimm/config/verl_job.txt` using the general runner's
timestamp/table/Reports format. Preserve unrelated entries, replace only this
pool's current row, and do not retain superseded job IDs. Use `n4-i0` rather
than ambiguous `i0` when multiple pool sizes share the index. Report phases as
training, exporting last checkpoint, evaluating the actual dataset, or building
report, with concurrent audio-prefetch progress alongside the main phase;
adapt best-checkpoint wording to this last-checkpoint policy.
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

Invoke `verl-asr-run` for the training-only handoff in the required sequence.
Use its submission, monitoring and recovery workflow subject to this skill's
constraints; the commands below describe execution through that loaded skill,
not an independent replacement runner.

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

As soon as this training submission is verified running, start section 5's
audio-prefetch task only on the main pod using the resolved eval
dataset inventory. Do not wait for the final checkpoint or HF export: audio
inputs do not depend on checkpoint weights. Run inventory preparation and
downloads asynchronously so they do not hold up training monitoring, export,
or evaluation submission.

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

## 5. Evaluate on the same pool and node count as training

Invoke `verl-asr-run` for the standalone-evaluation handoff only after the
last-checkpoint export is verified. Supply the exact eval YAML and exported
model path; do not enable the runner's best-checkpoint benchmark pipeline.

Compose the exact requested eval YAML and inventory its datasets. Retrieve
`training_nnodes` from the successful training run's effective resolved
config, including overrides, and confirm it against runtime logs. Set eval
`trainer.nnodes` explicitly to that value, overriding any eval YAML default.
For example, training with `trainer.nnodes=4` requires evaluation with
`trainer.nnodes=4`, even if the eval YAML defaults to 1.

Require the pool to be idle and at least `training_nnodes` healthy GPU Ray
nodes to be available before launching. Count GPU-capable nodes, not Brix
pool names or CPU-only Ray nodes. If too few are available, recover node
readiness or record a blocker and wait; never silently reduce the evaluation
node count, increase it to use spare nodes, or move evaluation to another
pool. If training topology provenance is missing, recover it from this run's
resolved config/logs rather than guessing from the pool size.

Sync current code again before evaluation. Reconcile the early background
main-pod prefetch task, but do not wait for its preparation or transfers to finish
before asking `verl-asr-run` to submit evaluation.

### Early, non-blocking evaluation audio precache

1. Use the **resolved `data.val_data` inventory from the actual eval config**,
   including overrides and nested dataset lists. The cache CLI accepts a
   dataset YAML, not a Hydra job YAML: do not pass `eval_2609r2_openall_mix_zip.yaml`
   directly. Keep evaluation pointed at its original manifests and config.
2. Build a cache-only JSONL inventory and dataset YAML beneath
   `tmp/run_openasr_exp/<train-stem>/audio_precache/` from all configured sources.
   Use the existing `jsonl_dataset`, `path_map`, and `rename_fields` helpers in
   `recipe/phimm/data/dataset.py` to read and normalize source references in the
   loader's order (`path_map` before `rename_fields`). Export only the effective
   audio reference fields; do not decode audio or run prompt/reward generation.
   The default ZIP suite loads `mixlang_fy26q2_zh_seg_zip.yaml`,
   `openasr_verb_langhint_zip.yaml`, and `openasr_ml_verb_langhint_zip.yaml`.
   MixLang renames `WavPath` to `audio_path`, AA entries rename `url` to
   `audio_path`, and the other OpenASR/ML ZIP manifests supply their audio
   references directly. Apply the actual configured mappings, not the legacy
   non-ZIP English `audio/` path mapping. Raw input scanning without these
   mappings is not sufficient.
   Preserve complete `archive.zip!offset:length` references and chunk/time
   selectors in the inventory. Use `verl.audio_cache._split_audio_source`
   to identify and deduplicate physical archives for expected cache counts;
   do not deduplicate distinct member references by discarding their offsets.
   The existing cache CLI downloads each physical ZIP archive once; do not
   extract members, rewrite manifests, or implement another archive downloader.
   Verify every configured source is covered and every normalized reference
   resolves to the same physical file the evaluator will read. If an override
   requires additional path-producing transforms, use its existing loader
   helpers to resolve those references too; do not guess paths.
3. The cache-only YAML must contain `dataset_name: jsonl` and `jsonl_paths`
   pointing to the generated inventory at its verified remote location. Sync
   it and the inventory only to the main pod along with the cache script.
   For a directly compatible dataset YAML such as
   `recipe/phimm/config/data/val_data/openasr_verb_langhint_zip.yaml`, the
   original dataset YAML can be passed directly only after verifying its raw
   fields match the supplied `--audio-fields` (including `url` for AA entries).
   A single dataset YAML does not replace inventory coverage of the full suite.
   Never rewrite source
   manifests. Unsupported dataset types/storage prefixes, unresolved paths,
   empty inventories, insufficient disk, or missing `bbb` authentication must
   be recorded as explicit prefetch failures. Diagnose them independently;
   they do not block evaluation or permit dropping evaluation datasets.
4. Identify the pool's main/head pod and use `brix ssh` to target that pod only,
   verifying the actual hostname and installed CLI's pod-selection behavior.
   Do not launch prefetchers on worker pods or fan out according to
   `training_nnodes`; evaluation topology remains unchanged.
   Use the evaluation interpreter and user/HOME, verify `bbb` authentication
   and available disk, and start this command from `/root/code/verl` on the main pod
   as a background execution, without waiting for it to finish:

   ```bash
   python -m recipe.phimm.cache_eval_audio \
     <verified-cache-dataset.yaml> --workers 16 --audio-fields audio_path
   ```

   This downloads physical ZIP archives into the evaluator's persistent Orange cache under that user's
   `~/data`, including existing retries, timeouts, locks, and atomic writes.
   If the verified inventory uses other effective audio fields, supply those
   explicitly. Do not change `HOME` or use a different cache root. This warms
   the main pod's cache only; do not claim worker-local caches are populated.
   Capture a remote process/background identity and
   verify startup and log activity, then immediately continue the main pipeline.
   Use the verified remote background execution mechanism so the task remains
   alive across subsequent pipeline phases. Use a single download process on
   the main pod; if I/O contention harms training or decoding, lower its concurrency.
5. Capture full logs, exit status, and JSON summary for the **main pod**. Require
   exit code zero and `unique_files == already_local + downloaded` only when
   marking the main pod's precache complete, and verify the expected unique
   physical-file count against the inventory (unique archives for ZIP references,
   not ZIP members or segments). Record running, failed, and
   completed states accurately; never wait for prefetch completion before evaluation.
   Diagnose and retry missing/failed prefetch work in the background. Existing
   files are reused, so the main pod need not redownload immutable audio.
   On resume, reconcile the main pod's identity, HOME, inventory hashes and local cache
   availability; start the CLI asynchronously on a replaced main pod or whenever
   cache evidence is stale, unless the same task is already running. Do not
   treat an old completion flag as proof on a new pod. Keep downloads running
   in parallel with decoding: existing file locks and atomic writes coordinate
   the prefetcher with the evaluator's on-demand cache safely.

### Submit and verify evaluation

Use a candidate-specific experiment name, for example
`<eval-stem>__<train-stem>_step<N>`, so evaluation outputs cannot overwrite
another model's results. Pass the verified HF export as the model, disable
new LoRA adapters for the already-merged model, and prevent training resume:

```bash
brix ssh <pool> -- 'bash -l /root/code/verl/quick_run.sh <eval.yaml> \
  trainer.experiment_name=<candidate-eval-name> \
  trainer.nnodes=<training-nnodes> \
  trainer.resume_mode=disable \
  actor_rollout_ref.model.path=<verified-hf-export> \
  actor_rollout_ref.model.lora_rank=0'
```

Check the composed config is validation-only and uses the intended candidate.
Before submission, verify `eval trainer.nnodes == training_nnodes`; audio
prefetch may still be preparing, running, or failed and is not a submission
gate. After
startup, verify the effective config and runtime worker topology use that
same node count. Do not advance a mismatched evaluation to reporting; correct
and resubmit only this pipeline's evaluation under its recovery policy.
Record the exact evaluation output root, code snapshot, job ID and W&B URL.
If remote model caching would duplicate the export per worker, use a
verified identical shared pod-local model path on every participating pod;
keep the original exported model URI in provenance.

Monitor until Ray `SUCCEEDED` and all configured datasets have final metrics
and detailed artifacts. For `eval_2609r2_openall_mix_zip.yaml`, this includes
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

Use one shared recurring five-minute monitor across the pipeline and both
`verl-asr-run` handoffs; do not install a competing generic runner monitor.
After verifying a live submission, install or update that monitor using the
scheduling tool. Its prompt must identify this skill, `verl-asr-run` for
training/evaluation and `openasr-report` for reporting, the durable
state path, node, train/eval configs and current Ray job ID, and explicitly
retain the **last complete checkpoint -> specified eval -> OpenASR report**
policy with **early main-pod-only audio precache in parallel, never gating decoding**.
Inspect existing schedules and replace only this pipeline's stale
monitor after resubmission or phase changes. Plain text claiming monitoring
is installed is not sufficient.

On each tick, rediscover current jobs and reconcile artifacts before taking
action; update durable state and the pool's status row. Resume from the
first incomplete phase. Never double-submit an export/evaluation, restart
training after it succeeded, or stop unrelated work encountered on recovery.
`status` only observes and reports; it does not schedule or advance work.

Keep monitoring through training, export, evaluation and workbook validation,
tracking background audio precache alongside each phase. Reconcile the main pod's
prefetch process, logs and summary without waiting or starting duplicate
commands. Once evaluation and report verification finish, stop only this
pipeline's still-running main-pod prefetch process by its verified exact process
identity, record it as cancelled after evaluation, and verify termination.
No unfinished optional prefetch task may keep the main pipeline open.
Stop the schedule only when all requested work is complete or the user
cancels it. If genuinely blocked, record the exact blocker and do not claim
success or retry destructively. Pause a pool resumed by this pipeline only
after it is idle and has no remaining scheduled work.

The final response includes the workbook path, actual selected last step and
checkpoint URI, English and ML mean `p_err` and WERR, separate MixLang results
when applicable, and training/evaluation IDs and links. Report a running
pipeline as running, not complete merely because a monitor was installed.

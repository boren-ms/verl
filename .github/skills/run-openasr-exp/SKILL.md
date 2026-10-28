---
name: run-openasr-exp
description: 'Orchestrate a multi-stage OpenASR workflow: optional training, exact or last-complete checkpoint export, standalone OpenASR evaluation, and canonical workbook generation. Use for train-and-report, checkpoint evaluation queues, or resuming that pipeline. Use verl-asr-run for one Ray job and openasr-report for workbook-only requests.'
---

# Run OpenASR Experiment

Own one durable pipeline:

```text
optional training -> requested or last complete checkpoint -> verified HF export
-> standalone OpenASR evaluation -> canonical OpenASR workbook
```

This is an orchestrator. Delegate each individual remote Ray job to
`verl-asr-run`; delegate workbook construction to `openasr-report`. Do not copy
their job mechanics or workbook formulas into this skill.

An instruction-only request changes these files only; it never authorizes
launching, stopping, resuming, or scheduling remote work.

## Routing and contract

| Request | Owner |
| --- | --- |
| One training/evaluation/generation job | `verl-asr-run` |
| Workbook from existing metrics | `openasr-report` |
| Full train/export/evaluate/report or checkpoint queue | this skill |

Inputs and defaults:

- Training config is required only when launching training. Evaluation-only
  requests recover producer config, base model, rank/alpha, topology, and output
  root from checkpoint provenance.
- Use the exact authorized Brix pool. Normalize abbreviations only when
  unambiguous; never choose another pool silently.
- Default eval config is
  `recipe/phimm/config/v2611_dev/eval/eval_2609r2_openall_zip.yaml`.
- An explicit checkpoint step/path wins. Otherwise select the numerically last
  complete checkpoint, not the best validation checkpoint.
- Default report path is
  `tmp/openasr_report/<train-stem>_step<step>.xlsx`.
- `status` is read-only. It must not submit, stop, schedule, or advance work.
- Stop/replace is disabled unless the user identifies one exact job or gives an
  unambiguous `stop previous` instruction.

Checkpoint artifact readiness—not source-training lifecycle—gates evaluation.
Never stop or change source training merely to evaluate a complete checkpoint.

## Durable state and authorization

Store pipeline state under ignored
`tmp/run_openasr_exp/<train-stem>/`. Record:

- normalized train/eval configs, authorized pools/topology, experiment names,
  overrides, code revisions, output roots, and checkpoint policy;
- producer base model, LoRA rank/alpha, world size, save cadence, and provenance;
- training/export/eval submission IDs and states, W&B/Ray links, and replacement
  authorization;
- selected checkpoint fingerprints, missing/complete shards, HF export proof,
  exact evaluation result paths, report inputs/path, and verification state;
- checkpoint-queue order, fresh readiness observations, deferral count/time,
  and reciprocal dependencies;
- audio-prefetch inventory hash, main-pod identity, user/HOME/cache root,
  process identity, logs, summary, and status;
- one shared monitor identity for the whole pipeline.

Do not store credentials or signed URLs. On resume, reconcile live jobs and
artifacts before reusing state. Same names or old files are not proof of the
same run.

## 1. Preflight and optional replacement

Use `verl-asr-run`'s pool checks across every GPU pod. An occupied pool is a
blocker unless replacement was authorized. If replacing a job, prove its exact
Ray submission ID and pool/config identity, stop only that submission, wait for
terminal state and resource release, and replace only its monitor. Never call a
broad cleanup wrapper or manage unrelated pools.

Compose the actual training and eval configs. Record effective topology,
model/rank/alpha, steps/epochs, save frequency, resume policy, datasets, and
output roots. Preserve recipe semantics and surface conflicts.

## 2. Optional training handoff

Skip this stage for an explicit/queued checkpoint evaluation. Otherwise invoke
`verl-asr-run` once for the supplied training config and authorized pool.
Request training only; checkpoint selection and downstream stages remain owned
here.

After the job is verified running, start the audio-prefetch task described
below. Continue monitoring training to its own terminal condition, but do not
make training completion an evaluation gate when a requested checkpoint is
already complete and authorized evaluation resources are available.

## 3. Resolve checkpoint readiness

Enumerate this producer's `global_step_<N>` directories numerically. A verl
checkpoint is ready only when every expected nonempty
`model_world_size_<W>_rank_<R>.pt` shard exists for `R=0..W-1`, writes/uploads
are complete, and provenance matches the requested producer. A directory name
or source job state is insufficient.

- For an explicit step/path, inspect only that checkpoint; never substitute.
- Without an explicit step, choose the largest complete step currently
  observed. An incomplete newer directory does not disqualify an older complete
  one.
- A verified HF export for the exact checkpoint is reusable. Missing HF weights
  with complete actor shards means export is needed, not that the checkpoint is
  unready.

### Queue policy

On each monitor tick, reconcile active work first and inspect each unfinished
request at most once. Move a genuinely missing/incomplete checkpoint to the
tail with a fresh observation, preserve all other relative order, and inspect
the next request immediately. Start the first ready request. Authentication,
unreadable artifacts, and unresolved provenance are errors to diagnose—not
fabricated missing-checkpoint results. Never duplicate an active export,
evaluation, or report.

## 4. Export the exact checkpoint

Use the producer's actual base model and effective rank/alpha. Run the current
converter from the repository:

```bash
python3 plugins/qwen35_audio/src/hf_qwen35_audio/convert_verl_to_pt.py \
  --input <checkpoint-root> \
  --output <hf-dir>/model.pt \
  --match-lora-merged \
  --lora-alpha <effective-alpha> \
  --lora-rank <effective-rank>
```

For `az://` shards, use a dedicated local cache. Verify complete rank coverage,
merged adapter-pair count, and rank/alpha provenance. Convert the top-level
tensor dictionary to `model.safetensors`, removing `.base_layer.` from keys
with collision checks and confirming no adapter tensors remain. Copy config,
tokenizer, preprocessor, and required custom model code from the actual base;
do not copy base weights or a stale sharded index.

Verify readability, keys/shapes, assets, and uploaded files before recording the
HF URI. Invalidate only identified stale cache/export files.

## 5. Early nonblocking audio precache

Prefetch is an optimization, never an evaluation gate.

1. Resolve the actual eval config's complete `data.val_data` inventory,
   including nested lists, `path_map`, and `rename_fields` in loader order.
2. Build a cache-only JSONL/YAML under the pipeline state directory. Preserve
   complete `archive.zip!offset:length` and segment selectors. Deduplicate
   physical archives without discarding distinct member references.
3. Verify every configured source is covered. Unsupported inputs, unresolved
   paths, empty inventory, auth, or disk problems are explicit prefetch
   failures; never drop evaluation datasets to make prefetch succeed.
4. On the verified main/head pod only, using the evaluator's user/HOME and
   cache root, start one background command from `/root/code/verl`:

   ```bash
   python -m recipe.phimm.cache_eval_audio \
     <verified-cache-dataset.yaml> --workers 16 --audio-fields <fields>
   ```

5. Capture PID/background identity, logs, exit status, and JSON summary. Do not
   launch worker-pod prefetchers or wait for completion before evaluation.
   Reconcile identity/hashes before retrying and lower concurrency if I/O harms
   decoding. At pipeline completion, stop only this pipeline's remaining exact
   prefetch process; optional prefetch must not keep the pipeline open.

A `_zip` filename alone does not prove ZIP-backed audio. Inspect real manifest
rows for ZIP byte-range references and verify a byte-range read when preparing
new ZIP manifests.

## 6. Standalone evaluation handoff

Invoke `verl-asr-run` once with the exact eval YAML and verified HF export.
Use a candidate-specific experiment name, set `trainer.resume_mode=disable`,
set the verified evaluation topology explicitly, and set
`actor_rollout_ref.model.lora_rank=0` for the merged model.

Default evaluation topology is the producer's effective `trainer.nnodes`, not
the eval YAML default, pool-name count, or currently idle node count. Change it
only with explicit authorization. Require the evaluation pool and required Ray
GPU nodes to be idle; source training on a separate pool may continue.

Sync current code, reconcile—but do not await—prefetch, submit, and verify the
effective config/runtime topology. Monitor until Ray reports `SUCCEEDED` and
every configured dataset has final `p_err` metrics plus exact detail artifacts.
One intermediate metric line is not completion. Preserve separately any
explicit non-OpenASR datasets rather than inserting them into canonical rows.

## 7. Report handoff

Create canonical input artifacts in the pipeline state directory:

- `metrics.json`: candidate `val-aux/<source>/p_err/mean@1` fractions only;
- `model_info.json`: export URI, eval config, pool, Ray ID, W&B URL, and code
  revision;
- `dataset_results.json`: canonical dataset key to exact detail-result file,
  never a parent directory.

Resolve source-to-canonical dataset identities from the composed manifests; do
not alias cleaned and uncleaned or short- and long-audio datasets by name.

Invoke `openasr-report` with those three files and the requested output path.
That skill owns baseline values, canonical row layout, aggregation, formulas,
formatting, workbook extension, and workbook quality gates. Do not duplicate or
override those rules here. Reopen and verify the workbook and its model/result
metadata before completing the pipeline.

## 8. Monitoring and completion

Maintain one chat-attached recurring monitor for the whole pipeline when a
native scheduling tool is available. Its durable prompt identifies this skill,
state path, current pool/config/submission, queue policy, prefetch identity,
recovery boundary, completion condition, and stop condition. Update that one
monitor across phase and submission changes; do not install competing monitors
for delegated `verl-asr-run` jobs. Keep unchanged polls quiet unless the user
requested every-poll updates.

Use the native scheduling tool, not raw scheduling directives or automation
storage edits. On each tick, rediscover live jobs and artifacts, resume the
first incomplete phase, and never double-submit. Stop the monitor when the
requested workbook is verified, the user cancels, or a terminal blocker needs
user action.

The pipeline is complete only when the chosen checkpoint and export are
verified, the requested evaluation and all datasets succeeded, exact report
inputs are preserved, and the workbook passes `openasr-report` checks. Final
output includes checkpoint step/URI, training and evaluation identities/links,
report path, English and ML summary metrics, separate extra-suite results when
present, and material provenance limits.

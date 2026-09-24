---
name: verl-asr-run
description: 'Submit, inspect, monitor, or repair a specific ASR training, evaluation, or generation Ray job on Brix. Owns the per-job lifecycle and HF checkpoint export; optionally hands an explicitly requested post-training benchmark to eval-2609-benchmark-report. Use for a config/node/job ID, not report-only analysis.'
argument-hint: '<config-path-or-name> [--node <pool>] [status|submit|monitor|repair] [post-training evaluation when requested]'
---

# verl ASR Job Lifecycle

Own one job's input resolution, submission, monitoring, repair, and result
retrieval. Read the [shared remote execution contract](../references/remote-execution.md)
before remote actions. Use `remote-development` for connectivity and environment setup.

## Scope and Handoffs

- A training request ends with training results and durable terminal state.
  Post-training export/evaluation is disabled unless explicitly requested.
- A status request returns a read-only snapshot. A monitoring request follows
  the requested job; persistent scheduling follows the shared contract.
- [eval-2609-benchmark-report](../eval-2609-benchmark-report/SKILL.md) owns the
  standard suite, its evaluation queue/recovery, and workbook. For several
  explicitly requested models, keep their queues separate and assign disjoint
  pools or process them sequentially.
- When invoked for a child benchmark job, run its supplied config and overrides
  only. Return state/artifacts to the caller; do not invoke the benchmark parent
  again or install a child schedule.
- For result-only utterance analysis use
  [asr-detail-compare](../asr-detail-compare/SKILL.md), not a new evaluation job.

## Inputs and Entrypoints

Resolve the exact config path under `recipe/phimm/config/`, not just a basename
that may occur in multiple directories. Preserve its effective model path,
dataset/scoring settings, output root, and training `trainer.save_freq`.

| Input | Meaning |
|---|---|
| Config | Required for a new job; recoverable from an existing job command for status/monitoring. |
| Node/pool | User/caller allowlist, or an eligible verified-free Ready `verl-*` pool. |
| Model path | Config default unless explicitly overridden. |
| Post-training evaluation | Opt-in; selects one best complete saved checkpoint after training succeeds. |
| Report | Owned by the selected benchmark skill, not this job runner. |

[quick_run.sh](../../../quick_run.sh) is authoritative for prefix routing:

| Prefix | Python module |
|---|---|
| `long_eval_*` | `recipe.phimm.main_long_eval_asr` |
| `gen_*` | `recipe.phimm.main_asr_gen` |
| `eval_*` | `recipe.phimm.main_asr_eval` |
| `remax_*` | `recipe.phimm.main_asr_remax` |
| `grpo_*`, `gdpo_*`, `gmpo_*`, `gspo_*` | `recipe.phimm.main_asr_grpo` |
| `rloo_*` | `recipe.phimm.main_asr_rloo` |
| Other training configs | `recipe.phimm.main_asr_dapo` |

`eval_*` loads `actor_rollout_ref.model.path`; `long_eval_*` loads `model.path`.
Long evaluation uses SVAD-explode, generation, regrouping, and TER/EER scoring,
and writes per-dataset detail JSONL and `measures.json` under `data.output_path`.
HF model paths are not `trainer.resume_from_path`.

## Procedure

### Step 0 -- Select a Ready Pool

Apply the shared occupancy/topology checks and caller restrictions. Do not pause
other idle pools, suppress failed Ray queries, or cancel jobs by config name.
For requested post-training evaluation, keep the training pool reserved for
sequential child work after its training processes release resources.

### Step 1 -- Resolve Inputs and Topology

Record config path/directory/name, module, effective model and output paths,
experiment identity, and verified `trainer.nnodes`. For evaluation always pass
the verified participating node count explicitly. Reusing a training pool does
not imply `trainer.nnodes=1` unless it is actually a one-node pool.

If post-training evaluation is enabled, record that scope before submission.
Do not change save cadence to manufacture extra evaluation checkpoints.

### Step 1b: Maintain the Pipeline Cache

Use `recipe/phimm/config/verl_job.txt` for the latest tracked training/evaluation
state. Live Brix/Ray state wins over cached placement. Cached entries are recovery
hints for the requested jobs, not authorization to adopt unrelated work.

For submit/monitor/repair actions, update after submission, observed status/phase changes, resubmission, and each
export/report transition. With a workload coordinator, return these updates to
it; only the coordinator writes the aggregate cache. Otherwise read/merge the
existing cache, preserving unrelated rows and valid report entries.
For a status-only request, report the snapshot without rewriting the cache.

Keep the existing human-readable format:

1. `updated_at_utc: <ISO-8601 UTC timestamp>`
2. One aligned node/job table with columns `NODE`, `RAY JOB ID`, `STATUS`,
   `JOB / CONFIG`, and `PROGRESS / PHASE`.
3. `Reports:` with one concise current-state bullet per tracked pipeline.

Use at most one current tracked row per pool, naturally sorted by node number.
Replace superseded Ray IDs in place, including when export or benchmark work
replaces training on that pool. Use `none` for unknown IDs; do not invent them.
Preserve an existing Unicode table style. A typical report bullet is
``- `training-config` [`i0`]: evaluating digits_enus step 100``.

Derive phase from current logs:

- `training X/N (P%)`, or `training startup`
- `evaluating <dataset> (<phase>)` or `generating <dataset> (<phase>)`
- `selecting best checkpoint`, `waiting for best checkpoint step <step>`,
  `exporting best checkpoint step <step>`, or `building report step <step>`
- Terminal activity with `(complete)` / `(failed)` and the actual Ray status.

Standalone jobs need no report bullet. Keep artifact paths, detailed metrics,
requested checkpoint sets, and historical provenance in the owning workload's
durable manifests/state, not duplicate snapshots in this table. Preserve valid
cached report intent during recovery; migrate legacy queue/history fields to
durable workload state before simplifying them. Never infer new evaluation
authorization merely from a completed training row.

Prepare a complete merged cache update, then atomically replace the file using a
sibling temporary file and rename. Re-read if another writer changed the source;
do not overwrite concurrent updates. Never store secrets or raw logs, and do not
commit routine cache refreshes.

### Step 2 -- Sync and Submit

Verify/sync the workspace as described in the shared contract. For an unchanged
config, after sync use [submit_job.sh](../../../submit_job.sh) with cleanup and
implicit sync disabled:

```bash
bash submit_job.sh <POOL> recipe/phimm/config/<family>/<CONFIG>.yaml false false false
```

Its optional arguments are `dry_run`, `cleanup`, `sync_code`; omitted cleanup
defaults to `true`, so never rely on defaults. Its dry-run mode still performs
enabled sync/cleanup and is not a side-effect-free validation command.

For model/output/topology overrides, submit directly from the remote repository
root, preserving `quick_run.sh`'s runtime environment:

```bash
cd /root/code/verl
python3 ray_tool.py prepare_env
cuda_compat_ld_path="/usr/local/cuda-13.0/compat:/root/.pyenv/versions/3.12.9/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
ray job submit --working-dir=/root/code/verl \
  --runtime-env-json "{\"env_vars\":{\"LD_LIBRARY_PATH\":\"${cuda_compat_ld_path}\"}}" \
  --no-wait -- python3 -m <MODULE> \
  --config-dir recipe/phimm/config/<family> --config-name <CONFIG> \
  'trainer.experiment_name=<UNIQUE_EXPERIMENT>' \
  'trainer.nnodes=<VERIFIED_NNODES>' \
  '<MODEL_OVERRIDE_KEY>=<MODEL_PATH>'
```

Add the caller's output overrides (`trainer.default_hdfs_dir` for validation
JSONL; `data.output_path` for long evaluation) to prevent collisions. Shell-quote
resolved values. Capture the Ray submission ID and record it immediately.
Submission success is not job completion.

### Step 3 -- Monitor and Recover

Inspect status, logs, and GPU activity during each active monitoring pass:

```bash
brix ssh <POOL> -- 'bash -l -c "ray job status <JOB_ID>"'
brix ssh <POOL> -- 'bash -l -c "ray job logs <JOB_ID> | tail -n 40"'
brix ssh <POOL> -- 'bash -l -c "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"'
```

Rediscover replacement IDs by config, experiment, and model/output paths, not
just the original ID. Track startup, model download/loading, LoRA application,
vLLM/CUDA-graph initialization, W&B initialization, training, validation,
checkpointing, decoding, scoring, and artifact upload as applicable.

Confirm Ray `SUCCEEDED` and expected outputs before claiming success. `FAILED`
and `STOPPED` are not success; disappearance from a job listing or a final WER
line is insufficient. Recover expired Ray history only from complete matching
durable artifacts.

For an authorized repair: retain the traceback, diagnose locally, make a
targeted fix, validate it, sync, and resubmit only the failed work. Record the
replacement ID. Do not change the requested datasets, checkpoint, or scoring
contract to make a job pass. Common diagnostics:

| Symptom | Check |
|---|---|
| OOM | Batch sizes, rollout count, GPU memory utilization; preserve scoring semantics. |
| Missing data/checkpoint | Exact blob path, shard completeness, config/environment. |
| Import/function failure | Installed package and module/API compatibility. |
| Model incompatibility | HF/PEFT/vLLM support and checkpoint key mapping. |
| Sync/connectivity failure | Ready state and `remote-development` setup. |

Return failures explicitly when repair is outside scope or blocked.

### Step 3g -- Metrics and Status

Show current job ID, Ray status, progress, pool/config, GPU utilization/memory,
and actual W&B/Ray dashboard links when available. Read URLs from logs rather
than inventing a run identity. Under a workload coordinator, return structured
updates for its aggregate display instead of duplicating node tables.

Accumulate metrics across polls; show only new rows during monitoring and the
full trajectory in the final training summary:

| Training field | Log key |
|---|---|
| score/mean | `critic/score/mean` |
| entropy, pg_loss, grad_norm, lr | `actor/<field>` |
| throughput | `perf/throughput` |
| step_time | `timing_s/step` |
| progress | `progress` |

For validation status, display `val-aux/<dataset>/p_err/mean@1` and `p_edge`
in separate tables, as percentages. Show quality metrics only when `p_fmt` or
`p_lang` differs from 100%, or `p_bracket` differs from 0%; otherwise note that
they are nominal. Do not dump raw count/reward fields in status tables.
Benchmark-specific final metrics (including digits CER/WER and `val-core`
DTER) remain governed by the calling report skill, not this display filter.

For OpenASR-ML add arithmetic per-language and overall averages over the actual
dataset rows, excluding intermediate averages. Map `de/fr/it/es/pt` to
German/French/Italian/Spanish/Portuguese. Never invent missing datasets.
Error rates must be non-negative; WER can exceed 100% because of insertions.
Flag unexpectedly high rates or diverging training metrics for investigation
rather than clamping values.

### Step 3h -- Select the Best Checkpoint (Opt-in)

Only after training `SUCCEEDED`, and only for requested post-training evaluation:

1. Enumerate complete saved/uploaded checkpoints and validation `p_err` steps.
2. Compare the arithmetic mean across the largest common configured dataset set
   present at all eligible steps. Choose the lowest mean, breaking ties in favor
   of the later step.
3. If no eligible step has comparable metrics, use the latest complete checkpoint
   and explicitly report the fallback. Record the selected step, score, and
   dataset set.
4. Reconcile existing export/benchmark artifacts to avoid duplicate work. Wait
   for the same training pool to pass occupancy checks before export/evaluation.
   Do not launch external benchmarks during training or move to another pool
   without an authorized scope change.

### Step 4 -- Training Summary

Report the full validation trajectory, final training metrics, checkpoint
location, elapsed training time, W&B link, and WER trend. Include the selected
step/score only when post-training evaluation was requested. Training-only work
ends here after terminal cache state is verified.

### Step 4a -- Export a Qwen3.5-Audio Checkpoint to HF

This section also owns export requested by an evaluation workload. Resolve the
exact output root from the effective training config/provenance and verify all
actor shards for the selected `global_step_<STEP>`. Do not assume every model
uses the fixed baseline/rank from
[lora-weight-transfer](../lora-weight-transfer/SKILL.md).

For LoRA checkpoints, read `lora_alpha` and `lora_rank` (or the actual scaling
rule) from training provenance. Missing/incompatible provenance blocks export;
never silently assume 640/320. From the remote repository root:

```bash
python3 plugins/qwen35_audio/src/hf_qwen35_audio/convert_verl_to_pt.py \
  --input '<CHECKPOINT_DIR_OR_AZ_PATH>' --output '<EXPORT_DIR>/model.pt' \
  --match-lora-merged --lora-alpha <TRAIN_ALPHA> --lora-rank <TRAIN_RANK>
```

Use `--lora-scaling` instead only when the training scaling is explicitly known
and not alpha/rank. For full-weight checkpoints, omit LoRA flags and unwrap the
converter's `module` payload before HF serialization. The LoRA-merged mode emits
a bare state dict; both modes need validation, not an assumed payload shape.

Before publishing:

1. Normalize `.base_layer.` key segments to `.` and reject duplicate resulting
   keys; verify every value is a tensor and no unmerged LoRA adapters remain.
2. Save contiguous tensors with `safetensors.torch.save_file`, reopen the result,
   and check keys/shapes/dtypes against the base architecture. Delete temporary
   `model.pt` only after validation succeeds.
3. Copy config, tokenizer, processor, and required custom code from the actual
   base model into the export directory. Do not hide failed copies or substitute
   metadata from a different model. Check HF compatibility and expected size.
4. Upload the complete directory to the selected checkpoint's `qwen_hf/` path,
   verify remote readability, and record the exact checkpoint/base/scaling.
5. If a failed export left stale cache data, identify and remove only the exact
   affected cache files after checking that no active job uses them. Never use
   recursive wildcard cache deletion.

Use a model-appropriate exporter for other architectures; do not apply this
Qwen-specific key conversion blindly.

### Step 4b -- Requested Best-Checkpoint Benchmark

Invoke the benchmark owner once with a single checkpoint and the training-pool
allowlist:

```text
/eval-2609-benchmark-report "<TRAIN_CONFIG>" --checkpoint "<STEP>=<CHECKPOINT_PATH>" --node "<TRAIN_POOL>" --out "tmp/eval_2609_reports/<TRAIN_CONFIG>.xlsx"
```

The [benchmark contract](../eval-2609-benchmark-report/SKILL.md#benchmark-contract)
is authoritative for required datasets (including en-US digits), baselines,
artifacts, prefixed sheet names, and final workbook checks. Add
`--include-digits-tier1` only when requested. On this one allowed pool child jobs
run sequentially with its verified topology.

Return each child's current phase/ID to the pipeline owner. Finish only after
the selected checkpoint's consolidated workbook passes its quality gates.
Present the workbook, benchmark summary, selected checkpoint/score, training
summary, and candidate/reference provenance; mark the report status complete.

### Step 5 -- Monitoring Ownership and Final Checks

Follow the [shared monitoring rules](../references/remote-execution.md#completion-and-monitoring).
Do not install a recurring schedule for a normal submit/status/monitor request,
or create one when a workload coordinator already owns monitoring.

Before delivery verify the tracked cache rows match current state, replacement
IDs and dataset phases are correct, and unrelated entries were preserved. For
training-only runs verify terminal training outputs; for an enabled benchmark
apply all of the benchmark owner's quality gates. Never claim the full pipeline
complete from job submission, a stopped job, or a partial workbook.

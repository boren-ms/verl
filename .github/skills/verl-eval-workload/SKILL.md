---
name: verl-eval-workload
description: "Launch, monitor, recover, and report 2607 ASR evaluation workloads for user-supplied model checkpoints on the dedicated verl-n1-i10 through verl-n1-i15 pool. Use when: launch evaluation jobs, evaluate checkpoints, run 2607 benchmarks, resume evaluation workload, refill eval nodes, or build evaluation reports."
argument-hint: "<model-or-checkpoint-paths> [steps] [status|launch|resume|report]"
---

# Run 2607 Evaluation Workloads

Launch or recover a 2607 ASR benchmark workload for the model names,
checkpoint paths, and steps supplied by the user. Preserve successful work,
avoid unrelated jobs, package durable outputs, and build reports as soon as
their inputs are complete.

This skill coordinates `eval-2607-benchmark-report` and `verl-asr-run`. The
live cluster and durable blob artifacts are authoritative; never rely on job
IDs or completion state copied from an earlier session.

## Required Inputs

Resolve these values from the user's request before submitting work:

1. Candidate model name or report label.
2. One or more candidate HF checkpoint paths, or a path template plus steps.
3. Benchmarks to run. Default to the complete benchmark suite below.
4. Durable artifact root. Derive the default from the candidate label and
   step, but confirm that it cannot collide with an unrelated workload.

If a required model or checkpoint cannot be determined unambiguously, ask the
user. Status and recovery operations may discover these values from existing
Ray commands and artifact manifests.

For a conventional verl training output, use:

```text
az://orngwus2cresco/data/boren/outputs/ver_2607/<MODEL>/global_step_<STEP>/qwen_hf/
```

Default evaluation artifacts:

```text
az://orngwus2cresco/data/boren/outputs/eval_2607_reports/<MODEL>_step<STEP>/<BENCHMARK>/candidate/
```

Default local reports:

```text
tmp/eval_2607_reports/<MODEL>_step<STEP>.xlsx
tmp/eval_2607_reports/<MODEL>_all_steps.xlsx
```

Sanitize user-provided labels before using them in experiment names or paths.
Never overwrite an artifact root whose manifest identifies a different model
path, config, or workload.

## Benchmark Suite

Unless the user requests a subset, run all three benchmarks for every supplied
checkpoint:

| Benchmark | Config | Model override |
|---|---|---|
| In-house DTER | `long_eval_inhouse_2607_all_seg30` | `model.path` |
| OpenASR-ML | `eval_openasr_ml_verb_2607` | `actor_rollout_ref.model.path` |
| MixLang | `long_eval_mixlang_fy26q2_zh_seg_2607` | `model.path` |

Always pass `trainer.nnodes=1`. Digits benchmarks are excluded unless the user
explicitly requests them through a separate workflow.

Default reference model:

```text
az://orngwus2cresco/data/speech/projects/phi-fastllm-2607/amlt-results/fast-llm-2607-qwen3-5-9b-s2-data-v3.4-sr-afteraudio/45000/qwen_hf/
```

## Dedicated Evaluation Pool

Only these nodes may run evaluation, checkpoint export, packaging, or report
work for this skill:

```text
verl-n1-i10
verl-n1-i11
verl-n1-i12
verl-n1-i13
verl-n1-i14
verl-n1-i15
```

- Never schedule this workload on any `verl-n1-i*` node outside that list.
- Inspect only the dedicated pool when reconstructing occupancy, and show only
  those nodes in user-facing workload tables.
- Do not stop, replace, or overwrite unrelated jobs on a dedicated node.
- A node is free only when it has no running Ray job and every GPU has at most
  5% utilization and at most 5000 MiB allocated.
- Reuse complete durable outputs. Never rerun a benchmark merely because its
  original Ray job is no longer listed.
- Package every completed evaluation before reassigning its node.
- Build a checkpoint workbook immediately when all requested benchmark outputs
  for that checkpoint are complete and packaged.
- Merge a model workbook immediately when every requested checkpoint workbook
  for that model is valid.

## Procedure

### 1. Build the work matrix

Expand the user inputs into one row per model, checkpoint, and benchmark.
Record the candidate path, report label, config, artifact root, and state:
`unavailable`, `missing`, `running`, `scoring`, `packageable`, `complete`,
`failed`, or `reported`.

Check that each candidate HF directory contains `model.safetensors`, config,
tokenizer, processor, and required custom model code before marking it
available. Do not invent missing steps or wait for unspecified future
checkpoints.

### 2. Reconstruct dedicated-pool occupancy

List Ready nodes, restricted to the dedicated pool:

```bash
brix pools 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E '^verl-n1-i(10|11|12|13|14|15)([[:space:]]|$)'
```

On every Ready dedicated node, collect both signals:

```bash
brix ssh <NODE> -- 'bash -l -c "
  python /root/code/verl/ray_job.py list 2>/dev/null || true
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits
"'
```

Classify a Ray job from its full command line and associate it with a work row
using the candidate model path, config, and experiment name. Commands using
`main_asr_eval` or `main_long_eval_asr` are evaluation jobs, but they belong to
this workload only when their inputs match the current work matrix.

### 3. Reconstruct durable completion

Inspect each requested candidate artifact root before launching anything. A
benchmark is complete only when:

- Long evaluation: every expected dataset has readable `details.jsonl` and
  `measures.json`.
- OpenASR-ML: all 13 expected dataset JSONL outputs exist and final W&B
  `val-aux` metrics were retained.
- The root has readable `wandb.json`, `key_metrics.json`, and
  `artifact_manifest.json`.

Expected complete-suite long-evaluation counts:

- In-house DTER: 18 dataset directories.
- MixLang: one `mixlang_fy26q2` directory.

If scorer outputs are complete but metadata files are missing, package the
existing run instead of rerunning it. The manifest must record the actual node,
Ray job ID, model path, config, `trainer.nnodes=1`, detailed-output paths, and
completion timestamp.

### 4. Export missing HF checkpoints when possible

When the user supplied a verl checkpoint rather than a ready HF checkpoint,
export it before evaluation:

1. Merge the verl checkpoint with the LoRA parameters from its training config.
   Do not assume fixed alpha or rank values; read them from provenance or ask
   the user when unavailable.
2. Replace every `.base_layer.` key segment with `.` when the exported state
   uses wrapped base-layer keys.
3. Save `model.safetensors` and remove temporary `model.pt` only after the
   safetensors file validates.
4. Copy base-model metadata, tokenizer, processor, and custom-code files.
5. Upload the complete directory and verify tensor readability, dtype, model
   config compatibility, and expected size from the source architecture.
6. Clear stale node-local blob caches before evaluation.

Do not schedule an unavailable checkpoint. Continue with other available work
and report the exact missing prerequisite.

### 5. Reconnect, repair, and launch

- For a listed matching job, query `ray job status <JOB_ID>` and tail its logs.
- If `SUCCEEDED`, package outputs, build any newly eligible report, and submit
  the next missing work row to the same node if it remains free.
- If `FAILED`, retain the traceback, repair the root cause, and rerun only that
  benchmark with the same durable output root.
- Treat blob connection timeouts as transient when built-in retries remain.
- Submit missing rows only to free Ready nodes in the dedicated pool.
- Use a unique experiment name containing the sanitized model label, checkpoint
  label or step, benchmark, and a short collision-resistant suffix.
- Before submission, print and verify the resolved candidate path, config,
  override key, `trainer.nnodes=1`, artifact root, and chosen node.
- Avoid workspace synchronization when no code changed. After a required sync,
  verify that the base-image Ray head on port 6380 and dashboard on port 9209
  are healthy before submission; do not restart Ray services casually.

### 6. Build and validate reports

Use:

```text
.github/skills/eval-2607-benchmark-report/scripts/build_2607_report.py
```

Supply durable in-house and MixLang roots. Supply OpenASR-ML from retained
metrics JSON/text or `ray:<node>:<job-id>` while Ray history is available. Use
the embedded 2607v1 baselines unless the user supplied another reference.

For a requested subset, clearly label the report as partial and do not claim
complete-suite validation. For the complete suite, reopen each workbook with
`openpyxl` and confirm:

- Exact sheets: `summary`, `inhouse_dter`, `openasr_ml`, `mixlang`.
- No digits sheets.
- Baseline and candidate numeric values are non-empty.
- Delta cells retain `=1-C<row>/B<row>` formulas.
- Model paths and config provenance are present.
- Expected row counts and percentage formats are intact.

Use `merge_2607_reports.py` once all requested checkpoint reports for one model
are individually valid. Order merged sheets by numeric checkpoint step when
steps exist; otherwise preserve the user's checkpoint order.

## Monitoring and Completion

For a long-running workload, maintain one five-minute monitor. Each poll must:

- rediscover Ready nodes only in `verl-n1-i10` through `verl-n1-i15`;
- rediscover matching Ray jobs instead of relying on saved job IDs;
- reconstruct durable completion before deciding to rerun work;
- package completed outputs before refilling nodes;
- refill every permitted free node from the current work matrix;
- build every newly eligible checkpoint report during the same poll;
- merge every newly eligible model report during the same poll;
- display matching evaluation/report work and permitted free nodes only.

Stop the monitor when every available row in the user-defined work matrix is
complete and packaged, every eligible checkpoint report passes its quality
gates, and every eligible model merge is valid. Report unavailable checkpoints
and failed rows explicitly; never redefine completion using a hard-coded model
or workbook count.

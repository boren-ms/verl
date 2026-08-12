---
name: verl-eval-workload
description: "Recover and continue the active six-model 2607 ASR benchmark batch after context loss, reconnecting to Ray jobs, preserving completed artifacts, refilling permitted nodes, and immediately building checkpoint and merged Excel reports. Use when: recover evaluation workload, resume 2607 benchmark batch, continue interrupted eval jobs, reconstruct benchmark status, or restore the six-model evaluation schedule."
argument-hint: "[status|resume|rebuild-reports]"
---

# Recover The Active 2607 Benchmark Workload

Recover the ongoing 2607 benchmark batch without duplicating successful work or
overwriting unrelated jobs. This skill is a workload-specific recovery layer
over `eval-2607-benchmark-report` and `verl-asr-run`.

The live cluster and durable blob artifacts are authoritative. The snapshot
below is only a recovery starting point; always rediscover current jobs,
checkpoint availability, and report files before acting.

## Workload Contract

Evaluate these model families at steps `30`, `50`, `100`, and `150`:

1. `remax_2607v1_bad_mixcv15_openml_verb_s200_bs256_n2_scale8_lid0_ilv`
2. `remax_2607v1_mixcv15_openml3t_s200_bs256_scale2_lid0_swtich`
3. `remax_2607v1_mix_openml_verb_s200_bs256_scale2_lid0_ilv3`
4. `remax_2607v1_mix_openml_verb_s200_bs256_lid0_ilv2`
5. `gdpo_2607v1_mix_openml_verb_s200_bs256_w10_lid0_ilv`
6. `remax_2607v1_mix_openml_verb_s200_bs256_scale2_lid0_ilv`

Every available checkpoint requires exactly these benchmarks:

| Benchmark | Config | Model override |
|---|---|---|
| In-house DTER | `long_eval_inhouse_2607_all_seg30` | `model.path` |
| OpenASR-ML | `eval_openasr_ml_verb_2607` | `actor_rollout_ref.model.path` |
| MixLang | `long_eval_mixlang_fy26q2_zh_seg_2607` | `model.path` |

Always pass `trainer.nnodes=1`. Digits benchmarks are not part of this batch.

Reference model:

```text
az://orngwus2cresco/data/speech/projects/phi-fastllm-2607/amlt-results/fast-llm-2607-qwen3-5-9b-s2-data-v3.4-sr-afteraudio/45000/qwen_hf/
```

Candidate HF checkpoints:

```text
az://orngwus2cresco/data/boren/outputs/ver_2607/<MODEL>/global_step_<STEP>/qwen_hf/
```

Evaluation artifacts:

```text
az://orngwus2cresco/data/boren/outputs/eval_2607_reports/<MODEL>_step<STEP>/<BENCHMARK>/candidate/
```

Local reports:

```text
tmp/eval_2607_reports/<MODEL>_step<STEP>.xlsx
tmp/eval_2607_reports/<MODEL>_all_steps.xlsx
```

## Hard Scheduling Rules

- Never use `verl-n1-i4` for evaluation, export, packaging, or report work.
  Inspect it only to avoid scheduling conflicts and omit it from user-facing
  evaluation tables.
- Do not stop, replace, or display nodes occupied by unrelated training jobs.
- A node is free only when it has no running Ray job and all GPUs have at most
  5% utilization and at most 5000 MiB allocated.
- Reuse complete durable outputs. Never rerun a benchmark merely because its
  original Ray job is no longer listed.
- After every completed long evaluation, package its metrics and detailed
  outputs before reassigning the node.
- Build a checkpoint workbook immediately when all three benchmark outputs are
  complete and packaged. Do not wait for other checkpoints.
- Merge a model workbook immediately when all four checkpoint workbooks are
  valid. Do not wait for other models.

## Recovery Procedure

### 1. Reconstruct cluster occupancy

List all Ready nodes:

```bash
brix pools 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep '^verl-n1-i'
```

On every Ready node, collect both signals:

```bash
brix ssh <NODE> -- 'bash -l -c "
  python /root/code/verl/ray_job.py list 2>/dev/null || true
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits
"'
```

Classify each job from its command line. Only commands using
`main_asr_eval` or `main_long_eval_asr` with `eval2607_...` experiment names
belong to this workload.

### 2. Reconstruct durable completion state

For each model, step, and benchmark, inspect the candidate artifact root.
A benchmark is complete only when:

- Long evaluation: every expected dataset has readable `details.jsonl` and
  `measures.json`.
- OpenASR-ML: all 13 expected dataset JSONL outputs exist and final W&B
  `val-aux` metrics were retained.
- The root has readable `wandb.json`, `key_metrics.json`, and
  `artifact_manifest.json`.

Long-evaluation expected counts:

- In-house DTER: 18 dataset directories.
- MixLang: one `mixlang_fy26q2` directory.

If scorer outputs are complete but metadata files are missing, package the
existing run instead of rerunning it. The manifest must record the actual node,
Ray job ID, model path, config, `trainer.nnodes=1`, detailed-output paths, and
completion timestamp.

### 3. Recover missing HF exports

Check for `model.safetensors`, config, tokenizer, processor, and custom model
code under each requested `qwen_hf/` directory. When absent:

1. Merge the verl checkpoint with
   `--match-lora-merged --lora-alpha 640 --lora-rank 320`.
2. Replace every `.base_layer.` key segment with `.`.
3. Save `model.safetensors` and remove the intermediate `model.pt`.
4. Copy base-model metadata and tokenizer/custom-code files.
5. Upload the complete directory and verify it contains 1,274 tensors and an
   approximately 18.8 GB BF16 safetensors file.
6. Clear stale node-local blob caches before evaluation.

The ilv3 step-150 export was still gated at the snapshot time because training
had reached only `global_step_130`. Recheck the blob root on every recovery.

### 4. Reconnect, repair, and refill

- For a listed workload job, query `ray job status <JOB_ID>` and tail its logs.
- If `SUCCEEDED`, package outputs, generate any newly eligible report, and
  immediately submit the next missing benchmark to that permitted node.
- If `FAILED`, retain the traceback, repair the root cause, and rerun only that
  benchmark with the same durable output root.
- Blob connection timeouts are usually transient; allow built-in retries.
- If a Ready permitted node has no Ray head, start a single-node head with:

```bash
ray start --head --port=6380 --dashboard-port=9209
```

- Workspace synchronization may restart a pod and destroy Ray state. Avoid an
  unnecessary sync when no code changed; after any required sync, verify and
  restore the Ray head before submission.

### 5. Build reports immediately

Use:

```text
.github/skills/eval-2607-benchmark-report/scripts/build_2607_report.py
```

Supply the durable in-house and MixLang roots. Supply OpenASR-ML from a retained
metrics JSON/text file or `ray:<node>:<job-id>` while that Ray history remains
available. Use the embedded 2607v1 baselines.

Validate each workbook by reopening it with `openpyxl` and confirming:

- Exact sheets: `summary`, `inhouse_dter`, `openasr_ml`, `mixlang`.
- No digits sheets.
- Baseline and candidate numeric values are non-empty.
- Delta cells retain `=1-C<row>/B<row>` formulas.
- Model paths and config provenance are present.
- Expected row counts and percentage formats are intact.

Use `merge_2607_reports.py` as soon as steps 30, 50, 100, and 150 for one model
are individually valid.

## Snapshot At 2026-08-12 22:11 UTC

Treat all job IDs as initial discovery hints, not permanent identities.

| Node | Job ID | Work item | Snapshot phase |
|---|---|---|---|
| `i1` | `raysubmit_e3FKjgyPB2uXk35v` | bad-mix@30 OpenASR-ML | running |
| `i3` | `raysubmit_GRjRFqKMd33R6zU5` | bad-mix@150 OpenASR-ML | running |
| `i6` | `raysubmit_XYAUAK4fPJ2P9hsx` | ilv3@30 MixLang | generation starting |
| `i7` | `raysubmit_bubewAv7dpy7GjZU` | mixcv15@30 OpenASR-ML | running |
| `i8` | `raysubmit_WeV1JcMMS3N7DJS1` | mixcv15@100 OpenASR-ML | running; W&B `vhejv3q2` |
| `i10` | `raysubmit_3pxtc3JcAdLPN87n` | bad-mix@100 OpenASR-ML | running |
| `i11` | `raysubmit_WPgdrrFY7DeSbEAH` | mixcv15@50 OpenASR-ML | running |
| `i12` | `raysubmit_SurXy5sS3pLLyvcs` | mixcv15@150 in-house | scoring |
| `i13` | `raysubmit_EnMQet42w8zxQZZ1` | mixcv15@150 OpenASR-ML | running |

Known completed and packaged long runs include all bad-mix MixLang steps,
bad-mix in-house steps 30/50/100/150, and mixcv15 MixLang steps
30/50/100/150 plus in-house steps 30/50/100.

Known completed OpenASR-ML:

- bad-mix@50, W&B run `9q6z1vq6`.

Validated checkpoint workbook:

```text
tmp/eval_2607_reports/remax_2607v1_bad_mixcv15_openml_verb_s200_bs256_n2_scale8_lid0_ilv_step50.xlsx
```

It contains `summary`, `inhouse_dter`, `openasr_ml`, and `mixlang`.

## Persistent Monitor

Maintain one five-minute schedule. Its prompt must:

- enforce the `i4` exclusion;
- rediscover every Ready node and replacement Ray job ID;
- package completed outputs;
- refill permitted free nodes;
- build every newly eligible checkpoint workbook during the same poll;
- merge every newly eligible model during the same poll;
- show only evaluation/report nodes and permitted free nodes in the table;
- stop only after 24 checkpoint workbooks and six merged workbooks pass all
  quality gates.

At snapshot time this behavior was installed as schedule `#7`; schedule IDs may
change, so inspect active schedules rather than assuming that ID still exists.

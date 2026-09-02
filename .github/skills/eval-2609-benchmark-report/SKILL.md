---
name: eval-2609-benchmark-report
description: "Evaluate one or more checkpoint steps of the same 2609 ASR model by running dataset evaluations in parallel across free Ready Brix nodes, then create one baseline-aware multi-step Excel report per model. Use when: evaluate 2609 benchmarks, run parallel 2609 model evaluation, build a consolidated 2609 benchmark report, compare checkpoints across steps, or compare a 2609 model against its reference model."
argument-hint: '<model-label> --checkpoint <step>=<model-path> [--checkpoint <step>=<model-path> ...] [--node <verl-node> ...] [--include-digits-tier1] [--artifact-root <az-path>] [--out <xlsx>]'
---

# 2609 Benchmark Evaluation And Consolidated Model Report

Evaluate one or more checkpoint steps of the same candidate ASR model on the standard 2609 benchmark suite on remote `verl-*` Brix nodes. Dispatch independent checkpoint/dataset evaluations concurrently across all eligible free Ready pools, then produce exactly one baseline-aware `.xlsx` workbook for the model. The workbook contains a cross-step `summary` sheet and separately named benchmark sheets for every requested checkpoint. Do not deliver or retain a separate Excel workbook for each step. Preserve each evaluation's W&B metadata, key metric logs, and detailed decoding results under a durable remote blob path.

Delegate each remote job's submission, monitoring, failure recovery, and result retrieval to the `verl-asr-run` skill. This skill owns the cross-node work queue, benchmark selection, baseline discipline, and workbook assembly.

## Inputs

| Input | Required | Meaning |
|---|---:|---|
| `model-label` | Yes | Shared model/run label such as `remax_2609`; used for the consolidated workbook name and summary. |
| `--checkpoint <step>=<model-path>` | Yes | Checkpoint label and candidate HF path, normally an `az://.../qwen_hf/` directory. Repeat in desired step order. Labels must be unique and paths must belong to the same model family. |
| `--node` | No | Restrict scheduling to this Ready `verl-n<N>-*` pool. Repeat to provide an allowlist. When omitted, use every eligible free Ready pool. Each evaluation uses its assigned pool's verified `N` as `trainer.nnodes`. |
| `--include-digits-tier1` | No | Add the optional Tier 1 digits benchmark. |
| `--out` | No | Consolidated workbook path. Default: `tmp/eval_2609_reports/<model-label>.xlsx`. |
| `--artifact-root` | No | Durable `az://.../` root for raw evaluation artifacts. Default: the common remote evaluation output root for `<model-label>`; it must not be node-local storage. |

## Benchmark Contract

Run every required config with the candidate model path as a Hydra override. Each config's committed reference model is the baseline for its sheet; do not substitute a global or fixed baseline from another report.

| Workbook sheet | Config | Metric | Reference source |
|---|---|---|---|
| `inhouse_dter` | `recipe/phimm/config/eval/long_eval_inhouse_2609_all_seg30.yaml` | micro-DTER | `model.path` in the config and its reference result directory. |
| `digits_enus` | `recipe/phimm/config/eval/eval_digits_enus_2609.yaml` | Digit CER and WER | Required by default; baseline is `actor_rollout_ref.model.path`. |
| `openasr_ml` | `recipe/phimm/config/eval/eval_openasr_ml_verb_2609.yaml` | WER / `p_err` | `actor_rollout_ref.model.path` in the config. |
| `mixlang` | `recipe/phimm/config/eval/long_eval_mixlang_fy26q2_zh_seg_2609.yaml` | DTER / TER | `model.path` in the config and its reference result directory. |
| `digits_tier1` | `recipe/phimm/config/eval/eval_digits_tier1_2609.yaml` | Digit CER and WER | Only include when `--include-digits-tier1`; baseline is `actor_rollout_ref.model.path`. |

The reference checkpoint and candidate must use the same config, data, locale, scoring backend, and segment settings. In particular, do not compare MixLang's zh-CN TER to an unrelated DTER result.

## Procedure

1. Resolve the exact committed config files and record the reference model path from each file before submitting jobs. Confirm every requested candidate path is an HF-loadable model directory, checkpoint labels are unique, and all paths belong to the same model family.
2. Build a work matrix with one row for every requested checkpoint and required candidate benchmark, including `digits_enus`. Add a matching `digits_enus` reference row because it has no embedded baseline. Add other reference rows only for optional benchmarks without an embedded baseline, such as requested `digits_tier1`, or when the caller requests a refreshed reference. Give every row a unique experiment name and output path below `<artifact-root>/<checkpoint>/<benchmark>/<candidate|reference>/`. Rows are independent and may run concurrently; never submit the same row twice.
3. Reconstruct occupancy across all Ready `verl-*` GPU pools, or only the repeated `--node` allowlist when supplied, using `verl-asr-run`'s two occupancy signals: active Ray jobs and GPU utilization/memory. A pool is free only when it has no running Ray job and all GPUs are idle. Do not stop, replace, or share a pool with unrelated work. If no eligible pool is free, keep pending rows queued and wait for a permitted pool rather than overcommitting it.
4. For each free pool, derive `EVAL_NNODES=N` from its `verl-n<N>-*` name and confirm that count against the healthy nodes reported by `ray status`. Do not copy `trainer.nnodes` from the config or assume it is `1`. If the name and live Ray count disagree, mark that pool ineligible until its health/count is resolved. Push the current workspace to every newly selected pool with `rcall-brix sync <node>` before its first submission.
5. Fill all eligible free pools in parallel, assigning at most one work-matrix row to each pool. Submit each row through `verl-asr-run` with its candidate or reference model path, unique output/experiment overrides, and `trainer.nnodes=${EVAL_NNODES}` for that assigned pool. Configure detailed decoding outputs to write or upload to the row's durable remote path, never only to node-local storage. Prefer spreading different datasets and checkpoints across pools; scheduling order must not change workbook order.
6. Monitor all active Ray jobs as one workload. When a row reaches `SUCCEEDED`, package and verify its artifacts before marking the row complete, then immediately refill that free pool with the next pending row. On failure, use `verl-asr-run` to diagnose and repair the root cause, then rerun only that failed row on any eligible free pool with the newly assigned pool's independently verified `trainer.nnodes`. Do not start workbook creation until every required row is complete.
7. Use the embedded 2609v1 baseline for in-house DTER, OpenASR-ML, and MixLang unless an explicit matching baseline source is supplied. Candidate and reference rows for the same benchmark may run on different pools, but both must use the same committed config, data, locale, scoring backend, and segment settings.
8. Before releasing a node or creating the workbook, retain the raw information for every candidate run and every explicitly executed reference run under `<artifact-root>/<checkpoint>/<benchmark>/<candidate|reference>/`:
   - `wandb.json`: W&B entity, project, run ID, run name, and run URL. If W&B is disabled or unavailable, record the reason explicitly instead of inventing a run identity.
   - `key_metrics.log` or `key_metrics.json`: the unmodified evaluation metric records used to build the report, including the final `val-aux/...` values or equivalent scorer output.
   - Detailed decoding results: upload all per-utterance hypothesis/reference outputs, such as `result_details*.jsonl`, together with scorer outputs such as `measures.json`. Preserve the dataset subdirectory structure when multiple datasets are evaluated.
   - `artifact_manifest.json`: benchmark, role, config path and revision, model path, Ray node and job ID, verified `trainer.nnodes`, W&B metadata path, key-metrics path, detailed-decoding remote paths, and completion timestamp.

   Confirm every recorded `az://` object or prefix exists and is readable. A W&B URL alone is not a substitute for the raw key-metrics log, and a Ray job log or node-local path is not a durable detailed-decoding result.
9. Extract metrics from the final candidate and reference outputs:
   - `inhouse_dter`: calculate micro-DTER as $\sum edits / \sum reference\ tokens$ per corpus; do not use segment-macro DTER. Compute `overall avg` as the arithmetic mean of the displayed corpus micro-DTER values so it matches the embedded baseline convention; do not pool counts across corpora.
   - Required `digits_enus` and optional `digits_tier1`: preserve both Digit CER and WER, with every evaluated dataset as a row.
   - `openasr_ml`: use final `p_err`/WER per dataset, language averages, and an overall average.
   - `mixlang`: use the final reference-compatible DTER/TER from `measures.json`, retaining the configured zh-CN scoring context.
10. Write one artifact sidecar per checkpoint under `tmp/eval_2609_reports/<model-label>/`, for example `step560.artifacts.json`. Sidecars are durable provenance inputs to the consolidated report; never place them directly under `tmp/eval_2609_reports/`. Then use [scripts/build_2609_report.py](./scripts/build_2609_report.py) to create a temporary workbook for each checkpoint under `tmp/eval_2609_reports/<model-label>/.staging/`. These workbooks are implementation artifacts only: do not present them as reports, and remove the staging directory after the consolidated workbook passes all quality gates. Each temporary benchmark sheet holds a reference column (`A`), a candidate column (`B`), and a delta column `A->B` computed exactly like `inhouse-dter-report`:

   $$\mathrm{delta}=1-\frac{\mathrm{candidate\ error}}{\mathrm{reference\ error}}$$

   Every group and overall `avg` value cell uses an Excel `AVERAGE(...)` formula over the displayed dataset cells for that metric; no fixed, weighted, pooled-count, or stored-numeric override is allowed. Overall formulas reference only dataset ranges and exclude intermediate group-average rows. The delta column is labelled per metric — `TERR` for DTER/TER, `WERR` for WER, `CERR` for CER. Positive delta means the candidate improved. Data and average-row delta cells use the Excel formula `=1-C{row}/B{row}`. All error and delta cells use percentage formatting, and the delta column carries a red→white→green color scale centered at 0.
11. Build the single deliverable workbook with [scripts/merge_2609_reports.py](./scripts/merge_2609_reports.py), even when only one checkpoint is requested. Pass each staged workbook as `--report <checkpoint-label>=<staged-xlsx>` in requested order and write the result to `--out` or `tmp/eval_2609_reports/<model-label>.xlsx`. The consolidated workbook contains:
   - `summary`: one row per checkpoint, benchmark, and metric, with the checkpoint label, baseline, candidate, delta, config, model paths, result sources, artifact sidecars, and source workbook.
   - `<checkpoint>_<benchmark>`: a faithful copy of each benchmark sheet, retaining formulas, percentage formatting, conditional formatting, widths, and provenance values. Sheet names are sanitized for Excel, limited to 31 characters, and given a stable hash suffix only when truncation causes a collision.

   Do not copy staged per-checkpoint `summary` sheets verbatim; consolidate them into the single cross-step `summary`. Reject the build if checkpoint labels repeat or if checkpoints differ in included benchmark/metric/config schema. Do not silently combine checkpoints from different model families, baseline revisions, optional benchmark sets, or config revisions.
12. The consolidated `summary` uses the same red→white→green delta color scale as benchmark sheets and contains one untitled clustered-column, delta-only chart for all included benchmark metrics and checkpoint steps. Dataset names are the x-axis categories, and checkpoint steps are separate column series clustered within each dataset group. Assign stable distinct colors to checkpoint series. Show a top checkpoint legend and percentage-only data labels without legend keys. Disable negative inversion. Add 15% of the observed delta span below and above the y-axis range while keeping zero visible. Use column overlap `-40%` and gap width `260`. Keep baseline and candidate in the table but not in chart series or labels. Anchor the chart in column `A` below the visible summary table, store chart source data in hidden helper columns on the same sheet, and disable `plotVisOnly`. Always include `digits_enus` rows and categories; include `digits_tier1` only when explicitly requested.
13. Preserve baseline and raw-artifact provenance in the consolidated workbook and checkpoint sidecars: config path, reference model path, candidate path, metric definition, W&B run URL/ID, key-metrics remote path, detailed-decoding remote paths, and artifact-manifest remote path. After validating the final workbook, delete all staged `.xlsx` files. The only retained Excel deliverable for the model is the consolidated workbook.

### Build checkpoint staging inputs

Each benchmark takes a candidate source and accepts a baseline source. The baseline source is required for `digits_enus` and any included `digits_tier1` because neither has an embedded baseline; it is optional for benchmarks with embedded baselines. A source is auto-detected as one of: an `az://.../` root or local directory holding `<slug>/measures.json` (used for the DTER benchmarks `inhouse_dter` and `mixlang`); a `*.json` metrics file (`{ds: value}` or `{ds: {metric: value}}`); a text log with `val-aux/<ds>/<metric>/mean@1` lines; or `ray:<node>:<job_id>` to pull `ray job logs`.

**Collecting the baseline result.** The baseline (column `A` in the report schema) uses the current embedded 2609v1 values when `--<bench>-baseline` is omitted:

| Benchmark | Embedded label | Source column | Embedded overall |
|---|---|---:|---:|
| `inhouse_dter` | `2609v1,LID` | C | 17.77% |
| `openasr_ml` | `2609v1` | B | 2.92% |
| `mixlang` | `2609v1` | B | 21.24% |
| `digits_enus` / `digits_tier1` | none | — | Supply `ray:<node>:<job_id>`, log, or JSON via `--<bench>-baseline`. |

An explicit `--<bench>-baseline` source overrides the embedded values. For required `digits_enus` and optional `digits_tier1`, collect the baseline by running the same config with the reference model through `verl-asr-run` and passing its Ray job as the baseline source (or a captured log/JSON). Collect candidate results with the candidate model path.

```bash
/home/boren/.virtualenvs/openai/bin/python \
  .github/skills/eval-2609-benchmark-report/scripts/build_2609_report.py \
   --label "step560" \
   --candidate-model-path az://orngwus2cresco/.../global_step_560/qwen_hf/ \
   --artifacts-sidecar-local tmp/eval_2609_reports/remax_2609/step560.artifacts.json \
  --inhouse-dter  az://orngwus2cresco/.../cand/inhouse_2605_all_seg30/ \
   --digits-enus       ray:verl-n1-i1:raysubmit_DIGITS_CAND \
   --digits-enus-baseline ray:verl-n1-i2:raysubmit_DIGITS_BASE \
  --mixlang           az://orngwus2cresco/.../cand/long_audio_mixlang_fy26q2_zh_seg/ \
  --openasr-ml           ray:verl-n1-i0:raysubmit_CAND2 \
   --out tmp/eval_2609_reports/remax_2609/.staging/step560.xlsx
```

The `inhouse_dter`, `openasr_ml`, and `mixlang` baselines come from the embedded table above, so their `--*-baseline` flags may be omitted. Always add `--digits-enus <cand>` with `--digits-enus-baseline <base>`. Add `--digits-tier1 <cand>` with `--digits-tier1-baseline <base>` only when `--include-digits-tier1` was requested. Only benchmarks whose candidate source is supplied get a sheet.

### Build the single model report

After all staged checkpoint inputs pass validation, use the merge helper to create the only Excel deliverable. The repeated `--report` arguments define checkpoint and sheet order. This command also applies to a single checkpoint.

```bash
/home/boren/.virtualenvs/openai/bin/python \
   .github/skills/eval-2609-benchmark-report/scripts/merge_2609_reports.py \
   --model-label "remax_2609" \
   --report "step280=tmp/eval_2609_reports/remax_2609/.staging/step280.xlsx" \
   --report "step560=tmp/eval_2609_reports/remax_2609/.staging/step560.xlsx" \
   --report "step840=tmp/eval_2609_reports/remax_2609/.staging/step840.xlsx" \
   --out tmp/eval_2609_reports/remax_2609.xlsx
```

Keep each checkpoint's benchmark sheet self-contained so its baseline, formulas, and provenance remain auditable. Use the consolidated `summary` sheet for cross-step sorting and comparison. Once the final workbook passes the quality gates, remove `tmp/eval_2609_reports/remax_2609/.staging/`.

### Repair existing workbooks

Use [scripts/repair_2609_avg_formulas.py](./scripts/repair_2609_avg_formulas.py) to update an existing consolidated or legacy 2609 workbook in place. The utility replaces stored numeric group and overall averages with arithmetic Excel `AVERAGE(...)` formulas, refreshes summary aggregates from dataset rows, and enables full automatic recalculation.

```bash
/home/boren/.virtualenvs/openai/bin/python \
   .github/skills/eval-2609-benchmark-report/scripts/repair_2609_avg_formulas.py \
   tmp/eval_2609_reports/remax_2609.xlsx
```

## Quality Gates

Before delivering the workbook, verify all of the following:

- Every required benchmark has a successful candidate job and either the matching embedded 2609v1 baseline or an explicitly supplied matching reference result.
- Every submitted candidate and reference evaluation explicitly used `trainer.nnodes` equal to the verified node count of its running `verl-n<N>-*` pool.
- Every eligible free pool was filled while pending work remained, with at most one active evaluation row per pool and no duplicate work-matrix submissions.
- Every selected pool was verified free using both Ray-job and GPU occupancy signals, and no unrelated job was displaced or colocated.
- `digits_enus` is present for every checkpoint, and `digits_tier1` is absent unless explicitly requested.
- Baselines are the embedded 2609v1 values or verified matching reference outputs supplied explicitly.
- Candidate and baseline use identical data/locale/scoring parameters for each sheet.
- Every executed evaluation has a readable remote `artifact_manifest.json`, raw key-metrics log, and detailed decoding result path; none points only to node-local storage.
- Every executed evaluation records its W&B run ID and URL, or an explicit reason W&B was unavailable, and the W&B identity agrees with the experiment represented in the workbook.
- In-house corpus values are micro-DTER and its `overall avg` is the arithmetic corpus average; OpenASR-ML includes language and overall averages; any included digits sheet contains CER and WER; MixLang records its zh-CN scoring backend.
- Every sheet has non-empty baseline and candidate values, percentage formatting, and a correctly signed delta (`1 - B/A`).
- Every summary delta column has a red→white→green conditional color scale centered at 0, matching the benchmark sheets.
- Every summary contains one delta-only column chart with one non-empty, percentage-labelled bar per checkpoint/benchmark category and no baseline/candidate chart series.
- Every cell style references an existing fill, font, border, alignment, and number-format record; in particular, no `fillId` may be greater than or equal to the declared fills count in `xl/styles.xml`.
- Exactly one model-level Excel workbook is retained; no per-checkpoint `.xlsx` files remain after validation.
- The consolidated workbook contains exactly one `summary` sheet plus one prefixed benchmark sheet per checkpoint and included benchmark; it contains no copied per-checkpoint summary sheets.
- Checkpoint labels are unique, benchmark/metric/config/baseline schemas and optional Tier 1 digits coverage are identical, and candidate paths identify steps of the same model family.
- The consolidated `summary` has a non-empty row for every checkpoint/benchmark/metric combination and retains both local and remote artifact sidecar paths.
- Every local checkpoint JSON sidecar is inside `tmp/eval_2609_reports/<model-label>/`; no checkpoint JSON file is written directly under `tmp/eval_2609_reports/`.
- Reopen the consolidated workbook with `openpyxl`, confirm its expected sheet names and row counts, and verify at least one copied delta cell still contains an Excel formula rather than a cached value.

## Existing Helpers

Use existing report skills as parsers/formatting references only after checking their embedded baseline matches the current config reference:

- `inhouse-dter-report` for the all-segment micro-DTER schema.
- `digits-report` for digits CER/WER presentation.
- `openasr-report` for OpenASR-ML dataset and language-average presentation.

Those helpers may contain fixed historical baselines. Their baseline must be overridden or the metrics assembled directly when it differs from the reference model declared in the 2609 config.
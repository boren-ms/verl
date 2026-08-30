---
name: eval-2607-benchmark-report
description: "Evaluate one or more checkpoint steps of the same 2607 ASR model by running dataset evaluations in parallel across free Ready Brix nodes, then create one baseline-aware multi-step Excel report per model. Use when: evaluate 2607 benchmarks, run parallel 2607 model evaluation, build a consolidated 2607 benchmark report, compare checkpoints across steps, or compare a 2607 model against its reference model."
argument-hint: '<model-label> --checkpoint <step>=<model-path> [--checkpoint <step>=<model-path> ...] [--node <verl-node> ...] [--include-digits-tier1] [--artifact-root <az-path>] [--out <xlsx>]'
---

# 2607 Benchmark Evaluation And Consolidated Model Report

Evaluate one or more checkpoints of the same candidate ASR model on the standard 2607 suite. Dispatch independent checkpoint/dataset evaluations concurrently across eligible free Ready `verl-*` Brix pools, then produce exactly one baseline-aware `.xlsx` workbook. The workbook contains one cross-step `summary` and separately named benchmark sheets for every checkpoint. Never deliver or retain one workbook per step.

Delegate each remote job's submission, monitoring, recovery, and result retrieval to the `verl-asr-run` skill. This skill owns cross-node scheduling, benchmark selection, baseline discipline, artifact retention, and workbook assembly.

## Inputs

| Input | Required | Meaning |
|---|---:|---|
| `model-label` | Yes | Shared run label used for the workbook and summary. |
| `--checkpoint <step>=<model-path>` | Yes | Unique checkpoint label and HF-loadable model directory. Repeat in desired order; all paths must belong to one model family. |
| `--node` | No | Repeatable Ready-pool allowlist. When omitted, use every eligible free Ready pool. Derive `trainer.nnodes` from each assigned `verl-n<N>-*` pool. |
| `--include-digits-tier1` | No | Add optional Tier 1 digits. |
| `--artifact-root` | No | Durable remote root for raw artifacts; never node-local storage. |
| `--out` | No | Default: `tmp/eval_2607_reports/<model-label>.xlsx`. |

## Benchmark Contract

This repository uses its validation-only `long_eval_asr` stack. Run every config with the candidate model path as a Hydra `model.path` override. The committed reference model in each config is the matching baseline; do not substitute a global baseline.

| Workbook sheet | Config | Metric | Baseline |
|---|---|---|---|
| `inhouse_dter` | `recipe/phimm/config/eval/long_eval_inhouse_2607v1_all_seg30.yaml` | micro-DTER | Config `model.path` and matching reference results. |
| `digits_enus` | `recipe/phimm/config/eval/eval_digits_enus_2607v1.yaml` | Digit CER and WER | Required; run the config reference model because no baseline is embedded. |
| `openasr_ml` | `recipe/phimm/config/eval/eval_openasr_ml_verb_2607v1.yaml` | WER / `p_err` | Config `model.path` and matching reference results. |
| `mixlang` | `recipe/phimm/config/eval/long_eval_mixlang_fy26q2_zh_seg_2607v1.yaml` | DTER / TER | Config `model.path`, zh-CN scorer, and matching reference results. |
| `digits_tier1` | `recipe/phimm/config/eval/eval_digits_tier1_2607v1.yaml` | Digit CER and WER | Optional; run the same config with its reference model. |

Candidate and reference must use the same committed config, data, locale, scorer, and segmentation. Do not compare MixLang's zh-CN TER with an unrelated DTER result.

## Procedure

1. Resolve each committed config and record its reference `model.path`. Validate all candidate directories, unique checkpoint labels, and common model family before submission.
2. Build one work-matrix row per checkpoint and required candidate benchmark. Always include `digits_enus`. Add matching reference rows for digits and for any benchmark whose embedded baseline is not being used. Give every row a unique experiment name and durable output path below `<artifact-root>/<checkpoint>/<benchmark>/<candidate|reference>/`; never submit a row twice.
3. Determine occupancy using both active Ray jobs and GPU utilization/memory. A pool is free only when it has no running Ray job and every GPU is idle. Never stop, replace, or colocate unrelated work. Keep rows queued when no eligible pool is free.
4. For every selected `verl-n<N>-*` pool, verify the live healthy Ray node count equals `N`; otherwise mark it ineligible. Set `trainer.nnodes=N`. Before the first submission to a pool, sync this workspace with `rcall-brix sync <node>`.
5. Fill all eligible pools in parallel, at most one matrix row per pool. Submit through `verl-asr-run` with the assigned model, unique output/experiment overrides, and verified `trainer.nnodes`. Detailed decoding must go to durable remote storage.
6. Monitor active jobs as one workload. On success, verify and package artifacts, then refill the pool. On failure, diagnose through `verl-asr-run`, fix the root cause, and rerun only that row. Do not build reports until all required rows complete.
7. Retain for every executed candidate/reference run:
   - `wandb.json` with entity, project, run ID/name/URL, or an explicit unavailability reason.
   - Unmodified `key_metrics.log` or `key_metrics.json` containing the report inputs.
   - Detailed hypotheses/references such as `result_details*.jsonl`, plus scorer outputs such as `measures.json`.
   - `artifact_manifest.json` containing role, benchmark, config path/revision, model path, Ray pool/job ID, verified node count, artifact locations, and completion time.
8. Confirm every recorded remote object exists and is readable. A W&B URL, Ray log, or node-local path is not a substitute for durable raw results.
9. Extract metrics consistently:
   - In-house: micro-DTER per corpus is $\sum edits / \sum reference\ tokens$; `overall avg` is the arithmetic mean of displayed corpus values.
   - Digits: preserve both CER and WER for every dataset.
   - OpenASR-ML: preserve dataset WER/`p_err`, language averages, and overall average.
   - MixLang: preserve final reference-compatible DTER/TER and zh-CN scoring context.
10. Write one sidecar per checkpoint inside `tmp/eval_2607_reports/<model-label>/`, then build temporary checkpoint workbooks under `tmp/eval_2607_reports/<model-label>/.staging/` with `scripts/build_2607_report.py`.
11. Every benchmark sheet has reference column `A`, candidate column `B`, and metric-specific delta (`TERR`, `WERR`, or `CERR`) computed as:

   $$\mathrm{delta}=1-\frac{\mathrm{candidate\ error}}{\mathrm{reference\ error}}$$

   Use Excel `AVERAGE(...)` formulas for all group and overall averages over displayed dataset cells. Overall formulas exclude intermediate averages. Use `=1-C{row}/B{row}` for data and average deltas. Format errors/deltas as percentages and apply a red→white→green scale centered at zero.
12. Merge staged workbooks with `scripts/merge_2607_reports.py`, even for one checkpoint. Pass repeated `--report <label>=<path>` arguments in requested order. Reject duplicate labels or mismatched benchmark, metric, config, baseline, optional-set, or model-family schemas.
13. The consolidated workbook contains:
   - `summary`: checkpoint, benchmark, metric, baseline, candidate, delta, config, model paths, result sources, artifact sidecars, and source workbook.
   - `<checkpoint>_<benchmark>`: faithful benchmark-sheet copies with formulas, formatting, widths, conditional formatting, and provenance.
14. The summary has one untitled clustered-column delta-only chart. Datasets are categories and checkpoint steps are series. Use stable colors, top legend, percentage data labels, no negative inversion, zero-visible y-axis with 15% padding, overlap `-40%`, gap width `260`, hidden same-sheet helper data, and `plotVisOnly=false`.
15. Validate the final workbook, remove `.staging/`, and retain exactly one Excel deliverable.

## Build Staging Inputs

Sources auto-detect as an `az://` or local measures tree, JSON metrics file, text log containing `val-aux/...` records, or `ray:<node>:<job_id>`. Explicit baseline sources override embedded values. `digits_enus` and included `digits_tier1` always require an explicit matching baseline source.

Embedded report values are provided for in-house (`2607v1,LID`, overall 17.77%), OpenASR-ML (`2607v1`, overall 2.92%), and MixLang (`2607v1`, overall 21.24%). Before using them, verify they match the committed config reference revision; otherwise run and supply the reference explicitly.

Run the helper from the repository environment, which includes `openpyxl`:

```bash
uv run python .github/skills/eval-2607-benchmark-report/scripts/build_2607_report.py \
  --label step560 \
  --candidate-model-path az://orngwus2cresco/.../global_step_560/qwen_hf/ \
  --artifacts-sidecar-local tmp/eval_2607_reports/remax_2607/step560.artifacts.json \
  --inhouse-dter az://orngwus2cresco/.../candidate/inhouse_2605_all_seg30/ \
  --digits-enus ray:verl-n1-i1:raysubmit_DIGITS_CAND \
  --digits-enus-baseline ray:verl-n1-i2:raysubmit_DIGITS_BASE \
  --openasr-ml ray:verl-n1-i0:raysubmit_OPENASR_CAND \
  --mixlang az://orngwus2cresco/.../candidate/long_audio_mixlang_fy26q2_zh_seg/ \
  --out tmp/eval_2607_reports/remax_2607/.staging/step560.xlsx
```

## Build The Single Model Report

```bash
uv run python .github/skills/eval-2607-benchmark-report/scripts/merge_2607_reports.py \
  --model-label remax_2607 \
  --report step280=tmp/eval_2607_reports/remax_2607/.staging/step280.xlsx \
  --report step560=tmp/eval_2607_reports/remax_2607/.staging/step560.xlsx \
  --out tmp/eval_2607_reports/remax_2607.xlsx
```

## Repair Existing Workbooks

Use `scripts/repair_2607_avg_formulas.py` to replace stored average numerics with arithmetic Excel formulas, refresh summaries, rebuild charts, repair known fill records, and enable full recalculation:

```bash
uv run python .github/skills/eval-2607-benchmark-report/scripts/repair_2607_avg_formulas.py \
  tmp/eval_2607_reports/remax_2607.xlsx
```

## Quality Gates

- Every required candidate job succeeded and has the exact matching embedded or executed reference.
- Every run used the assigned pool's independently verified `trainer.nnodes`.
- Pending rows filled all eligible free pools, with no duplicate rows, displacement, or colocation.
- `digits_enus` exists for every checkpoint; `digits_tier1` appears only when requested.
- Candidate/reference config, data, locale, scorer, and segmentation match.
- Every run has readable durable manifests, raw metrics, decoding details, and W&B identity/unavailability reason.
- In-house uses corpus micro-DTER and arithmetic corpus average; OpenASR-ML has dataset/language/overall values; digits has CER and WER; MixLang records zh-CN scoring.
- Every sheet has non-empty baseline/candidate percentages and correctly signed deltas.
- Every summary delta has the zero-centered red→white→green scale and the required delta-only chart.
- No cell style references a nonexistent fill/font/border/alignment/number format.
- The final workbook has exactly one `summary` and one prefixed sheet per checkpoint/benchmark; no staged or per-checkpoint workbook remains.
- Checkpoint sidecars live under `tmp/eval_2607_reports/<model-label>/`, never directly under `tmp/eval_2607_reports/`.
- Reopen with `openpyxl`, verify sheet names/row counts, and confirm copied delta and average cells retain formulas.

## Existing Helpers

Use `inhouse-dter-report`, `digits-report`, and `openasr-report` only as parsing/formatting references after verifying their historical baselines against the current config. Override mismatched baselines or assemble metrics directly.

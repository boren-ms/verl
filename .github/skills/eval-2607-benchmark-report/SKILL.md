---
name: eval-2607-benchmark-report
description: "Evaluate a 2607 ASR model on all standard benchmarks remotely and create one baseline-aware Excel workbook. Use when: evaluate 2607 benchmarks, run 2607 model evaluation, build 2607 benchmark Excel report, evaluate inhouse DTER plus digits plus OpenASR-ML plus MixLang, or compare a 2607 checkpoint against its reference model."
argument-hint: '<model-label> <model-path> [--node <verl-node>] [--include-digits-tier1] [--out <xlsx>]'
---

# 2607 Benchmark Evaluation And Excel Report

Evaluate one candidate ASR model on the standard 2607 benchmark suite on a remote `verl-*` Brix node. Produce one `.xlsx` workbook with one benchmark sheet per evaluation, containing the candidate, the config-defined reference model, and relative improvement versus that reference.

Delegate remote submission, monitoring, failure recovery, and result retrieval to the `verl-asr-run` skill. This skill owns benchmark selection, baseline discipline, and workbook assembly.

## Inputs

| Input | Required | Meaning |
|---|---:|---|
| `model-label` | Yes | Short workbook column label, such as `remax_2607@step560`. |
| `model-path` | Yes | Candidate HF checkpoint path, normally an `az://.../qwen_hf/` directory. |
| `--node` | No | Ready remote `verl-*` node. Let `verl-asr-run` select one when omitted. |
| `--include-digits-tier1` | No | Add the optional Tier 1 digits benchmark. |
| `--out` | No | Final workbook path. Default: `tmp/eval_2607_reports/<model-label>.xlsx`. |

## Benchmark Contract

Run every required config with the candidate model path as a Hydra override. Each config's committed reference model is the baseline for its sheet; do not substitute a global or fixed baseline from another report.

| Workbook sheet | Config | Metric | Reference source |
|---|---|---|---|
| `inhouse_dter` | `recipe/phimm/config/eval/long_eval_inhouse_2607_all_seg30.yaml` | micro-DTER | `model.path` in the config and its reference result directory. |
| `digits_enus` | `recipe/phimm/config/eval/eval_digits_enus_2607.yaml` | Digit CER and WER | `actor_rollout_ref.model.path` in the config. |
| `openasr_ml` | `recipe/phimm/config/eval/eval_openasr_ml_verb_2607.yaml` | WER / `p_err` | `actor_rollout_ref.model.path` in the config. |
| `mixlang` | `recipe/phimm/config/eval/long_eval_mixlang_fy26q2_zh_seg_2607.yaml` | DTER / TER | `model.path` in the config and its reference result directory. |
| `digits_tier1` | `recipe/phimm/config/eval/eval_digits_tier1_2607.yaml` | Digit CER and WER | Only include when `--include-digits-tier1`; baseline is `actor_rollout_ref.model.path`. |

The reference checkpoint and candidate must use the same config, data, locale, scoring backend, and segment settings. In particular, do not compare MixLang's zh-CN TER to an unrelated DTER result.

## Procedure

1. Resolve the exact committed config files and record the reference model path from each file before submitting jobs. Confirm the candidate path is an HF-loadable model directory.
2. Select a free Ready `verl-*` node using `verl-asr-run`'s occupancy checks. Push the current workspace with `rcall-brix sync <node>` before submitting work.
3. For each required config, run a candidate evaluation remotely through `verl-asr-run`, using a unique experiment name and output path. Override only the candidate model path and output/experiment identifiers.
4. Run the same config for its reference only when canonical results for that exact reference path, scoring contract, and config revision are not already available. Never label a hard-coded baseline from another reporting skill as the config reference without verifying that it is the same checkpoint.
5. Monitor every Ray job to `SUCCEEDED`. On failure, use `verl-asr-run` to diagnose and repair the root cause, then rerun only the failed benchmark. Do not start workbook creation from incomplete or failed output.
6. Extract metrics from the final candidate and reference outputs:
   - `inhouse_dter`: calculate micro-DTER as $\sum edits / \sum reference\ tokens$ per corpus; do not use segment-macro DTER.
   - `digits_enus` and `digits_tier1`: preserve both Digit CER and WER, with every evaluated dataset as a row.
   - `openasr_ml`: use final `p_err`/WER per dataset, language averages, and an overall average.
   - `mixlang`: use the final reference-compatible DTER/TER from `measures.json`, retaining the configured zh-CN scoring context.
7. Build the workbook with [scripts/build_2607_report.py](./scripts/build_2607_report.py). It reuses existing eval outputs and produces sheets in this order: `summary`, `inhouse_dter`, `digits_enus`, `openasr_ml`, `mixlang` (its own separate sheet), then optional `digits_tier1`. Each benchmark sheet holds a reference column (`A`), a candidate column (`B`), and a delta column `A->B` computed exactly like `inhouse-dter-report`:

   $$\mathrm{delta}=1-\frac{\mathrm{candidate\ error}}{\mathrm{reference\ error}}$$

   The delta column is labelled per metric — `TERR` for DTER/TER, `WERR` for WER, `CERR` for CER. Positive delta means the candidate improved. Data-row delta cells use the Excel formula `=1-C{row}/B{row}`; per-group and overall averages store the computed numeric delta. All error and delta cells use percentage formatting, and the delta column carries a red→white→green color scale centered at 0.
8. The script also emits the `summary` sheet with one row per benchmark metric: the aggregate baseline and candidate values, the overall delta, and the source config. Tier 1 rows appear only when `--digits-tier1` was supplied.
9. Preserve baseline provenance: keep the config path, reference model path, candidate path, metric definition, and result locations with the workbook. Do not silently merge outputs produced by differing config revisions.

### Consolidated report script

Each benchmark takes a candidate source and an optional baseline source. A source is auto-detected as one of: an `az://.../` root or local directory holding `<slug>/measures.json` (used for the DTER benchmarks `inhouse_dter` and `mixlang`); a `*.json` metrics file (`{ds: value}` or `{ds: {metric: value}}`); a text log with `val-aux/<ds>/<metric>/mean@1` lines; or `ray:<node>:<job_id>` to pull `ray job logs`.

**Collecting the baseline result.** The baseline (column `A`) is the config reference model's own eval output. When `--<bench>-baseline` is omitted the script auto-collects it from the config `output_path` for the long-audio DTER benchmarks:

| Benchmark | Default baseline source (reference `output_path`) | Auto-collect |
|---|---|---|
| `inhouse_dter` | `az://orngwus2cresco/data/boren/data/verl/eval/qwen_2607_45000/inhouse_2605_all_seg30/` | Yes (measures tree). Embedded `2607 vllm` constants are the fallback. |
| `mixlang` | `az://orngwus2cresco/data/boren/data/verl/eval/qwen_2607_45000/long_audio_mixlang_fy26q2_zh_seg/` | Yes when the reference eval has been run to that path. |
| `digits_enus` / `digits_tier1` / `openasr_ml` | none — `eval_*` jobs are `val_only` and do not persist `measures.json` | No. Supply the reference `ray:<node>:<job_id>`, log, or JSON via `--<bench>-baseline`. |

For the eval-only benchmarks, collect the baseline by running the same config with the reference model through `verl-asr-run` and passing its Ray job as the baseline source (or a captured log/JSON). Collect the candidate result the same way with the candidate model path.

```bash
/home/boren/.virtualenvs/openai/bin/python \
  .github/skills/eval-2607-benchmark-report/scripts/build_2607_report.py \
  --label "cand@step560" --baseline-label "qwen_2607_45000" \
  --inhouse-dter  az://orngwus2cresco/.../cand/inhouse_2605_all_seg30/ \
  --mixlang           az://orngwus2cresco/.../cand/long_audio_mixlang_fy26q2_zh_seg/ \
  --digits-enus          ray:verl-n1-i0:raysubmit_CAND \
  --digits-enus-baseline ray:verl-n1-i0:raysubmit_BASE \
  --openasr-ml           ray:verl-n1-i0:raysubmit_CAND2 \
  --openasr-ml-baseline  ray:verl-n1-i0:raysubmit_BASE2 \
  --out tmp/eval_2607_reports/cand_step560.xlsx
```

The `inhouse_dter` and `mixlang` baselines are auto-collected from the table above, so their `--*-baseline` flags may be omitted. Add `--digits-tier1 <cand>` (and `--digits-tier1-baseline <base>`) only when the optional Tier 1 benchmark is requested. Only benchmarks whose candidate source is supplied get a sheet.

## Quality Gates

Before delivering the workbook, verify all of the following:

- Every required benchmark has a successful candidate job and a matching reference result.
- Tier 1 is absent unless explicitly requested.
- Baseline paths come from the evaluation configs or verified matching canonical reference outputs.
- Candidate and baseline use identical data/locale/scoring parameters for each sheet.
- In-house values are micro-DTER; OpenASR-ML includes language and overall averages; digits sheets include CER and WER; MixLang records its zh-CN scoring backend.
- Every sheet has non-empty baseline and candidate values, percentage formatting, and a correctly signed delta (`1 - B/A`).
- Workbook opens successfully and contains exactly the expected sheet set, with `mixlang` on its own separate sheet.

## Existing Helpers

Use existing report skills as parsers/formatting references only after checking their embedded baseline matches the current config reference:

- `inhouse-dter-report` for the all-segment micro-DTER schema.
- `digits-report` for digits CER/WER presentation.
- `openasr-report` for OpenASR-ML dataset and language-average presentation.

Those helpers may contain fixed historical baselines. Their baseline must be overridden or the metrics assembled directly when it differs from the reference model declared in the 2607 config.
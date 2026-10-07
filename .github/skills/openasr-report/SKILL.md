---
name: openasr-report
description: Build an OpenASR + OpenASR_ML xlsx comparison report for a trained verl ASR model, with the new model inserted as the next column (B, C, D, ...) next to the 2609r2 baseline (column A) and per-model WERR columns. Use when summarizing eval_openasr/eval_openasr_ml results from a Ray job, blob checkpoint dir, or local metrics file into the standardized Excel report with per-language and overall averages, comparing a new checkpoint against the baseline, or extending an existing xlsx report with another model column.
argument-hint: '<model-label> [--from-ray <node> <job-id> [<job-id> ...]] | [--from-text <file>] | [--metrics <json>] [--model-info <json>] [--dataset-results <json>] [--extend-xlsx <xlsx>] [--out <xlsx>]'
---

# OpenASR Comparison Report (xlsx)

Generate an Excel report (`.xlsx`) that compares a new model column against the `2609r2` baseline (and any other model columns already present in a prior report). The updated layout includes cleaned English datasets, Monsoon English/Hindi, and Dutch.

## When to Use

- A verl ASR eval job (`eval_openasr` and/or `eval_openasr_ml`) has just produced per-dataset WERs and you need the canonical Excel report.
- Adding another checkpoint as a new column (B, C, D, ...) to an existing xlsx report.
- Reporting on a `global_step_<N>` checkpoint after training.

## Layout

Three sheets: `openasr`, `model_info`, and `dataset_results`. The `openasr` metrics layout remains:

- **Row 2** `Header`: `<baseline-label>`, `<model-label-1>`, `<model-label-2>`, ..., `WERR` (one WERR header per non-baseline model column).
- **Row 3** `Column`: `A`, `B`, `C`, ..., `A->B`, `A->C`, ...
- **Rows 4–11**: OpenASR datasets — `ami_clean`, `earnings22-cleaned-aa-chunked`, `gigaspeech_clean`, `librispeech-clean`, `librispeech-other`, `monsoon_en_in`, `spgispeech`, `voxpopuli-cleaned-aa`.
- **Row 12** `avg`: numeric mean across the 8 OpenASR datasets per column.
- **Row 13**: repeated `Column` header for the ML section.
- **Rows 14–37**: ML datasets grouped by language with per-language `<lang> avg` rows interleaved:
  - de: `de_fleurs`, `de_mcv`
  - es: `es_fleurs`, `es_mcv`, `es_mls`
  - fr: `fr_fleurs`, `fr_mcv`, `fr_mls`
  - hi: `monsoon_hi_in`
  - it: `it_fleurs`, `it_mcv`, `it_mls`
  - nl: `nl_fleurs`, `nl_mcv`, `nl_mls`
  - pt: `pt_fleurs`, `pt_mls`
- **Row 38** `ml avg`: numeric unweighted mean of the seven unrounded language means per column. With partial coverage, include only languages with available datasets.
- **WERR columns**: compute `1 - model / baseline` for every available pair with a positive baseline. Per-dataset cells use Excel formulas; average rows store numeric WERR. Positive values mean improvement; negative values mean regression. Do not remove deltas because baseline checkpoint/scoring provenance is incomplete.

Formatting:
- All data columns (everything except column A) center-aligned.
- Header / `Column` rows: light blue.
- Per-language `<lang> avg` rows: light yellow.
- `avg` and `ml avg` rows: light green.
- WERR columns carry a 3-color scale rule (red ← 0 → green) with midpoint fixed at 0.
- Numeric cells formatted as `0.00%`.
- Keep `openasr` limited to the canonical table. Do not add notes, disclaimers, cell comments, coverage banners, or footers. Put the requested structured model information and result paths in the two metadata sheets instead.

### Model information and dataset result files

- `model_info`: one row per baseline/model column, with `Model`, `Model path`, `Config`, `Node`, `Ray job ID`, `W&B URL`, and `Code snapshot`.
- `dataset_results`: one row per model and canonical dataset, with `Model`, `Dataset`, and `Result filepath`. Use the exact local or remote detailed-result file, such as `az://.../val_data_gen/<source>/0.jsonl` or `result_details.jsonl`, not just its parent directory or a metrics summary.
- Collect these fields from the actual run and artifact records. Leave unavailable fields blank; do not invent baseline checkpoint paths or paths for datasets that were not evaluated.
- Keep result paths associated with the correct model when extending a workbook. Existing metadata is preserved. Metric-only older workbooks with the current dataset layout can still be extended; their unknown metadata remains blank.

## Procedure

1. Determine the model label (e.g. `remax_qwen_bad_bracket_e1a@step80`).
2. Collect per-dataset metrics from one of:
   - **Ray job(s)**: `--from-ray <node> <job-id>`. Repeat for separate openasr / openasr_ml jobs. The script runs `brix ssh <node> -- ray job logs <id>` and parses `val-aux/<dataset>/p_err/mean@1:<float>`.
   - **Text dump**: `--from-text <path>` containing lines like `val-aux/ami_clean/p_err/mean@1:0.0705`.
   - **JSON**: `--metrics <path>` with `{"ami_clean": 0.0705, ...}` (fractions, not percent). ML names in either order (`fleurs_de` / `de_fleurs`) are accepted for logs and JSON, including baseline overrides.
3. Optionally pass `--extend-xlsx <prior.xlsx>` to append the new model as the next column after the existing baseline/model columns. Older dataset layouts are rejected rather than misreading rows; rebuild them from matching metrics. Do not alias uncleaned English datasets to cleaned versions.
   Collect the new model's metadata and pass `--model-info <model.json>` and `--dataset-results <results.json>`:
   ```json
   {
     "model_path": "az://storage/container/model/global_step_80/qwen_hf",
     "config": "recipe/phimm/config/eval/eval_openasr.yaml",
     "node": "verl-n1-i4",
     "ray_job_id": "raysubmit_example",
     "wandb_url": "https://wandb.example/project/runs/example",
     "code_snapshot": "commit-id"
   }
   ```
   The separate result-path JSON maps canonical dataset names (or supported ML aliases) to exact files:
   ```json
   {"ami_clean": "az://storage/container/results/ami/0.jsonl", "de_fleurs": "az://storage/container/results/de_fleurs/0.jsonl"}
   ```
4. Run [scripts/build_openasr_xlsx.py](./scripts/build_openasr_xlsx.py). Output defaults to `tmp/openasr_report/<label>.xlsx`.
5. Reopen the workbook and verify every eligible dataset has a WERR formula and every eligible average has a numeric WERR. Verify all three sheets, the model identity/path, and each supplied dataset result filepath; when extending, verify the earlier models' metadata is unchanged. Do not post-process delta cells into blanks. Refresh an existing report if its deltas were previously withheld.

### Required delta handling

- Use the supplied `2609r2` baseline by default. Missing checkpoint or scoring metadata must not suppress WERR or cause notes or comments to be added.
- Never fill missing candidate metrics from the baseline. A dataset WERR is unavailable only when either value is missing or the baseline is zero; leave that delta blank without a note or comment. A zero candidate against a positive baseline is a valid 100% WERR.
- For English and language average deltas, use the same intersection of available datasets for baseline and candidate: `1 - mean(candidate_common) / mean(baseline_common)`. For overall ML WERR, first compute each language mean over its matched datasets, then average those unrounded means with equal language weights on both sides. Do not average individual WERR values or compare means with different coverage. Displayed means summarize each column's available datasets with the same aggregation rules.
- Verify both positive and negative deltas, all model columns when extending a workbook, missing/zero-baseline handling, and matched-coverage averages. Preserve percentage formatting and the zero-centered WERR color scale.

## Examples

```bash
/home/boren/.virtualenvs/openai/bin/python \
  .github/skills/openasr-report/scripts/build_openasr_xlsx.py \
  "remax_qwen_bad_bracket_e1a@step80" \
  --from-ray verl-n1-i4 raysubmit_XJKM1qFA9ZisBfgz \
  --from-ray verl-n1-i12 raysubmit_yzwjKruFFcAq4Tuh \
  --model-info tmp/model_info.json \
  --dataset-results tmp/dataset_results.json \
  --out tmp/openasr_report/remax_qwen_bad_bracket_e1a_step80.xlsx
```

Extend an existing xlsx with another model column:

```bash
/home/boren/.virtualenvs/openai/bin/python \
  .github/skills/openasr-report/scripts/build_openasr_xlsx.py \
  "new_model@step100" \
  --metrics tmp/wer.json \
  --model-info tmp/model_info.json \
  --dataset-results tmp/dataset_results.json \
  --extend-xlsx tmp/openasr_report/prior_report.xlsx \
  --out tmp/openasr_report/combined.xlsx
```

## Baseline

Column `A` uses the user-supplied 2609r2 dataset percentages dated 2026-10-07, divided by 100.
These latest values replace the 2026-10-01 defaults. Continue to collect each
new candidate's actual metrics; never fill candidate values from the baseline.
The user identified the
supplied numbers as 2609r2; no checkpoint path was supplied. Do not attribute these
values to the historical 2607 FP8 checkpoint or to 2609v1.
Check dataset identities. Use these supplied
values to compute WERR even when full baseline provenance is unavailable; do not
claim independently verified scoring equivalence. If the user supplies a more
appropriate baseline, override with `--baseline <json>` and
`--baseline-label <name>`. Never silently alias genuinely different datasets.

| Dataset / summary | Column A |
| --- | --- |
| ami_clean | 6.72% |
| earnings22-cleaned-aa-chunked | 5.02% |
| gigaspeech_clean | 7.58% |
| librispeech-clean | 1.26% |
| librispeech-other | 2.74% |
| monsoon_en_in | 3.39% |
| spgispeech | 2.15% |
| voxpopuli-cleaned-aa | 1.63% |
| avg | 3.81% |
| de_fleurs | 2.32% |
| de_mcv | 1.90% |
| de avg | 2.11% |
| es_fleurs | 2.71% |
| es_mcv | 2.20% |
| es_mls | 2.58% |
| es avg | 2.50% |
| fr_fleurs | 2.86% |
| fr_mcv | 4.17% |
| fr_mls | 2.37% |
| fr avg | 3.13% |
| monsoon_hi_in | 8.78% |
| hi avg | 8.78% |
| it_fleurs | 1.30% |
| it_mcv | 1.78% |
| it_mls | 3.99% |
| it avg | 2.36% |
| nl_fleurs | 3.88% |
| nl_mcv | 1.64% |
| nl_mls | 4.53% |
| nl avg | 3.35% |
| pt_fleurs | 3.10% |
| pt_mls | 3.36% |
| pt avg | 3.23% |
| ml avg | 3.64% |

Recompute averages from the dataset values with full precision, displaying
two decimal places. English averages the eight datasets (`3.81125%`).
ML averages the seven unrounded language means (`3.636666...%`), not all
17 datasets (`3.145294...%`). Apply this language-macro rule to candidates
and matched-coverage ML WERR too. Summary rows are not dataset metrics.

When extending a workbook, preserve its existing baseline dataset values
and metadata; recompute average rows and WERR with the current aggregation
rules. To replace an older baseline with these latest defaults, rebuild
the report from the original candidate metrics rather than silently changing
the baseline of an existing comparison.

## Python Environment

Use `/home/boren/.virtualenvs/openai/bin/python`; the system Python lacks `openpyxl`.

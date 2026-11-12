---
name: openasr-report
description: Build or extend the canonical OpenASR + OpenASR_ML + OpenExt Excel comparison workbook from completed evaluation metrics, metadata, and exact result paths. Use for baseline-aware WER/WERR reporting; do not use it to launch evaluations, analyze utterance-level errors, or update an Obsidian vault.
---

# OpenASR Comparison Report (xlsx)

Generate an Excel report (`.xlsx`) that compares a new model column against the `r2611v1` baseline (and any other model columns already present in a prior report). The updated layout includes cleaned English datasets, Monsoon English/Hindi, URGENT2024, Dutch, a separate grouped OpenLong section, and OpenExt expansion datasets.

This skill is report-only. It consumes completed evaluation evidence and writes
an `.xlsx`; it does not launch jobs or publish results to another repository.

## When to Use

- A verl ASR eval job (`eval_openasr` and/or `eval_openasr_ml`) has just produced per-dataset WERs and you need the canonical Excel report.
- Adding another checkpoint as a new column (B, C, D, ...) to an existing xlsx report.
- Reporting on a `global_step_<N>` checkpoint after training.

## Layout

Three sheets: `openasr`, `model_info`, and `dataset_results`. The `openasr` metrics layout is:

- **Row 2** `Header`: `<baseline-label>`, `<model-label-1>`, `<model-label-2>`, ..., `WERR` (one WERR header per non-baseline model column).
- **Row 3** `OpenEN`: `A`, `B`, `C`, ..., `A->B`, `A->C`, ...
- **Rows 4–12**: OpenASR datasets — `ami_clean`, `earnings22-cleaned-aa-chunked`, `gigaspeech_clean`, `librispeech-clean`, `librispeech-other`, `monsoon_en_in`, `spgispeech_clean`, `voxpopuli-cleaned-aa`, `urgent2024`.
- **Row 13** `avg`: numeric mean across the nine OpenASR datasets per column.
- **Row 14** `OpenML`: repeated column identifiers for the ML section.
- **Rows 15–38**: ML datasets grouped by language with per-language `<lang> avg` rows interleaved:
  - de: `de_fleurs`, `de_mcv`
  - es: `es_fleurs`, `es_mcv`, `es_mls`
  - fr: `fr_fleurs`, `fr_mcv`, `fr_mls`
  - hi: `monsoon_hi_in`
  - it: `it_fleurs`, `it_mcv`, `it_mls`
  - nl: `nl_fleurs`, `nl_mcv`, `nl_mls`
  - pt: `pt_fleurs`, `pt_mls`
- **Row 39** `avg` (ML): numeric unweighted mean of the seven unrounded language means per column. With partial coverage, include only languages with available datasets.
- **Row 40** `OpenLong`: repeated column identifiers for the long-audio section.
- **Rows 41–43**: `earnings21`, `earnings22`, `coraal`.
- **Row 44** `avg` (OpenLong): numeric unweighted mean across all three OpenLong datasets per column. With partial coverage, average only available datasets; never fill missing candidate metrics from the baseline. Its numeric WERR uses matched dataset coverage, independently of the English and ML averages.
- **Row 45** `OpenExt`: repeated column identifiers for the expansion section from [openasr_ml_expansion_verb_langhint_zip.yaml](../../../recipe/phimm/config/data/val_data/openasr/openasr_ml_expansion_verb_langhint_zip.yaml).
- **Rows 46–51**: expansion datasets grouped by language, with a `<lang> avg` after each group:
  - zh: `zh_fleurs`, `zh_aishell1`, then `zh avg` (row 48)
  - vi: `vi_fleurs`, `vi_mcv`, then `vi avg` (row 51)
- **Row 52** `avg` (OpenExt): unweighted mean of the unrounded available language means, using the same partial-coverage and matched-coverage WERR rules as OpenML. OpenExt is independent of OpenEN, OpenML, and OpenLong averages. Chinese scores are CER; Vietnamese scores are WER, so this section summarizes the configured error rates.
- **WERR columns**: compute `1 - model / baseline` for every available pair with a positive baseline. Per-dataset cells use Excel formulas; average rows store numeric WERR. Positive values mean improvement; negative values mean regression. Do not remove deltas because baseline checkpoint/scoring provenance is incomplete.

Formatting:
- All data columns (everything except column A) center-aligned.
- Header / section header rows (`OpenEN`, `OpenML`, `OpenLong`, `OpenExt`): light blue.
- Per-language `<lang> avg` rows: light yellow.
- `avg` rows (English, ML, OpenLong, and OpenExt): light green.
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
   - **Ray job(s)**: `--from-ray <node> <job-id>`. Repeat for separate openasr / openasr_ml jobs. The script runs `brix ssh <node> -- ray job logs <id>` and parses `val-core/<dataset>/p_err/mean@1:<float>` or `val-aux/<dataset>/p_err/mean@1:<float>`. Prefer core when both namespaces occur for a dataset; do not substitute `val-aux/*/wer` for `p_err`.
   - **Text dump**: `--from-text <path>` containing lines like `val-core/ami_clean/p_err/mean@1:0.0705`.
   - **JSON**: `--metrics <path>` with `{"ami_clean": 0.0705, ...}` (fractions, not percent). ML/OpenExt language-first names in either order (`fleurs_de` / `de_fleurs`, `fleurs_zh` / `zh_fleurs`, `aishell1_zh` / `zh_aishell1`) are accepted for logs and JSON, including baseline overrides.
3. Optionally pass `--extend-xlsx <prior.xlsx>` to append the new model as the next column after the existing baseline/model columns. Reports with the current English/ML layout but no OpenLong or OpenExt section can be extended: the missing sections are added while existing baseline/model values remain unchanged, including missing OpenLong/OpenExt baselines. Do not backfill the new default OpenExt baseline into an existing workbook. Rebuild from original candidate metrics to adopt the new baseline defaults. Other incompatible dataset layouts are rejected rather than misreading rows. Do not alias uncleaned English datasets to cleaned versions.
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

### English dataset names

Collect English metrics and dataset-result keys under their exact canonical
names, including `ami_clean`, `gigaspeech_clean`, and `spgispeech_clean`.
The builder maps the evaluation sources `ls_clean` and `ls_other` to the report
rows `librispeech-clean` and `librispeech-other`. Preserve those original source
names in evaluation configs. The user-confirmed alias `earnings22-cleaned-aa`
maps to `earnings22-cleaned-aa-chunked`. Other English names retain their exact identities.
Keep the exact detailed-result filepath unchanged, independently of the metric
key. Verify the input names against the effective evaluation YAML before building
so a naming mismatch is resolved rather than reported as missing evaluation.

Before delivery, inspect rows4 (`ami_clean`),6 (`gigaspeech_clean`), and10
(`spgispeech_clean`) in every evaluated model column: completed values,
eligible WERR formulas and result paths must be present. Include all completed English datasets in the English average; missing baseline
scores remain blank and average WERR uses matched coverage.

### Required delta handling

- Use the supplied `r2611v1` baseline by default. Missing checkpoint or scoring metadata must not suppress WERR or cause notes or comments to be added.
- Never fill missing candidate metrics from the baseline. A dataset WERR is unavailable only when either value is missing or the baseline is zero; leave that delta blank without a note or comment. A zero candidate against a positive baseline is a valid 100% WERR.
- For English and language average deltas, use the same intersection of available datasets for baseline and candidate: `1 - mean(candidate_common) / mean(baseline_common)`. For overall ML and OpenExt WERR, first compute each language mean over its matched datasets, then average those unrounded means with equal language weights on both sides. Do not average individual WERR values or compare means with different coverage. Displayed means summarize each column's available datasets with the same aggregation rules.
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

Column `A` defaults to **r2611v1**, using the user-supplied table headed
“2611 s2 / shadow” on 2026-11-09. Values below are percentages; the builder
stores fractions. This replaces the 2609r2 defaults for newly built reports.
No checkpoint path, scoring configuration, or detailed-result paths were supplied;
leave those metadata fields blank. Do not infer a checkpoint step or claim
independently verified scoring equivalence.

| Dataset | WER |
| --- | --- |
| ami_clean | 6.68% |
| earnings22-cleaned-aa-chunked | 5.26% |
| gigaspeech_clean | 7.36% |
| librispeech-clean | 1.18% |
| librispeech-other | 2.70% |
| monsoon_en_in | 3.03% |
| spgispeech_clean | 2.10% |
| voxpopuli-cleaned-aa | 1.49% |
| urgent2024 | 9.85% |
| de_fleurs | 2.57% |
| de_mcv | 1.92% |
| es_fleurs | 2.60% |
| es_mcv | 2.19% |
| es_mls | 2.64% |
| fr_fleurs | 3.43% |
| fr_mcv | 4.19% |
| fr_mls | 2.37% |
| hy_fleurs (supplementary) | 6.38% |
| hy_mcv (supplementary) | 11.90% |
| monsoon_hi_in | 8.61% |
| it_fleurs | 1.46% |
| it_mcv | 1.82% |
| it_mls | 4.04% |
| nl_fleurs | 4.01% |
| nl_mcv | 1.68% |
| nl_mls | 3.82% |
| pt_fleurs | 3.18% |
| pt_mls | 5.75% |
| earnings21 | 9.21% |
| earnings22 | 10.05% |
| coraal (supplied as coraal_avg) | 13.02% |
| zh_fleurs | 2.97% |
| zh_aishell1 | 8.45% |
| vi_fleurs | 4.19% |
| vi_mcv | 8.12% |

The four OpenExt baseline scores were supplied on 2026-11-12. OpenExt includes
only Chinese and Vietnamese. Exclude `eslo_telephony_fr` and `accueil_ubs_fr`
from the workbook, result-path metadata, and averages, including when extending
a workbook that previously included these French expansion rows.
OpenExt baseline means are **5.71% zh**, **6.155% vi**, and **5.9325% overall**
(displayed as **5.71%**, **6.16%**, and **5.93%**). Compute these from the
unrounded dataset scores rather than hard-coding summaries. Overall WERR
includes only languages and datasets shared with the baseline. The YAML's exclusion from
established headline averages does not exclude these datasets from their own
OpenExt summary.

ML corpus-first input names (`fleurs_de`, `mcv_de`, etc.) map to the
language-first names above. The supplied `coraal_avg` identifies the grouped
`coraal` baseline row; it does not identify any single CORAAL location.
The user explicitly confirmed that the supplied `spgispeech` score belongs
to `spgispeech_clean`; this confirmation applies to this baseline and does not
introduce a general English dataset alias. The Armenian scores are preserved
in `BASELINE_SUPPLEMENTARY_METRICS`; the current workbook keeps its
seven-language layout (de/es/fr/hi/it/nl/pt) and has no Armenian rows.

`urgent2024_clean` is excluded from the report, including result-path metadata
and averages, even when supplied as input. Extending an older report with that
row removes it while preserving the other dataset metrics and metadata.
The user confirmed that `earnings22-cleaned-aa-chunked` and
`earnings22-cleaned-aa` identify the same cleaned Earnings22 dataset, including
the former `earnings22_cleaned_aa` row. Keep the supplied **5.26%** baseline
only in the English section under `earnings22-cleaned-aa-chunked`. Exclude
`earnings22_cleaned_aa` from OpenLong, result-path metadata, and averages,
including when extending an older report. Preserve the remaining dataset
metrics and exact result paths.

The user supplied summary percentages of **4.41% English**, **3.92% ML**,
**3.48% ML without Armenian**, and **10.76% OpenLong**. Preserve these as
source summaries, not extra dataset metrics or hard-coded workbook averages.
Recompute displayed averages from available dataset scores at full precision.
The canonical ML average continues to average the seven unrounded language
means, giving **3.82%** for r2611v1; the supplied ML summaries use a different
aggregation and must not replace this calculation. OpenLong `avg` gives
**10.76%** across `earnings21`, `earnings22`, and `coraal`. OpenLong has one
summary row, `avg`; extending older workbooks drops the redundant `avg0` row. English
averages its nine available canonical datasets, giving **4.41%**.
Use matched coverage for all average WERR calculations as described above.

Use these supplied scores for WERR even when full baseline provenance is
unavailable. To use another baseline, pass `--baseline <json>` and
`--baseline-label <name>`.

When extending a workbook, preserve its existing baseline identity, dataset
values, and metadata; recompute summaries and WERR with the current aggregation
rules. Rebuild from the original candidate metrics to replace a historical
baseline with r2611v1. Existing reports are not rewritten by this default update.

## Python Environment

Use `/home/boren/.virtualenvs/openai/bin/python`; the system Python lacks `openpyxl`.

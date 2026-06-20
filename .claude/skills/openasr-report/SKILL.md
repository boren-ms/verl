---
name: openasr-report
description: Build an OpenASR + OpenASR_ML xlsx comparison report for a trained verl ASR model, with the new model inserted as the next column (B, C, D, ...) next to the fixed Qwen3.5-audio baseline (column A) and per-model WERR columns. Use when summarizing eval_openasr/eval_openasr_ml results from a Ray job, blob checkpoint dir, or local metrics file into the standardized Excel report with per-language and overall averages, comparing a new checkpoint against the baseline, or extending an existing xlsx report with another model column.
argument-hint: '<model-label> [--from-ray <node> <job-id> [<job-id> ...]] | [--from-text <file>] | [--metrics <json>] [--extend-xlsx <xlsx>] [--out <xlsx>]'
---

# OpenASR Comparison Report (xlsx)

Generate an Excel report (`.xlsx`) that compares a new model column against the fixed `Qwen3.5-audio` baseline (and any other model columns already present in a prior report). Layout matches `~/code/MoE/results/template.xlsx`.

## When to Use

- A verl ASR eval job (`eval_openasr` and/or `eval_openasr_ml`) has just produced per-dataset WERs and you need the canonical Excel report.
- Adding another checkpoint as a new column (B, C, D, ...) to an existing xlsx report.
- Reporting on a `global_step_<N>` checkpoint after training.

## Layout

Single sheet `openasr`:

- **Row 2** `Header`: `Baseline`, `<model-label-1>`, `<model-label-2>`, ..., `WERR` (one WERR header per non-baseline model column).
- **Row 3** `Column`: `A`, `B`, `C`, ..., `A->B`, `A->C`, ...
- **Rows 4–11**: OpenASR datasets — `ami, earnings22, gigaspeech, ls_clean, ls_other, spgispeech, tedlium, voxpopuli`.
- **Row 12** `avg`: numeric mean across the 8 OpenASR datasets per column.
- **Row 13**: repeated `Column` header for the ML section.
- **Rows 14–31**: ML datasets grouped by language with per-language `<lang> avg` rows interleaved:
  - de: `de_fleurs`, `de_mcv`
  - es: `es_fleurs`, `es_mcv`, `es_mls`
  - fr: `fr_fleurs`, `fr_mcv`, `fr_mls`
  - it: `it_fleurs`, `it_mcv`, `it_mls`
  - pt: `pt_fleurs`, `pt_mls`
- **Row 32** `ml avg`: numeric mean across all 13 ML datasets per column.
- **WERR columns**: per-dataset cells use `=1-<model>/Baseline`; average rows store the computed numeric WERR so all cells are filled even before Excel recalculates.

Formatting:
- All data columns (everything except column A) center-aligned.
- Header / `Column` rows: light blue.
- Per-language `<lang> avg` rows: light yellow.
- `avg` and `ml avg` rows: light green.
- WERR columns carry a 3-color scale rule (red ← 0 → green) with midpoint fixed at 0.
- Numeric cells formatted as `0.00%`.

## Procedure

1. Determine the model label (e.g. `remax_qwen_bad_bracket_e1a@step80`).
2. Collect per-dataset metrics from one of:
   - **Ray job(s)**: `--from-ray <node> <job-id>`. Repeat for separate openasr / openasr_ml jobs. The script runs `brix ssh <node> -- ray job logs <id>` and parses `val-aux/<dataset>/p_err/mean@1:<float>`.
   - **Text dump**: `--from-text <path>` containing lines like `val-aux/ami/p_err/mean@1:0.1351`.
   - **JSON**: `--metrics <path>` with `{"ami": 0.1351, ...}` (fractions, not percent).
3. Optionally pass `--extend-xlsx <prior.xlsx>` to append the new model as the next column after the existing baseline/model columns.
4. Run [scripts/build_openasr_xlsx.py](./scripts/build_openasr_xlsx.py). Output defaults to `tmp/openasr_report/<label>.xlsx`.

## Examples

```bash
/home/boren/.virtualenvs/openai/bin/python \
  .github/skills/openasr-report/scripts/build_openasr_xlsx.py \
  "remax_qwen_bad_bracket_e1a@step80" \
  --from-ray verl-n1-i4 raysubmit_XJKM1qFA9ZisBfgz \
  --from-ray verl-n1-i12 raysubmit_yzwjKruFFcAq4Tuh \
  --out tmp/openasr_report/remax_qwen_bad_bracket_e1a_step80.xlsx
```

Extend an existing xlsx with another model column:

```bash
/home/boren/.virtualenvs/openai/bin/python \
  .github/skills/openasr-report/scripts/build_openasr_xlsx.py \
  "new_model@step100" \
  --metrics tmp/wer.json \
  --extend-xlsx tmp/openasr_report/prior_report.xlsx \
  --out tmp/openasr_report/combined.xlsx
```

## Baseline

Column `A` is the fixed `Qwen3.5-audio` baseline embedded in the script — currently the `fast-llm-2605-qwen3-5-9b-s2-st-example-r2 @ step 90000` checkpoint (OpenASR avg 4.95%, OpenASR-ML avg 2.87%). Override with `--baseline <json>` and `--baseline-label <name>` if needed.

## Python Environment

Use `/home/boren/.virtualenvs/openai/bin/python`; the system Python lacks `openpyxl`.

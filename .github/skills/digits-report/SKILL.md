---
name: digits-report
description: "Build a digits-dataset XLSX comparison report for a trained verl ASR model, with Digit CER for enus_digits_random and enus_digits_repeat, using the fixed Qwen3.5-audio baseline. Use when summarizing digit evaluation results, comparing digits CER, reporting enus_digits_random or enus_digits_repeat, or extending a digits report with another checkpoint."
argument-hint: '<model-label> [--from-ray <node> <job-id> [<job-id> ...]] | [--from-text <file>] | [--metrics <json>] [--extend-xlsx <xlsx>] [--out <xlsx>]'
---

# Digits Comparison Report (xlsx)

Generate an Excel report (`.xlsx`) for the two digits validation datasets. It compares one or more models with the fixed `Qwen3.5-audio` baseline and reports digit character error rate (CER).

## When to Use

- A digits evaluation job has just produced results for `enus_digits_random` and `enus_digits_repeat`.
- Comparing a checkpoint against the established Qwen3.5-audio digits baseline.
- Appending a later checkpoint to an existing digits report.

## Layout

Single sheet `digits`:

- **Row 2** `Header`: `Baseline`, one column per added model, and one `CER reduction` column per non-baseline model.
- **Row 3** `Column`: spreadsheet column labels and baseline-to-model comparison labels.
- **Rows 4-5**: One `Digit CER` row per dataset.
- **CER reduction**: $1 - \frac{\text{model CER}}{\text{baseline CER}}$; positive values improve over the baseline.

The baseline values are:

| Dataset | Digit CER |
|---|---:|
| `enus_digits_random` | 0.04% |
| `enus_digits_repeat` | 0.92% |

All numeric cells use percentage formatting. Header rows are light blue, metric rows are visually grouped by dataset, and CER-reduction columns use a red-white-green conditional scale.

## Procedure

1. Determine the model label, such as `remax_digits@step80`.
2. Collect the metrics using one of these inputs:
  - **Ray job(s)**: `--from-ray <node> <job-id>`. The script retrieves `ray job logs` and parses `val-aux/<dataset>/cer/mean@1:<float>`.
   - **Text dump**: `--from-text <path>` with the same metric lines.
  - **JSON**: `--metrics <path>` using `{"enus_digits_random": 0.0004, "enus_digits_repeat": 0.0092}`. JSON values are fractions, not percentages.
3. Optionally pass `--extend-xlsx <prior.xlsx>` to append a model column.
4. Run [scripts/build_digits_xlsx.py](./scripts/build_digits_xlsx.py). Output defaults to `tmp/digits_report/<label>.xlsx`.

## Example

```bash
/home/boren/.virtualenvs/openai/bin/python \
  .github/skills/digits-report/scripts/build_digits_xlsx.py \
  "remax_digits@step80" \
  --from-ray verl-n1-i4 raysubmit_XJKM1qFA9ZisBfgz \
  --out tmp/digits_report/remax_digits_step80.xlsx
```

Append a new checkpoint to an existing report:

```bash
/home/boren/.virtualenvs/openai/bin/python \
  .github/skills/digits-report/scripts/build_digits_xlsx.py \
  "remax_digits@step160" \
  --metrics tmp/digits_metrics.json \
  --extend-xlsx tmp/digits_report/prior_report.xlsx \
  --out tmp/digits_report/combined.xlsx
```

## Python Environment

Use `/home/boren/.virtualenvs/openai/bin/python`; the system Python lacks `openpyxl`.
---
name: inhouse-dter-report
description: Build an in-house DTER xlsx comparison report for a trained verl ASR model, with the new model inserted as the next column (B, C, D, ...) next to the fixed Qwen3.5-audio baseline (column A) and per-model WERR columns. Use when summarizing eval_inhouse_2605 (or similar in-house DTER) results from a Ray job, blob checkpoint dir, or local metrics file into the standardized Excel report with per-locale and overall averages, comparing a new checkpoint against the inhouse baseline, or extending an existing xlsx inhouse-DTER report with another model column.
argument-hint: '<model-label> [--from-ray <node> <job-id> [<job-id> ...]] | [--from-text <file>] | [--metrics <json>] [--extend-xlsx <xlsx>] [--out <xlsx>]'
---

# In-house DTER Comparison Report (xlsx)

Generate an Excel report (`.xlsx`) that compares a new model column against the fixed `Qwen3.5-audio` baseline (and any other model columns already present in a prior report) on the in-house DTER eval corpora (en-US + nl-NL). Layout mirrors `.github/skills/openasr-report` but uses **micro-DTER** values.

## When to Use

- A verl ASR eval job using `eval_inhouse_2605.yaml` (or similar inhouse DTER eval) has just produced per-corpus DTER and you need the canonical Excel report.
- Adding another checkpoint as a new column (B, C, D, ...) to an existing inhouse-DTER xlsx report.
- Reporting on a `global_step_<N>` checkpoint after training.

## Layout

Single sheet `inhouse_dter`:

- **Row 2** `Header`: `Baseline`, `<model-label-1>`, `<model-label-2>`, ..., `WERR` (one per non-baseline model column).
- **Row 3** `Column`: `A`, `B`, `C`, ..., `A->B`, `A->C`, ...
- **Rows 4–6** en-US datasets:
  - `Conversation_DTEST_FY21Q1_en-US`
  - `Conversation_OnlineMeetings_DTEST_FY25Q3_en-US_DTEST_OfflineDataCollection`
  - `Dictation_Commonset_OfficeOffline_FY24Q3_en-US_DTEST_OfflineDataCollection`
- **Row 7** `en-US avg` (light yellow, per-locale average).
- **Rows 8–10** nl-NL datasets:
  - `Conversation_DTEST_FY23Q2_nl-NL_DTEST`
  - `Conversation_OnlineMeetings_DTEST_FY23Q1_nl-NL_DTEST`
  - `Dictation_DTEST_L_D_FY23Q4_nl-NL_DTEST`
- **Row 11** `nl-NL avg` (light yellow, per-locale average).
- **Row 12** `overall avg` (light green, average across all 6 datasets).
- **WERR columns**: per-dataset cells use `=1-<model>/Baseline`; average rows store the computed numeric WERR so all cells are filled even before Excel recalculates.

Formatting:
- All data columns (everything except column A) center-aligned.
- Header / `Column` rows: light blue.
- Per-locale `<locale> avg` rows: light yellow.
- `overall avg` row: light green.
- WERR columns carry a 3-color scale rule (red ← 0 → green) with midpoint fixed at 0.
- Numeric cells formatted as `0.00%`.

## Micro vs macro DTER

verl logs `val-aux/<corpus>/dter/mean@1` (a per-segment macro average) which **does not match** the in-house reference DTER. The canonical metric is **micro-DTER = sum(edits) / sum(tokens)**, which can be recovered from the logged per-corpus aggregates:

```
micro_dter[corpus] = mean@1(dter_n_err[corpus]) / mean@1(dter_n_ref[corpus])
```

Both `mean@1` aggregates share the same sample count per corpus, so the ratio equals `sum/sum`. The build script parses the two log keys per corpus and emits the micro value. The bare `dter/mean@1` line is ignored.

## Procedure

1. Determine the model label (e.g. `remax_qwen_inhouse_e1a@step80`).
2. Collect per-corpus aggregates from one of:
   - **Ray job(s)**: `--from-ray <node> <job-id>`. Repeat for separate jobs. The script runs `brix ssh <node> -- ray job logs <id>` and parses `val-aux/<corpus>/dter_n_err/mean@1` and `val-aux/<corpus>/dter_n_ref/mean@1`, then computes the micro-DTER per corpus.
   - **Text dump**: `--from-text <path>` containing the same `val-aux/...` lines.
   - **JSON**: `--metrics <path>` with `{"Conversation_DTEST_FY21Q1_en-US": 0.1828, ...}` (fractions, not percent).
3. Optionally pass `--extend-xlsx <prior.xlsx>` to append the new model as the next column after the existing baseline/model columns.
4. Run [scripts/build_inhouse_dter_xlsx.py](./scripts/build_inhouse_dter_xlsx.py). Output defaults to `tmp/inhouse_dter_report/<label>.xlsx`.

### Dataset name matching

verl's `data_source` is typically the short corpus name (e.g. `Conversation_DTEST_FY21Q1`) without the locale suffix. When matching parsed log keys to the canonical dataset names in the report, the script tries (in order):

1. exact match on the canonical name (`Conversation_DTEST_FY21Q1_en-US`),
2. stripped locale suffix (`Conversation_DTEST_FY21Q1`),
3. case-insensitive variants of the above.

Pass `--metrics` directly if you need to override the mapping.

## Examples

```bash
/home/boren/.virtualenvs/openai/bin/python \
  .github/skills/inhouse-dter-report/scripts/build_inhouse_dter_xlsx.py \
  "remax_qwen_inhouse_e1a@step80" \
  --from-ray verl-n1-i4 raysubmit_XJKM1qFA9ZisBfgz \
  --out tmp/inhouse_dter_report/remax_qwen_inhouse_e1a_step80.xlsx
```

Extend an existing xlsx with another model column:

```bash
/home/boren/.virtualenvs/openai/bin/python \
  .github/skills/inhouse-dter-report/scripts/build_inhouse_dter_xlsx.py \
  "new_model@step100" \
  --metrics tmp/inhouse_dter.json \
  --extend-xlsx tmp/inhouse_dter_report/prior_report.xlsx \
  --out tmp/inhouse_dter_report/combined.xlsx
```

## Baseline

Column `A` is the fixed `Qwen3.5-audio` baseline embedded in the script:

| Locale | Dataset | DTER% |
|---|---|---|
| en-US | average | 14.16 |
| en-US | Conversation_DTEST_FY21Q1_en-US | 18.63 |
| en-US | Conversation_OnlineMeetings_DTEST_FY25Q3_en-US_DTEST_OfflineDataCollection | 13.74 |
| en-US | Dictation_Commonset_OfficeOffline_FY24Q3_en-US_DTEST_OfflineDataCollection | 10.10 |
| nl-NL | average | 21.56 |
| nl-NL | Conversation_DTEST_FY23Q2_nl-NL_DTEST | 24.76 |
| nl-NL | Conversation_OnlineMeetings_DTEST_FY23Q1_nl-NL_DTEST | 24.22 |
| nl-NL | Dictation_DTEST_L_D_FY23Q4_nl-NL_DTEST | 15.70 |

Override with `--baseline <json>` and `--baseline-label <name>` if needed.

## Python Environment

Use `/home/boren/.virtualenvs/openai/bin/python`; the system Python lacks `openpyxl`.

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

## Schemas

Select the dataset schema (and its embedded baseline) with `--schema`:

- `--schema default` (default): the 6-dataset en-US + nl-NL canonical schema above.
- `--schema enus_seg`: the 5 en-US TER corpora from the segmented long-audio eval
  `inhouse_2605_enus_seg` (the two `CustomerSpeechDomainSet_*` Entity sets are excluded).
  `data_source` keys are the short corpus names. Column `A` is the fixed
  `Qwen3.5-audio` baseline (`eval_qwen/inhouse_2605_enus_seg`), embedded as
  micro-DTER (`dter_n_err / dter_n_ref`):

  | Locale | Dataset | DTER% |
  |---|---|---|
  | en-US | average | 18.32 |
  | en-US | Conversation_DTEST_FY21Q1 | 18.57 |
  | en-US | Conversation_OnlineMeetings_DTEST_FY25Q3 | 13.56 |
  | en-US | Dictation_Commonset_OfficeOffline_FY24Q3 | 10.13 |
  | en-US | OnlineMeetings_CS_Product_FY22_FullMeeting | 23.32 |
  | en-US | OnlineMeetings_CS_Shiproom_FY22 | 26.02 |

  Example:

  ```bash
  /home/boren/.virtualenvs/openai/bin/python \
    .github/skills/inhouse-dter-report/scripts/build_inhouse_dter_xlsx.py \
    "remax_r2_punc_h2k_n12_s200@step32" \
    --schema enus_seg --metrics tmp/seg_target.json \
    --out tmp/inhouse_dter_report/remax_r2_punc_h2k_n12_s200_step32_seg.xlsx
  ```

- `--schema nlnl_seg`: the 3 nl-NL TER corpora from the segmented long-audio eval
  `inhouse_2605_nlnl` (the two `Conversation_DomainSet_*_Entity_*` sets are excluded).
  `data_source` keys are the short corpus names. Column `A` is the fixed
  `Qwen3.5-audio` baseline (`eval_qwen/inhouse_2605_nlnl`), embedded as
  micro-DTER (`dter_n_err / dter_n_ref`):

  | Locale | Dataset | DTER% |
  |---|---|---|
  | nl-NL | average | 21.46 |
  | nl-NL | Conversation_DTEST_FY23Q2 | 24.72 |
  | nl-NL | Conversation_OnlineMeetings_DTEST_FY23Q1 | 23.94 |
  | nl-NL | Dictation_DTEST_L_D_FY23Q4 | 15.73 |

  Example:

  ```bash
  /home/boren/.virtualenvs/openai/bin/python \
    .github/skills/inhouse-dter-report/scripts/build_inhouse_dter_xlsx.py \
    "remax_r2_nlnl@step32" \
    --schema nlnl_seg --metrics tmp/nlnl_target.json \
    --out tmp/inhouse_dter_report/remax_r2_nlnl_step32_seg.xlsx
  ```

- `--schema dadk_seg`: the 3 da-DK TER corpora from the segmented long-audio eval
  `inhouse_2605_dadk`. `data_source` keys are the short corpus names. Column `A`
  is the fixed `Qwen3.5-audio` baseline (`eval_qwen/inhouse_2605_dadk`), embedded
  as micro-DTER (`dter_n_err / dter_n_ref`):

  | Locale | Dataset | DTER% |
  |---|---|---|
  | da-DK | average | 23.47 |
  | da-DK | Conversation_DTEST_FY21Q3 | 23.60 |
  | da-DK | Conversation_OnlineMeetings_DTEST_FY23Q1 | 24.33 |
  | da-DK | Dictation_DTEST_L_D_FY23Q4 | 22.49 |

- `--schema huhu_seg`: the 3 hu-HU TER corpora from the segmented long-audio eval
  `inhouse_2605_huhu`. `data_source` keys are the short corpus names. Column `A`
  is the fixed `Qwen3.5-audio` baseline (`eval_qwen/inhouse_2605_huhu`), embedded
  as micro-DTER (`dter_n_err / dter_n_ref`):

  | Locale | Dataset | DTER% |
  |---|---|---|
  | hu-HU | average | 23.01 |
  | hu-HU | Conversation_DTEST_FY22Q4 | 22.80 |
  | hu-HU | Conversation_OnlineMeetings_DTEST_FY24Q2 | 21.93 |
  | hu-HU | Dictation_DTEST_L_D_FY25Q2 | 24.30 |

- `--schema nbno_seg`: the 3 nb-NO TER corpora from the segmented long-audio eval
  `inhouse_2605_nbno`. `data_source` keys are the short corpus names. Column `A`
  is the fixed `Qwen3.5-audio` baseline (`eval_qwen/inhouse_2605_nbno`), embedded
  as micro-DTER (`dter_n_err / dter_n_ref`):

  | Locale | Dataset | DTER% |
  |---|---|---|
  | nb-NO | average | 21.19 |
  | nb-NO | Conversation_DTEST_FY21Q3 | 21.88 |
  | nb-NO | Conversation_OnlineMeetings_DTEST_FY23Q1 | 20.54 |
  | nb-NO | Dictation_DTEST_L_D_FY23Q4 | 21.14 |

  Per-locale example:

  ```bash
  /home/boren/.virtualenvs/openai/bin/python \
    .github/skills/inhouse-dter-report/scripts/build_inhouse_dter_xlsx.py \
    "my_model@step32" \
    --schema dadk_seg --from-ray verl-n1-i2 raysubmit_XXXX \
    --out tmp/inhouse_dter_report/my_model_step32_dadk.xlsx
  ```

  To publish just the embedded baseline column for any schema, use
  `--baseline-only`:

  ```bash
  /home/boren/.virtualenvs/openai/bin/python \
    .github/skills/inhouse-dter-report/scripts/build_inhouse_dter_xlsx.py \
    --schema dadk_seg --baseline-only \
    --out tmp/inhouse_dter_report/inhouse_2605_dadk_baseline.xlsx
  ```

## Python Environment

Use `/home/boren/.virtualenvs/openai/bin/python`; the system Python lacks `openpyxl`.

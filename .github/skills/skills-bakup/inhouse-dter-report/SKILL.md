---
name: inhouse-dter-report
description: Build an in-house DTER xlsx comparison report for a trained verl ASR model, with the new model inserted as the next column (B, C, D, ...) next to the fixed Qwen3.5-audio baseline (column A) and per-model WERR columns. Use when summarizing eval_inhouse_2605 (or similar in-house DTER) results from a Ray job, blob checkpoint dir, or local metrics file into the standardized Excel report with per-locale and overall averages, comparing a new checkpoint against the inhouse baseline, or extending an existing xlsx inhouse-DTER report with another model column.
argument-hint: '<model-label> [--from-ray <node> <job-id> [<job-id> ...]] | [--from-text <file>] | [--metrics <json>] | [--model <label> <path> ...] [--extend-xlsx <xlsx>] [--out <xlsx>]'
---

# In-house DTER Comparison Report (xlsx)

Generate an Excel report (`.xlsx`) that compares a new model column against the fixed `Qwen3.5-audio` baseline (and any other model columns already present in a prior report) on the in-house DTER eval corpora (en-US + nl-NL). Layout mirrors `.github/skills/openasr-report` but uses **micro-DTER** values.

## When to Use

- A verl ASR eval job using `eval_inhouse_2605.yaml` (or similar inhouse DTER eval) has just produced per-corpus DTER and you need the canonical Excel report.
- Adding another checkpoint as a new column (B, C, D, ...) to an existing inhouse-DTER xlsx report.
- Reporting on a `global_step_<N>` checkpoint after training.

## Layout

Primary sheet `inhouse_dter`:

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

Second sheet `overall_improve_degrade`:

- One row per non-baseline model, sorted by overall WERR from best improvement to largest degradation.
- Columns: `Rank`, `Model`, `Direction`, `Baseline overall DTER`, `Model overall DTER`, `DTER delta`, `WERR`, `Datasets`.
- `DTER delta` is `Baseline overall DTER - Model overall DTER`; positive values are improvements.
- `Datasets` shows how many schema datasets contributed to that model's overall average.

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
   - **Text dump**: `--from-text <path>` containing the same `val-aux/...` lines (or `[<corpus>] DTER: x% [n_err/n_ref]` long-eval summary lines).
   - **JSON**: `--metrics <path>` with `{"Conversation_DTEST_FY21Q1_en-US": 0.1828, ...}` (fractions, not percent).

   The positional label plus the flags above build a **single** model column. To combine **multiple local results into one workbook in a single command** — each becoming its own column (B, C, D, ...) — use `--model` instead (next step).
3. **Multiple model columns (single command)**: pass `--model <label> <path>` once per model. `<path>` is auto-detected as one of:
   - an `az://` URL or local directory containing per-corpus `<slug>/measures.json` files (the canonical segmented long-audio eval output, e.g. `az://orngwus2cresco/.../inhouse_2605_5lang_seg_v2/`) — used together with `--schema all_seg` to populate **all 6 locales in one sheet** from a single source path;
   - a JSON `{dataset: dter_fraction}` mapping;
   - or a text log (the `val-aux/...` and/or `[<corpus>] DTER: ...` lines).
   Each `--model` becomes its own column in the order given, after the baseline. You may mix a positional-label column with additional `--model` columns; the positional column comes first. This replaces the need to chain `--extend-xlsx` through intermediate files when you already have all the local result files.
4. Optionally pass `--extend-xlsx <prior.xlsx>` to append the new model column(s) after the existing baseline/model columns of an existing report.
5. Run [scripts/build_inhouse_dter_xlsx.py](./scripts/build_inhouse_dter_xlsx.py). Output defaults to `tmp/inhouse_dter_report/<first-label>.xlsx`.

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

Combine **multiple local result files into a single workbook in one command** —
each `--model <label> <path>` becomes its own column (B, C, D, ...) next to the
baseline (A), with a matching `WERR` column (`A->B`, `A->C`, ...). Each `<path>`
is auto-detected as a JSON metrics file or a text eval log:

```bash
/home/boren/.virtualenvs/openai/bin/python \
  .github/skills/inhouse-dter-report/scripts/build_inhouse_dter_xlsx.py \
  --schema enus_seg \
  --model "step480@enus" tmp/logs/step480_enus.log \
  --model "step560@enus" tmp/logs/step560_enus.log \
  --model "qwen_ref"     tmp/metrics/qwen_enus.json \
  --out tmp/inhouse_dter_report/enus_step_sweep.xlsx
```

You can also start from a positional-label column and add more with `--model`
(the positional column is first), or append the `--model` columns onto an
existing report via `--extend-xlsx`.

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

- `--schema cscz_seg`: the 3 cs-CZ TER corpora from the segmented long-audio eval
  `inhouse_2605_cscz`. Internal keys are the per-corpus slug directory names
  (`cscz_conv_fy23q2`, `cscz_conv_om_fy24q2`, `cscz_dict_fy24q2`); display labels
  are the short corpus names. Column `A` is the fixed `Qwen3.5-audio` baseline
  (`eval_qwen/inhouse_2605_cscz`), embedded as micro-DTER (`dter_n_err / dter_n_ref`):

  | Locale | Dataset | DTER% |
  |---|---|---|
  | cs-CZ | average | 17.08 |
  | cs-CZ | Conversation_DTEST_FY23Q2 | 23.75 |
  | cs-CZ | Conversation_OnlineMeetings_DTEST_FY24Q2 | 14.42 |
  | cs-CZ | Dictation_DTEST_L_D_FY24Q2 | 13.07 |

- `--schema all_seg`: **all 6 locales × 3 corpora in one sheet** — the combined
  segmented long-audio eval (`inhouse_2605_5lang_seg_v2`). Internal keys are
  the per-corpus slug directory names (e.g. `enus_conv_fy21q1`,
  `dadk_conv_om_fy23q1`); display labels are the short corpus names prefixed with
  the locale code (e.g. `en-US_Conversation_DTEST_FY21Q1`).
  Column `A` is the fixed `Qwen3.5-audio` baseline (assembled from the per-locale
  baselines above). Rows: 3 en-US + 3 nl-NL + 3 da-DK + 3 hu-HU + 3 nb-NO + 3 cs-CZ,
  with 6 per-locale `<locale> avg` rows and one `overall avg` row:

  | Locale | Dataset (display) | Slug | DTER% |
  |---|---|---|---|
  | en-US | average | — | 14.09 |
  | en-US | en-US_Conversation_DTEST_FY21Q1 | `enus_conv_fy21q1` | 18.57 |
  | en-US | en-US_Conversation_OnlineMeetings_DTEST_FY25Q3 | `enus_conv_om_fy25q3` | 13.56 |
  | en-US | en-US_Dictation_Commonset_OfficeOffline_FY24Q3 | `enus_dict_office_fy24q3` | 10.13 |
  | nl-NL | average | — | 21.46 |
  | nl-NL | nl-NL_Conversation_DTEST_FY23Q2 | `nlnl_conv_fy23q2` | 24.72 |
  | nl-NL | nl-NL_Conversation_OnlineMeetings_DTEST_FY23Q1 | `nlnl_conv_om_fy23q1` | 23.94 |
  | nl-NL | nl-NL_Dictation_DTEST_L_D_FY23Q4 | `nlnl_dict_fy23q4` | 15.73 |
  | da-DK | average | — | 23.47 |
  | da-DK | da-DK_Conversation_DTEST_FY21Q3 | `dadk_conv_fy21q3` | 23.60 |
  | da-DK | da-DK_Conversation_OnlineMeetings_DTEST_FY23Q1 | `dadk_conv_om_fy23q1` | 24.33 |
  | da-DK | da-DK_Dictation_DTEST_L_D_FY23Q4 | `dadk_dict_fy23q4` | 22.49 |
  | hu-HU | average | — | 23.01 |
  | hu-HU | hu-HU_Conversation_DTEST_FY22Q4 | `huhu_conv_fy22q4` | 22.80 |
  | hu-HU | hu-HU_Conversation_OnlineMeetings_DTEST_FY24Q2 | `huhu_conv_om_fy24q2` | 21.93 |
  | hu-HU | hu-HU_Dictation_DTEST_L_D_FY25Q2 | `huhu_dict_fy25q2` | 24.30 |
  | nb-NO | average | — | 21.19 |
  | nb-NO | nb-NO_Conversation_DTEST_FY21Q3 | `nbno_conv_fy21q3` | 21.88 |
  | nb-NO | nb-NO_Conversation_OnlineMeetings_DTEST_FY23Q1 | `nbno_conv_om_fy23q1` | 20.54 |
  | nb-NO | nb-NO_Dictation_DTEST_L_D_FY23Q4 | `nbno_dict_fy23q4` | 21.14 |
  | cs-CZ | average | — | 17.08 |
  | cs-CZ | cs-CZ_Conversation_DTEST_FY23Q2 | `cscz_conv_fy23q2` | 23.75 |
  | cs-CZ | cs-CZ_Conversation_OnlineMeetings_DTEST_FY24Q2 | `cscz_conv_om_fy24q2` | 14.42 |
  | cs-CZ | cs-CZ_Dictation_DTEST_L_D_FY24Q2 | `cscz_dict_fy24q2` | 13.07 |
  | overall | average | — | 20.05 |

  The source for a model column is the canonical eval directory layout
  `<root>/<slug>/measures.json` (each `measures.json` carries `dter`,
  `dter_n_err`, `dter_n_ref`). Pass it directly via `--model <label> <root>`
  — the script lists subdirs and reads each `measures.json`, locally or over
  `az://` (using `bbb ls` / `bbb cat`).

  Example — build the full 5-locale report from a single blob path:

  ```bash
  /home/boren/.virtualenvs/openai/bin/python \
    .github/skills/inhouse-dter-report/scripts/build_inhouse_dter_xlsx.py \
    --schema all_seg \
    --model "remax_r2_punc_p0_n12_s1k@step560" \
      az://orngwus2cresco/data/boren/data/verl/eval/remax_r2_punc_p0_n12_s1k_step560/inhouse_2605_5lang_seg_v2/ \
    --out tmp/inhouse_dter_report/remax_r2_punc_p0_n12_s1k_step560_all_seg.xlsx
  ```

  Stack multiple checkpoints into a single workbook (each its own column):

  ```bash
  /home/boren/.virtualenvs/openai/bin/python \
    .github/skills/inhouse-dter-report/scripts/build_inhouse_dter_xlsx.py \
    --schema all_seg \
    --model "step310" az://.../remax_r2_punc_p0_n12_s1k_step310/inhouse_2605_5lang_seg_v2/ \
    --model "step560" az://.../remax_r2_punc_p0_n12_s1k_step560/inhouse_2605_5lang_seg_v2/ \
    --out tmp/inhouse_dter_report/remax_r2_punc_p0_n12_s1k_sweep_all_seg.xlsx
  ```

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

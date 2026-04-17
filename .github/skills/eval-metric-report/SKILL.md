---
name: eval-metric-report
description: 'Collect p_err and p_edge metrics from ASR evaluation runs and generate an Excel report. Use when: generating eval metric Excel, comparing eval runs, collecting p_err p_edge from logs, merging evaluation results into spreadsheet, creating eval comparison table.'
argument-hint: 'Specify eval run names (log file basenames or eval config names) to include in the report'
---

# Eval Metric Report

Collect `p_err` and `p_edge` from ASR evaluation Ray job logs and produce an Excel workbook with:
- Separate sheets for `p_err` and `p_edge`
- Multiple runs merged into the same table (columns: `dataset`, `run_1`, `run_2`, …)
- Values as percentages (×100)
- An averaged row at the bottom
- A header row with experiment name and date

## When to Use
- After completing one or more ASR eval jobs and wanting a comparison spreadsheet
- When asked to "collect eval metrics", "compare eval runs", or "generate eval report"
- When the user specifies eval run names (log basenames or config names)

## Input: Metric Log Lines

Metrics live in Ray job log output. Two equivalent formats appear:

**Dict format** (printed by TaskRunner):
```
(TaskRunner pid=XXXX) {'val-aux/<dataset>/p_err/mean@1': 0.0176, 'val-aux/<dataset>/p_edge/mean@1': 0.0001, ...}
```

**Step format** (printed by TaskRunner):
```
(TaskRunner pid=XXXX) step:0 - val-aux/<dataset>/p_err/mean@1:0.0176 - val-aux/<dataset>/p_edge/mean@1:0.0001 - ...
```

Both lines contain all datasets evaluated in that run. Parse the **last `step:` line** in each log to get final metrics.

## Procedure

### 1. Identify Runs

The user provides run identifiers — either:
- **Log file basenames**: e.g., `eval_openasr_h100`, `eval_phimm_7b` (maps to `logs/<node>/<name>.log`)
- **Config names**: e.g., `recipe/phimm/config/eval_openasr_h100.yaml`

### 2. Gather Metric Lines

For each run, obtain the metric line by one of:

**A) Local log files** (preferred if available):
```bash
grep "step:" logs/<node>/<eval_name>.log | tail -1
```

**B) Remote fetch** via Ray job logs:
```bash
rcall-brix ssh <node> -- 'bash -lc "ray job logs <job_id> 2>&1 | grep \"step:\" | tail -1"'
```

If fetching remotely, save the full log locally for future use:
```bash
rcall-brix ssh <node> -- 'bash -lc "ray job logs <job_id> 2>&1"' > logs/<node>/<eval_name>.log
```

### 3. Parse Metrics

Use the [collect_eval_metrics.py](./scripts/collect_eval_metrics.py) script:

```bash
python .github/skills/eval-metric-report/scripts/collect_eval_metrics.py \
    --runs "eval_openasr_h100:logs/verl-n1-i0/eval_openasr_h100_full.log" \
           "eval_phimm_7b:logs/verl-n1-i0/eval_phimm_7b_full.log" \
    --output tmp/eval_metrics.xlsx
```

**Arguments:**
- `--runs`: Space-separated `<run_label>:<log_file_path>` pairs
- `--output`: Output Excel path (default: `tmp/eval_metrics.xlsx`)
- `--exp-name`: Experiment name for header (default: derived from output filename)
- `--date`: Date string for header (default: today)

Or pass metric lines directly via `--metric-lines`:
```bash
python .github/skills/eval-metric-report/scripts/collect_eval_metrics.py \
    --metric-lines "eval_run_1|step:0 - val-aux/librispeech/p_err/mean@1:0.0155 - ..." \
                   "eval_run_2|step:0 - val-aux/librispeech/p_err/mean@1:0.0177 - ..." \
    --output tmp/eval_metrics.xlsx
```

### 4. Output Format

The script produces an Excel workbook with two sheets:

**Sheet `p_err` (%):**
| | eval_openasr_h100 | eval_phimm_7b |
|---|---|---|
| **Exp** | eval_metrics | |
| **Date** | 2026-04-17 | |
| dataset | eval_openasr_h100 | eval_phimm_7b |
| librispeech | 1.55 | 1.77 |
| tedlium | 2.17 | — |
| ... | ... | ... |
| **average** | **5.00** | **1.77** |

**Sheet `p_edge` (%):** Same layout.

### 5. Verify

Open the Excel file and confirm:
- All runs appear as columns
- All datasets appear as rows
- Values are percentages (multiplied by 100)
- Average row is correct
- Header row shows experiment name and date

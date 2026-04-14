---
name: asr-experiment-report
description: "Fetch, analyze, and organize ASR experiment results end-to-end. Use when given an experiment name (e.g. en_ht_v1_egs, en_ht_v1_egs_cer02_err10) under a results root (az:// path). Discovers datasets with bbb ls, fetches report summaries with report_summary.py, runs per-dataset word-error analysis with the asr-word-error-analysis skill, and reshapes the Excel report with the excel-metric-analysis skill."
argument-hint: "experiment name(s) and results root path"
---

# ASR Experiment Report

End-to-end workflow: discover experiment datasets under an Azure results root, fetch summary reports, run detailed per-dataset word-error analysis, and organize metrics into comparison tables.

## When to Use

- You have one or more experiment names (e.g. `en_ht_v1_egs`, `en_ht_v1_egs_cer02_err10`) under a results root like `az://orngwus2cresco/data/boren/data/results/gpt-4o-mini-asr-v1/`
- You want a complete picture: summary Excel + per-dataset word-error HTML reports + organized metric comparison sheets
- You want to compare multiple experiments on the same eval sets

## Inputs

| Parameter | Required | Example |
|-----------|----------|---------|
| **results_root** | Yes | `az://orngwus2cresco/data/boren/data/results/gpt-4o-mini-asr-v1/` |
| **experiments** | Yes | `en_ht_v1_egs_step_5394`, `en_ht_v1_egs_cer02_err10_step_2369` |
| **report_dir** | No (default: `~/data/results/report/`) | `~/data/results/report/` |
| **analysis_dir** | No (default: `~/data/results/word_error_analysis/<root_name>/`) | custom output path |

## Procedure

### Phase 1 — Discover & Fetch Report Summaries

1. **Discover experiments** under the results root:
   ```bash
   bbb ls <results_root>
   ```
   Confirm each experiment directory exists (e.g. `<results_root>/en_ht_v1_egs_step_5394/`).

2. **List dataset groups** for each experiment:
   ```bash
   bbb ls <results_root>/<experiment>/
   ```
   Typically yields groups like `en-US-entity-v3/` and `openasr/`.

3. **List individual datasets** within each group:
   ```bash
   bbb ls <results_root>/<experiment>/<group>/
   ```
   Record the full dataset list (e.g. `openasr/ami`, `en-US-entity-v3/Banking`, etc.).

4. **Fetch report summary** for each experiment using `report_summary.py`:
   ```bash
   cd ~/code/openai/data/speech/speech/eval
   python report_summary.py <results_root>/<experiment>/
   ```
   This produces `~/data/results/report/<experiment>_<timestamp>.xlsx` and `.md`.

### Phase 2 — Per-Dataset Word Error Analysis

5. **Run word-error analysis** on every dataset in each experiment using the `asr-word-error-analysis` skill script. Use `--write-html` for visual reports. Skip datasets where `report.html` already exists:

   ```bash
   SCRIPT="$HOME/.github/skills/asr-word-error-analysis/scripts/analyze_word_errors.py"
   OUTBASE="$HOME/data/results/word_error_analysis/<root_name>"

   for each experiment × dataset:
       ds_flat=$(echo "$dataset" | tr '/' '_')
       outdir="$OUTBASE/$experiment/$ds_flat"
       [ -f "$outdir/report.html" ] && continue
       python "$SCRIPT" \
           --model "$experiment" \
           --dataset "$dataset" \
           --results-root "$results_root" \
           --output-dir "$outdir" \
           --write-html \
           --top-n 50 \
           --top-confusions 30
   done
   ```

6. **Verify completeness**: count `report.html` files per experiment — should match the dataset count (e.g. 22 = 8 openasr + 14 entity-v3).

### Phase 3 — Organize Metrics (Optional)

7. **Reshape the Excel report** into comparison tables using the `excel-metric-analysis` skill:
   ```bash
   python ~/.github/skills/excel-metric-analysis/scripts/build_metric_sheets.py \
       ~/data/results/report/<experiment>_<timestamp>.xlsx \
       --output-dir ~/data/results/report/analysis/
   ```

8. **Cross-experiment summary**: collect `summary.json` from all word-error analyses and produce a side-by-side WER comparison table:
   ```python
   import json, os
   base = os.path.expanduser("~/data/results/word_error_analysis/<root_name>")
   for experiment in experiments:
       for dataset_dir in sorted(os.listdir(os.path.join(base, experiment))):
           with open(os.path.join(base, experiment, dataset_dir, "summary.json")) as f:
               s = json.load(f)
           # s["wer"], s["total_substitutions"], s["total_deletions"], s["total_insertions"]
   ```

## Output Artifacts

| Phase | Artifact | Location |
|-------|----------|----------|
| 1 | Summary Excel + Markdown | `~/data/results/report/<experiment>_<ts>.xlsx` |
| 2 | Per-dataset analysis (8 files each) | `~/data/results/word_error_analysis/<root>/<exp>/<dataset>/` |
| 2 | HTML visual report | `.../report.html` per dataset |
| 2 | Error CSVs | `summary.json`, `error_details.csv`, `substitutions.csv`, `deletions.csv`, `insertions.csv`, `error_patterns.csv`, `alignment_samples.txt` |
| 3 | Metric comparison sheets | `~/data/results/report/analysis/` |

## Quality Checks

- Phase 1: confirm `summary_files` count matches expected datasets — no "No summary files found" warnings.
- Phase 2: count `report.html` per experiment matches total dataset count. Warn on audio-download failures (non-fatal — temp audio paths may not exist).
- Phase 3: spot-check WER values in comparison sheets against `summary.json`.

## Common Datasets

Under `gpt-4o-mini-asr-v1`, experiments typically contain:

**openasr** (8): ami, earnings22, gigaspeech, ls-test-clean, ls-test-other, spgispeech, tedlium, voxpopuli

**en-US-entity-v3** (14): Banking, CapitalMarket, DoctorPatientConsultation, Energy, Gaming, Insurance, K12HigherEdu, LifeHealth, Manufactory, Media, PatientHistoryDictation, Retail, ScienceTech, Sustain

## Dependent Skills

- **asr-word-error-analysis**: Phase 2 script for per-dataset error analysis
- **excel-metric-analysis**: Phase 3 script for metric comparison tables
- **asr-detail-compare**: Use separately when comparing two models on the same dataset utterance-by-utterance

---
name: excel-metric-analysis
description: Analyze Excel metric reports into comparison tables. Use when Codex needs to reshape a workbook with `name`, `group`, `wer`, and `eer` columns into a new `.xlsx` file with one worksheet per group and metric, named like group_WER or group_EER, with datasets as rows, tags as columns, and rows kept only when the target metric exists. Input names may be exact datasets or dataset_tag values.
---

# Excel Metric Analysis

Reshape one workbook into a compact analysis workbook without rewriting the pivot logic each time. Prefer the bundled script for repeatable conversions.

## Workflow
1. Inspect the source workbook just enough to confirm the sheet and the required columns: `name`, `group`, `wer`, `eer`.
2. Provide the dataset list explicitly when dataset names can contain underscores. Otherwise the script infers datasets from exact-name rows first, then falls back to splitting on the first underscore.
3. Run `scripts/build_metric_sheets.py` and point `--output-dir` at the desired analysis folder.
4. Spot-check a few cells in both output sheets against the source workbook before delivering.

## Script
Run:

```bash
python scripts/build_metric_sheets.py INPUT.xlsx --output-dir OUTPUT_DIR
```

Optional arguments:
- `--sheet SHEET_NAME`: read one worksheet instead of all sheets.
- `--group-column group`: override the column used for sheet partitioning.
- `--datasets-file datasets.txt`: disambiguate names when dataset strings include underscores.
- `--datasets ds1 ds2`: inline dataset list instead of a file.
- `--tag-for-exact base`: label rows where `name` exactly matches the dataset.
- `--output-name report.xlsx`: override the generated workbook name.
- `--keep-empty`: keep datasets even when a metric sheet has no value for that dataset.

## Output Shape
- The script writes one workbook with one sheet per group and metric.
- Sheet names follow `group_WER` and `group_EER`.
- Rows are datasets.
- Columns are tags.
- `name == dataset` is written under the default tag, `base` unless overridden.
- `name == <dataset>_<tag>` is split into dataset and tag.
- Rows with blank `wer` are excluded from `WERs`.
- Rows with blank `eer` are excluded from `EERs`.
- Missing dataset/tag combinations are left blank.

## Review Expectations
- Confirm the dataset order matches the explicit dataset list or the source first-seen order.
- Confirm tag columns remain in first-seen order.
- Call out ignored names if the workbook contains rows that could not be matched to a dataset.

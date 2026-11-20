# Excel comparison table for full evaluation

Apply when the report includes completed OpenASR/full evaluation results.
Deliver both the editable workbook and an image covering the requested comparison
scope by default. This is report generation from existing results; it never
requires launching or rerunning evaluation.

## Workbook and source identity

Use [openasr-report](../../openasr-report/SKILL.md) for the canonical workbook,
formulas, aggregation, metadata and exact result paths. Preserve the report's
chosen reference and model order. Explicitly override the workbook builder's
default baseline when the requested reference is another experiment. Record
checkpoint steps, code/scoring identities and coverage in the evidence.

Keep the canonical dataset order and OpenEN, OpenML and OpenLong sections in the source workbook,
including language means and group means. Preserve supplementary results in a
separate table/image and data file; do not insert them into canonical means.
Keep Chinese CER separate from WER. A missing result stays missing; do not fill
it from a baseline or infer completion from an existing workbook cell.

## OpenLong-focused experiment view

For OpenASR Long / OpenLong comparisons, use this presentation in Chat, the
PDF evaluation section, Markdown and note/homepage table embeds:

| Experiment / checkpoint | Earnings21 WER | Earnings22 WER | CORAAL WER | Average WER | WERR vs Baseline |
| --- | ---: | ---: | ---: | ---: | ---: |

- Each experiment/checkpoint is one row; each dataset is one column. Preserve
  experiment order, LR/PPO identity, checkpoint step and the selected baseline.
  Show only these three OpenLong datasets and the two summary columns; omit
  OpenEN, OpenML, OpenExt and auxiliary datasets from this view.
- Average WER is the equal arithmetic mean of the three unrounded full-dataset
  WERs. WERR is `1 - experiment_average_WER / baseline_average_WER`, formatted
  as a percentage: positive means improvement, negative means regression.
  This is the ratio of averages, not the average of dataset-level WERRs.
- Require matching three-dataset coverage for the summary comparison. Leave
  pending values and incomplete averages/WERR blank; never use zero, substitute
  another checkpoint or backfill from the baseline. Baseline WERR may be 0%.
- Stopping execution does not remove measured historical results. Retain their
  rows, mark stopped/provisional status and distinguish audited output artifacts
  from an unverified native job terminal state. Keep new unmeasured rows pending.
- Include an editable `openlong_comparison` view with the same orientation and
  formula cells, alongside the canonical source sheets. Save the exact rendering
  values and coverage/provenance in `data/`. Other suites may remain in raw
  artifacts without being displayed. Reuse this focused view in each delivery.

The focused view overrides the dataset-row rendering below; the canonical
worksheet contract remains the source-data format for broader full-suite reports.

## Render the table

- Save `comparison.xlsx`, `figures/evaluation_comparison_table.png` (or numbered
  parts), and workbook-derived rendering values under `data/` as JSON or CSV.
- Render dataset names, exact model/checkpoint headers, per-model percentages
  and one baseline-relative WERR column per target. Include every canonical
  dataset, language-average row and group-average row for broader full-suite
  reports; use the focused view above when the scope is OpenLong.
- Follow the workbook's layout: blue headers/sections, yellow language means,
  green group means, aligned numeric columns, positive WERR green, negative
  WERR red and zero/missing neutral. Use enough precision to expose small
  changes, normally four decimal places for WER/CER and two for WERR.
- WERR is `1 - target / reference`, expressed as a relative percentage, not a
  percentage-point delta. Preserve eligible arithmetic WERR under the workbook
  skill. Explain unequal budgets and normalized-reference denominator differences
  beside the image/in evidence; do not present these as paired error-count or
  exact-shared-step improvements. Do not add caveats to the canonical worksheet.
- Reopen the workbook and resolve formulas from its verified input cells when
  cached values are unavailable. Do not render blank formula caches as missing
  metrics, silently use zero, or manually retype values from a screenshot.

## Verify and deliver

Check every rendered value, formula result and summary against the workbook
and saved evidence, including matched-coverage and language-macro aggregation.
Inspect the image for clipping, legible labels, signs, percentages and colors.
Record the workbook, rendering-data and image paths in the evidence package.

Embed the table image(s) in Chat and `report.md`, link the `.xlsx`, and include
the readable table view in the PDF's Validation section. Split long/wide tables
by sections/pages and repeat model headers rather than shrinking text. Keep
the view complete across parts, omit redundant endpoint presentations, and
retain training trajectory/shared-step evidence separately when available.
Pass the workbook, images and data to `obsidian-research-notes` for the same
detailed note; let that skill select the compact daily-homepage figures.

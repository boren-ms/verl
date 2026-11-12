# Excel comparison table for full evaluation

Apply when the report includes completed OpenASR/full evaluation results.
Deliver both the editable workbook and an image of its complete comparison
table by default. This is report generation from existing results; it never
requires launching or rerunning evaluation.

## Workbook and source identity

Use [openasr-report](../../openasr-report/SKILL.md) for the canonical workbook,
formulas, aggregation, metadata and exact result paths. Preserve the report's
chosen reference and model order. Explicitly override the workbook builder's
default baseline when the requested reference is another experiment. Record
checkpoint steps, code/scoring identities and coverage in the evidence.

Keep the canonical dataset order and OpenEN, OpenML and OpenLong sections,
including language means and group means. Preserve supplementary results in a
separate table/image and data file; do not insert them into canonical means.
Keep Chinese CER separate from WER. A missing result stays missing; do not fill
it from a baseline or infer completion from an existing workbook cell.

## Render the table

- Save `comparison.xlsx`, `figures/evaluation_comparison_table.png` (or numbered
  parts), and workbook-derived rendering values under `data/` as JSON or CSV.
- Render dataset names, exact model/checkpoint headers, per-model percentages
  and one baseline-relative WERR column per target. Include every canonical
  dataset, language-average row and group-average row; a selected-row summary
  is not the full table view.
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

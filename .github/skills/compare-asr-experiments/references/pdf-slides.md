# Slide-style PDF reporting

Produce `report.pdf` as a presentation, using landscape 16:9 pages, alongside
`report.md`, `evidence.json`, an editable deck source (for example `slides.html`,
`slides.pptx`, or `slides.tex`), and `figures/`. Use available local rendering
tools; choose the format that reliably exports a real PDF in the environment.
For HTML, define print page dimensions and explicit slide page breaks, embed
assets, and hide navigation. Do not depend on remote fonts, scripts, or images
at export time. Follow the user's requested language and presentation format.

## Narrative and figures

Scale the slide count to the evidence rather than forcing a fixed length. Cover:

- Main findings: per-dataset improvements, regressions, and tradeoffs, plus the
  best-supported next action and material comparison limits.
- Run identity and coverage: run IDs, reference run, training budgets, resumed
  segments, compatible cohorts, and missing metric/example coverage.
- Validation: a trajectory figure for every observed dataset, followed by a
  compact initial/latest/best and latest shared-step comparison with actual steps.
  Split crowded tables or panels across pages. Missing datasets remain visible.
- Rewards: figures for available comparable components of `reward-core`,
  `reward-delta`, and `reward-vote`; show missing families explicitly. Put metrics
  with different units or formulas in separate panels or comparison cohorts.
- Validation examples: quantified pattern figures when counts exist, and readable
  matched reference/output excerpts with dataset, stable sample identity,
  run/step, and source pointer. Show denominators, selection scope, and overlapping
  labels; do not invent counts to fill a chart.
- Prioritized fixes: evidence, hypothesis and uncertainty, bounded next action,
  success metric, and other-dataset guardrails. Include provenance and supporting
  tables in appendix slides when they would crowd the main narrative.

Generate scientific charts from collected numeric evidence with a plotting
library such as Matplotlib. Save vector PDF/SVG figures when supported, plus PNG
previews when useful. Embed figures in the exported PDF; do not merely link to
them. If histories have only one point, show points or a labeled value comparison
instead of implying a trajectory. If quantitative evidence is absent, use an
explicit coverage slide rather than fabricated figures.

Keep run colors consistent across slides and identify runs in legends. Label
axes with actual training step and metric units. Display `p_err` as percent,
lower is better, and differences as percentage points with negative better.
Use visible markers for observed validation points; connecting lines are visual
guides, never extra observations. Preserve gaps, resumed segments, differing
budgets, and incompatible cohorts. Distinguish latest and independently best
values from matched-step results. If smoothing training metrics, disclose the
window/statistic and retain raw data visibility. Avoid dual axes that imply
relationships between unrelated metrics.

Each figure caption identifies the dataset or exact metric key, run/step scope,
units, relevant cohort limits, and its evidence source. Store a figure manifest
in `evidence.json` mapping figure filenames to exact keys, run IDs, steps,
filters/transforms, and source records so the displayed results are reproducible.

## Export and quality check

Use presentation-scale type (roughly 24 pt body and 18 pt chart labels on a
standard widescreen slide), concise takeaway titles, and enough whitespace for
charts and excerpts to be read at normal slide size. Split dense content instead
of shrinking it into report-sized text. Keep citations and short source pointers
on the relevant slide; retain full paths and metric inventories in the evidence.

After export, open or render the PDF pages and inspect for clipped text, tiny
labels, overlapping legends, missing glyphs/images, blank pages, and broken page
boundaries. Verify page count and landscape dimensions with a PDF reader/library,
and cross-check plotted values and table summaries against `evidence.json`.
Fix observed export defects before delivery. Keep the source and figures even
if rendering is blocked, and report the specific failure and missing PDF clearly.

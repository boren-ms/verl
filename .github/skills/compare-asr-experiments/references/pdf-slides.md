# Slide-style PDF reporting

Produce `report.pdf` as a presentation, using landscape 16:9 pages, alongside
`report.md`, `evidence.json`, an editable deck source (for example `slides.html`,
`slides.pptx`, or `slides.tex`), and `figures/`. Use available local rendering
tools; choose the format that reliably exports a real PDF in the environment.
For HTML, define print page dimensions and explicit slide page breaks, embed
assets, and hide navigation. Do not depend on remote fonts, scripts, or images
at export time. Follow the user's requested language and presentation format.

## Narrative and figures

Keep the deck concise and use this order:

1. **Key config/code changes:** compact before/after table, actual run revisions,
   source pointers, and expected effects; mark unverified behavior explicitly.
2. **Validation `p_err`:** per-dataset trajectory figures and a compact matched-step
   comparison. Keep initial/latest/best values with actual steps in captions or a
   small table rather than a separate slide for every dataset's checkpoint list.
3. **Training reward means:** component `*/mean` trajectories plus a shared-window
   mean comparison figure for comparable `reward-core`, `reward-delta`, and
   `reward-vote` metrics. State the window, arithmetic averaging, observation counts,
   and any smoothing; keep different units/formulas in separate panels.
4. **Brief bad-case analysis:** a few matched improvements/regressions with dataset,
   stable ID, reference/output excerpts, error change, and source pointer. State
   selection scope and count denominators; finish with a short supported next check.

Place run identity and material limits near the relevant figures. Keep detailed
inventories, provenance, and exhaustive examples in the supporting artifacts;
avoid extra background or lengthy recommendation sections unless requested.

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

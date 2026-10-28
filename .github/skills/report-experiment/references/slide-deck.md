# Slide-Deck Reporting Guidance

Prepare the experiment-specific content contract that
[ppt-master](../../ppt-master/SKILL.md) will turn into an editable native PPTX.
`report.md`, `evidence.json`, plotted data, and figures are the authoritative
inputs; `report.pptx` is the primary presentation deliverable. This reference
governs comparison content and traceability. PPT Master governs routing,
composition, SVG authoring, export, and visual QA. Do not prescribe a parallel
HTML-first implementation.

## Slide composition

Each slide communicates one main message:

- Write the title as a takeaway, not a topic label. Prefer “Run B lowers LS
  other `p_err` after step 240” over “Validation Results.”
- Use one dominant visual area: a chart, a compact table, or a small set of
  metric cards. Avoid layouts that give equal weight to many unrelated items.
- Keep body copy brief. Put methodology, exhaustive values, and provenance in
  the supporting artifacts and use only short source/caveat text on the slide.
- Keep a consistent grid, margins, typography, experiment colors, and footer.
- Split dense chart grids or settings tables across slides instead of shrinking
  text below presentation size.

Comparison slides place experiments in the same visual frame. Use overlaid
lines, grouped bars, or side-by-side table columns with a stable reference and
experiment order. Do not present one chart per run and ask the reader to compare
across slides.

Use a dark or light professional scientific theme with strong contrast. Avoid
decorative gradients, stock imagery, or animation that competes with the data.
Animations are optional and must not hide information needed to read the static
PPTX or any rendered PDF.

## Recommended sequence

1. **Title / executive takeaway** — experiment names, evidence date, and the
   strongest supported dataset-level conclusion.
2. **Scope and comparability** — run identity, observed step ranges, shared
   budget, and the top two or three limitations.
3. **Configuration differences** — comparison tables with fully qualified
   effective keys and one value column per experiment; split by
   model/training/data/reward/runtime when needed.
4. **Validation trajectories** — one or more chart slides grouped by related
   datasets, followed by exact shared-step comparisons.
5. **Reward-core trajectories** — compatible components grouped by unit and
   directionality, followed by a shared-window summary when supported.
6. **Findings and limits** — observed facts, supported interpretation,
   hypotheses, and evidence gaps.

Do not add agenda, background, or methodology slides unless they materially
help the requested comparison.

## Figures

Generate charts from collected numeric evidence with a scientific plotting
library. Save vector PDF/SVG figures when supported and PNG previews when
useful. Give ppt-master the actual saved figures and their source data; do not
substitute screenshots or visually similar redraws with different values.

- Keep one color per experiment across the entire deck.
- Overlay comparable experiments in the same dataset/component axes and use the
  same x/y scale for the same metric.
- Use actual optimizer step on every trajectory x-axis.
- Show observed validation points with visible markers; connecting lines are
  visual guides, not interpolated observations.
- Display `p_err` as percent and label lower-is-better.
- Preserve gaps, resume boundaries, unequal budgets, and incompatible cohorts.
- Separate `reward-core` components with different units, definitions, or
  directionality. State higher/lower-better only after scorer verification.
- If smoothing is useful, retain raw observations and name the method/window.
  Never smooth sparse validation points.
- If a history has one point, show a labeled point or value comparison rather
  than implying a trajectory.

Keep legends near the data, use readable axis labels, and avoid dual axes that
imply a relationship between unrelated metrics. Each chart slide includes a
one-line evidence caveat when budgets, cadence, or definitions differ.

Required figure families:

| Figure family | Required content |
| --- | --- |
| Validation trajectories | One panel per comparable dataset; all experiments, observed markers, actual optimizer steps |
| Reward-core trajectories | One panel per shared comparable component; raw observations, units, directionality, actual optimizer steps |
| Optional compact summary | Matched-step validation deltas or shared-window reward means, only when exact comparable evidence exists |

Do not combine validation and reward on one axis. Do not omit a comparable
dataset/component merely to make the deck shorter; split across additional
slides. If a family has no comparable numeric evidence, replace the missing
chart with a coverage table naming the absent keys/runs and reason.

## Tables and metric cards

Configuration tables use the same schema throughout the deck:

| Group | Effective config key | Experiment columns... | Difference / expected effect | Evidence |
| --- | --- | --- | --- | --- |

Rows are material effective settings; each experiment gets one value column.
Use fully qualified keys, compact normalized values, and a narrow evidence
column with config/run pointers. Highlight cells whose effective values differ.
Use `N/A` for a setting that does not apply, `Unknown` for unresolved evidence,
and `Same` only for essential shared controls. Never guess from filenames.

Keep the complete configuration-difference row set in
`data/config_differences.csv` or JSON. Slide tables may summarize long values,
but their evidence entries must point to the exact structured value. Split
wide tables by coherent groups rather than dropping experiment columns.

Validation summaries keep datasets separate and distinguish exact shared-step
values from unmatched latest/best values. Reward summaries state the shared
step window, arithmetic mean, observation count, and directionality. Do not
average incomparable components.

The reference experiment is explicit in the title/header. For every other
experiment, pair its value column with a delta-versus-reference column. Use:

- validation delta: target minus reference in percentage points at the exact
  shared step;
- reward delta: target minus reference in the metric's native unit over the
  exact shared window;
- `N/A` when the exact step/window or metric definition is not shared.

Use color carefully: experiment colors identify value columns/lines; a separate
zero-centered improvement/regression cue may highlight deltas only when metric
direction is known. Always show the signed number so meaning does not depend on
color.

Use metric cards sparingly for a small number of high-value facts such as
shared step coverage, best observed dataset value, or a clearly defined
matched-step delta. Cards never replace trajectory evidence.

Required tables:

1. **Run identity and coverage:** run ID, state, code revision, observed
   optimizer-step range, validation-step range, and evidence timestamp.
2. **Material configuration difference:** columns for group, fully qualified
   key, each experiment, difference/expected effect, and evidence; one row per
   differing setting plus essential shared controls.
3. **Validation comparison:** dataset, exact shared step, reference value, each
   experiment value and delta in percentage points, plus separately labeled
   first/latest/best values and steps.
4. **Reward comparison:** exact `reward-core` key, definition/unit,
   directionality, shared window, reference mean/count, and each experiment's
   mean/count and delta; incompatible keys remain in coverage rows.

Split a table across slides by coherent row groups when it cannot remain
readable. Repeat the header and experiment-color cues on each continuation;
never shrink the complete table into unreadable text.

## Reproducibility

Every figure caption or footer identifies the exact metric key, run IDs, step
scope, units, and relevant cohort limit. In `evidence.json`, map each slide and
figure filename to its source keys, raw data file, filters, transformations, and
plotted step range.

Save the exact numeric rows behind every chart in `data/` and the structured
rows behind every table in `evidence.json` or a referenced CSV/JSON file. The
slide deck is a view of those artifacts, not an independent source of truth.

The ppt-master source brief maps every slide, figure, table, and takeaway to its
supporting evidence. Keep all local inputs available to its project workspace
and record the resulting project and PPTX paths in `evidence.json`. An optional
HTML/PDF rendering is a view of the verified PPTX, not a separately authored
report.

## Visual QA

Use presentation-scale text—roughly 24 pt body and at least 18 pt chart labels
on a standard widescreen slide. Let ppt-master apply its own typography and
quality gates. After native PPTX export, inspect its rendered pages or contact
sheet when the environment provides one and check:

- 16:9 landscape dimensions and one slide per page;
- clipped/overlapping text, legends, tables, or footers;
- tiny labels, missing glyphs, blank pages, and missing images;
- consistent experiment colors and correct metric-direction labels;
- agreement between plotted values, slide claims, `evidence.json`, and source
  data.

Fix observed defects through ppt-master's owning SVG/resource inputs and rerun
its required quality check and export. If rendering or optional PDF export is
blocked, still return the verified native PPTX, Markdown, evidence, data, and
figures while stating exactly which optional rendering step failed. If the PPTX
itself was not produced and verified, return only the evidence package and mark
the presentation handoff incomplete.

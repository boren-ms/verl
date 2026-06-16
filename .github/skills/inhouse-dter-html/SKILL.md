---
name: inhouse-dter-html
description: Turn an in-house DTER xlsx comparison report (sheets `inhouse_dter` + `overall_improve_degrade`, produced by the `inhouse-dter-report` skill) into a self-contained Chart.js HTML visualization. Use when summarizing an inhouse-DTER xlsx as bar charts, generating a single-file HTML report next to the xlsx, showing per-locale WERR averages plus per-dataset WERR deltas vs baseline, and rendering the overall improve/degrade ranking with full data tables. Triggers "html report for inhouse xlsx", "use chart for inhouse_dter", "visualize DTER report", "chart-based dter report".
argument-hint: '<path-to-xlsx> [--out <path-to-html>] [--baseline-label <name>] [--title <text>]'
---

# In-house DTER HTML Chart Report

Generate a single self-contained `.html` next to an in-house DTER xlsx that visualizes its checkpoints/models as Chart.js charts plus full data tables. The xlsx layout is the one produced by the `inhouse-dter-report` skill (`inhouse_dter` + `overall_improve_degrade` sheets).

## When to Use

- The user asks for a "chart" or "HTML report" for an `inhouse_dter` xlsx.
- A training run has been summarized into an xlsx with multiple checkpoint columns (e.g. `step200`, `step400`, ..., `step1000`) and you want a visual story of how DTER evolves.
- Multiple models were appended to the xlsx as columns (B, C, D, ...) and you want a side-by-side comparison report.

## Inputs

Required: path to the xlsx (`*.xlsx`) under `tmp/inhouse_dter_report/` or anywhere else.

Optional:
- `--out <path>` — output html path. Default: same dir/basename as the xlsx with `.html` extension.
- `--baseline-label <name>` — label shown for column A. Default: `Qwen3.5-audio`.
- `--title <text>` — header title. Default derived from xlsx basename.

If only an xlsx path is given, infer everything and proceed.

Run-specific target for this request:
- Input xlsx: `tmp/inhouse_dter_report/remax_r2_punc_p0_7_n12_s200_mean_step200_all_seg.xlsx`
- Output html: `tmp/inhouse_dter_report/remax_r2_punc_p0_7_n12_s200_mean_step200_all_seg.html`
- Example: `python .github/skills/inhouse-dter-html/scripts/build_inhouse_dter_html.py tmp/inhouse_dter_report/remax_r2_punc_p0_7_n12_s200_mean_step200_all_seg.xlsx --out tmp/inhouse_dter_report/remax_r2_punc_p0_7_n12_s200_mean_step200_all_seg.html`

Visual parity target:
- When this run-specific target is used, generate an html with the same layout style and section ordering as `tmp/inhouse_dter_report/remax_r2_punc_p0_7_n12_s200_mean_step200_all_seg.html` (header + TL;DR, Section 1..5 charts/tables/takeaways).
- Keep the page self-contained and preserve the same CSS theme tokens and Chart.js CDN usage from this skill.

## XLSX Schema Expected

Sheet `inhouse_dter`:
- Row 2 `Header`: `Baseline`, `<model-1>`, `<model-2>`, ..., then one `WERR` per non-baseline model.
- Row 3 `Column`: `A`, `B`, `C`, ..., `A->B`, `A->C`, ...
- Rows 4..N: per-dataset rows; locale-average rows have name ending in ` avg` (e.g. `en-US avg`); the final row is `overall avg`.
- WERR cells may be **formulas** like `=1-C4/B4` (not values). On `avg` rows they are numeric.

Sheet `overall_improve_degrade`:
- Header in row 2: `Rank`, `Model`, `Direction`, `Baseline overall DTER`, `Model overall DTER`, `DTER delta`, `WERR`, `Datasets`.
- Rows sorted by overall WERR (best improvement first).

## Locale Derivation

For each non-`avg` row, locale = the first underscore segment of the dataset name (e.g. `en-US_Conversation_DTEST_FY21Q1` → `en-US`). Locale-average rows have names like `en-US avg`, `nl-NL avg`, etc. The skill must not hardcode the set of locales — derive them from the xlsx.

## Procedure

### 1. Load the xlsx

Use `openpyxl.load_workbook(..., data_only=False)` so formula strings are preserved. For each dataset row collect:
- `name`, `lang` (first underscore segment), `baseline` (col B value).
- `dter[]` = [baseline, model_1, model_2, ...] across all model columns.
- `werr[]` = one entry per model. If the WERR cell is numeric, use it; if it is a formula or `None`, compute `1 - model/baseline` from the corresponding value column.

Classify each row:
- Ends with ` avg` and equals `overall avg` → overall row.
- Ends with ` avg` → locale average.
- Otherwise → dataset row (assign to its locale).

Load the ranking sheet rows as-is.

### 2. Build the HTML

Write a single self-contained file using **Chart.js 4 from CDN** (`https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js`). Layout (top to bottom):

1. **Header**
   - Title (paper-reading style: light card, blue left accent border).
   - Meta line: source xlsx path, baseline label, dataset count, locale count.
   - **TL;DR box** (`background:#f0f9ff; border-left:4px solid #1a6fb5`) summarizing in 2–3 lines: best model, overall WERR, and best/worst locale behavior. Use `.hl-green` for gains, `.hl-red` for regressions, `.hl-orange` for method/checkpoint names.

2. **Section 1 — Per-locale WERR averages**
   - Bar chart with grouped bars per locale: one bar per model showing WERR (Word Error Rate Reduction).
   - Y axis is percent (signed, can be positive or negative) and should auto-range (chart-specific min/max).
   - Show bar value labels in `.2%` format with sign indicator.

3. **Section 2 — Dataset WERR Delta Vs Baseline (Pre-Locale Charts)**
   - Bar chart over all datasets, ordered locale-by-locale.
   - Bar values are dataset-level WERR deltas (`1 - model/baseline`).
   - Bars colored by locale (`LANG_COLORS`; fallback gray).
   - Show bar value labels in `.2%` format.
   - Use a **two-layer custom x-axis tick renderer**:
     - Top layer: dataset names, colored by locale.
     - Bottom layer: locale names, bold and colored by locale, rendered once per contiguous locale group (do not duplicate locale text per dataset bar).

4. **Section 3 — Overall improve / degrade ranking table**
   - Render the `overall_improve_degrade` sheet as a table.
   - Color the `Direction` cell with a pill: green pill for `improve`, red pill for `degrade`.
   - Format DTER/WERR cells as percentages; signed for WERR/delta.

5. **Section 4 — Full per-dataset tables**
   - Full DTER table: rows ordered locale-by-locale (datasets first, then `<lang> avg` highlighted yellow), final `overall avg` highlighted green.
   - WERR table immediately below with the same row ordering; cells colored red/green based on sign.

6. **Section 5 — Takeaways**
   - 4–6 bullets that cite specific numbers: best checkpoint, locale ranking of degradation, and suggested next step.
   - Use `.hl-*` spans throughout.

### 3. Styling (must match)

```css
:root { --blue:#1a6fb5; --red:#dc2626; --green:#16a34a; --orange:#d97706; --gray:#475569; --bg:#f8fafc; --border:#e2e8f0; }
body { font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
       background: var(--bg); padding: 28px; color:#0f172a; }
.container { max-width: 1280px; margin: 0 auto; }
header { padding:18px 24px; background:white; border-left:5px solid var(--blue);
         border-radius:6px; margin-bottom:24px; box-shadow:0 1px 3px rgba(0,0,0,.06); }
.tldr { background:#f0f9ff; border-left:4px solid var(--blue); padding:16px 22px;
        margin-top:14px; line-height:1.85; border-radius:4px; font-size:14px; }
section { background:white; padding:18px 22px; border-radius:6px; margin-bottom:22px;
          box-shadow:0 1px 3px rgba(0,0,0,.06); }
section h2 { color:var(--blue); border-bottom:2px solid var(--blue); padding-bottom:6px;
             margin:0 0 12px; font-size:17px; }
.chart-box { position:relative; height:360px; margin:10px 0; }
.chart-box.tall { height:460px; }
.chart-grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
.chart-grid .chart-box { height:240px; }
table { width:100%; border-collapse:collapse; font-size:12px; margin-top:8px; }
th, td { padding:5px 8px; border-bottom:1px solid var(--border); text-align:right; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
th { background:#dbeafe; color:#1e3a8a; }
tr.lang-avg { background:#fef9c3; font-weight:600; }
tr.overall  { background:#dcfce7; font-weight:700; }
.neg { color:var(--red); } .pos { color:var(--green); }
.hl-red { color:var(--red); font-weight:700; }
.hl-green { color:var(--green); font-weight:700; }
.hl-orange { color:var(--orange); font-weight:700; }
.legend-tag { display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:700; }
.tag-improve { background:#dcfce7; color:#15803d; }
.tag-degrade { background:#fee2e2; color:#b91c1c; }
```

### 4. Locale color palette (deterministic)

```js
const LANG_COLORS = {
  'en-US':'#16a34a', 'nl-NL':'#dc2626', 'da-DK':'#d97706',
  'hu-HU':'#7c3aed', 'nb-NO':'#0ea5e9', 'cs-CZ':'#be185d',
  'fr-FR':'#0891b2', 'de-DE':'#a16207', 'es-ES':'#9333ea',
  'it-IT':'#65a30d', 'ja-JP':'#db2777', 'ko-KR':'#0d9488'
};
```
For unknown locales: `'#475569'`.

### 5. Formatters

```js
const fmtPct    = v => v == null ? '' : (v*100).toFixed(2) + '%';
const fmtSigned = v => v == null ? '' : (v >= 0 ? '+' : '') + (v*100).toFixed(2) + '%';
```

### 6. Injection pattern

Build a Python dict `{steps, datasets, lang_avgs, overall, ranking}`, `json.dumps` it, and substitute via a single placeholder in the HTML template:

```python
html = TEMPLATE.replace('__DATA__', json.dumps(data))
```

Do **not** use Python f-strings for the whole template (the JS / CSS contain `{` / `}` that would break formatting). Use simple `str.replace` for a single `__DATA__` token.

### 7. Write the output and report

- Default output path: same directory as the xlsx, same basename, `.html` extension.
- After writing, print `Wrote <path> <bytes>` and confirm `__DATA__` placeholder no longer remains in the file.
- Reply to the user with the markdown-linked html path, the sections it contains, and a 2–3 line synthesis of the actual numbers (best checkpoint, worst locale, where collapse begins).

## Quality Checklist

- [ ] File is **self-contained** (one html, only CDN fetch is Chart.js).
- [ ] Locales are derived from the xlsx, not hardcoded.
- [ ] WERR computed from values when the xlsx cell is a formula string.
- [ ] Per-locale DTER section is bar-based and includes bar labels in `.2%` format.
- [ ] Dataset WERR-delta section is bar-based and bars are colored by locale.
- [ ] Dataset WERR-delta x-axis is two-layer: dataset (top) + locale (bottom), with locale shown once per locale group.
- [ ] Locale-avg rows in tables highlighted yellow; `overall avg` highlighted green.
- [ ] Ranking table preserves the order from `overall_improve_degrade` and renders direction as a colored pill.
- [ ] TL;DR + Takeaways cite specific numbers and use `.hl-red` / `.hl-green` / `.hl-orange` spans.
- [ ] Output written to `<xlsx-stem>.html` next to the input by default.

## Anti-patterns

- ❌ Reading the xlsx with `data_only=True` — WERR cells are formulas and will come back as `None`.
- ❌ Hardcoding "step200/step400/..." or the en-US/nl-NL/... locale set — the same xlsx schema is reused for many runs.
- ❌ Using Python f-strings to assemble the HTML — `{` in CSS/JS will explode. Use `str.replace('__DATA__', ...)`.
- ❌ Embedding Chart.js source inline — keep the CDN script tag; the file should stay small (~25 KB).
- ❌ Writing a separate `_data.json` sidecar — the HTML must be a single file.
- ❌ Duplicating locale text on every dataset tick in the pre-locale WERR chart.
- ❌ Leaving the dataset x-axis as a single flat label line when two-layer locale+dataset ticks are required.
- ❌ Omitting the TL;DR / Takeaways with concrete numbers — a bare chart dump is not useful.

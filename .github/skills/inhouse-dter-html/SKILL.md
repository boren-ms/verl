---
name: inhouse-dter-html
description: Turn an in-house DTER xlsx comparison report (sheets `inhouse_dter` + `overall_improve_degrade`, produced by the `inhouse-dter-report` skill) into a self-contained Chart.js HTML visualization. Use when summarizing a multi-checkpoint or multi-model inhouse-DTER xlsx as charts, generating a single-file HTML report next to the xlsx, visualizing per-locale DTER/WERR trajectories across training steps, plotting small-multiple panels per locale, or rendering the overall improve/degrade ranking with charts and full data tables. Triggers "html report for inhouse xlsx", "use chart for inhouse_dter", "visualize DTER report", "chart-based dter report".
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
   - **TL;DR box** (`background:#f0f9ff; border-left:4px solid #1a6fb5`) summarizing in 2–3 lines: best model, overall WERR range, which locales collapse/improve fastest. Use `.hl-green` for gains, `.hl-red` for regressions, `.hl-orange` for method/checkpoint names.

2. **Section 1 — Overall DTER + WERR** (2 charts side by side in a 2-column CSS grid)
   - Left: line chart of overall DTER across `[baseline, model_1, ..., model_N]`. Y axis percent.
   - Right: bar chart of overall WERR per model. Bars colored green if `>=0`, red if `<0`.

3. **Section 2 — Per-locale averages**
   - Tall line chart: one series per locale of DTER across all columns (baseline first).
   - Line chart: one series per locale of WERR across all model columns.
   - Use a deterministic locale color palette; if a locale isn't in the palette, fall back to gray.

4. **Section 3 — Per-locale small multiples**
   - CSS grid (2 columns). One panel per locale.
   - Each panel: line chart with one series per dataset in that locale + a dashed "locale avg" series.
   - Y axis fixed to `[0, 1.0]` so degradation magnitude is comparable across panels.

5. **Section 4 — Overall improve / degrade ranking table**
   - Render the `overall_improve_degrade` sheet as a table.
   - Color the `Direction` cell with a pill: green pill for `improve`, red pill for `degrade`.
   - Format DTER/WERR cells as percentages; signed for WERR/delta.

6. **Section 5 — Full per-dataset tables**
   - Full DTER table: rows ordered locale-by-locale (datasets first, then `<lang> avg` highlighted yellow), final `overall avg` highlighted green.
   - WERR table immediately below with the same row ordering; cells colored red/green based on sign.

7. **Section 6 — Takeaways**
   - 4–6 bullets that cite specific numbers: best checkpoint, where collapse begins, locale ranking of degradation, whether late steps plateau, suggested next step (e.g. early stop, language-coverage penalty). Use `.hl-*` spans throughout.

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
- [ ] All N model columns from the xlsx are visualized — do not hardcode 5 steps.
- [ ] Locales are derived from the xlsx, not hardcoded.
- [ ] WERR computed from values when the xlsx cell is a formula string.
- [ ] Baseline value is shown as the first point on every DTER line chart.
- [ ] WERR bars use green for `>=0`, red for `<0`.
- [ ] Per-locale small multiples y-axis fixed to `[0, 1.0]`.
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
- ❌ Skipping the small-multiples section — per-locale panels are the most informative view of catastrophic forgetting.
- ❌ Omitting the TL;DR / Takeaways with concrete numbers — a bare chart dump is not useful.

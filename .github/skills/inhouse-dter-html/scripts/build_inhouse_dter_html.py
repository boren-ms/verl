#!/usr/bin/env python3
"""Build a self-contained Chart.js HTML report from an in-house DTER xlsx."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openpyxl import load_workbook


HEADER_ROW = 2
COLUMN_ROW = 3
DATA_START = 4
LANG_COLORS = {
    "en-US": "#16a34a",
    "nl-NL": "#dc2626",
    "da-DK": "#d97706",
    "hu-HU": "#7c3aed",
    "nb-NO": "#0ea5e9",
    "cs-CZ": "#be185d",
    "fr-FR": "#0891b2",
    "de-DE": "#a16207",
    "es-ES": "#9333ea",
    "it-IT": "#65a30d",
    "ja-JP": "#db2777",
    "ko-KR": "#0d9488",
}


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>In-house DTER Report</title>
  <style>
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
.meta { color:#334155; font-size:13px; line-height:1.7; }
.two-col { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
.small-multiples { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
.panel { border:1px solid var(--border); border-radius:6px; padding:12px; }
.panel h3 { margin:0 0 8px; color:#0f172a; font-size:14px; }
ul.takeaways { margin:0; padding-left:20px; line-height:1.8; }
.muted { color:#64748b; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
@media (max-width: 960px) {
  .chart-grid, .two-col, .small-multiples { grid-template-columns:1fr; }
  body { padding: 16px; }
}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1 id="report-title"></h1>
      <div class="meta" id="report-meta"></div>
      <div class="tldr" id="report-tldr"></div>
    </header>

    <section>
      <h2>Section 1 — Overall DTER + WERR</h2>
      <div class="chart-grid">
        <div class="chart-box"><canvas id="overall-dter-chart"></canvas></div>
        <div class="chart-box"><canvas id="overall-werr-chart"></canvas></div>
      </div>
    </section>

    <section>
      <h2>Section 2 — Per-locale averages</h2>
      <div class="chart-box tall"><canvas id="locale-dter-chart"></canvas></div>
      <div class="chart-box tall"><canvas id="locale-werr-chart"></canvas></div>
    </section>

    <section>
      <h2>Section 3 — Per-locale small multiples</h2>
      <div class="small-multiples" id="small-multiples"></div>
    </section>

    <section>
      <h2>Section 4 — Overall improve / degrade ranking table</h2>
      <table id="ranking-table"></table>
    </section>

    <section>
      <h2>Section 5 — Full per-dataset tables</h2>
      <div id="dter-table-wrap"></div>
      <div id="werr-table-wrap"></div>
    </section>

    <section>
      <h2>Section 6 — Takeaways</h2>
      <ul class="takeaways" id="takeaways"></ul>
    </section>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
  <script>
const DATA = __DATA__;
const LANG_COLORS = {
  'en-US':'#16a34a', 'nl-NL':'#dc2626', 'da-DK':'#d97706',
  'hu-HU':'#7c3aed', 'nb-NO':'#0ea5e9', 'cs-CZ':'#be185d',
  'fr-FR':'#0891b2', 'de-DE':'#a16207', 'es-ES':'#9333ea',
  'it-IT':'#65a30d', 'ja-JP':'#db2777', 'ko-KR':'#0d9488'
};
const fmtPct = v => v == null ? '' : (v * 100).toFixed(2) + '%';
const fmtSigned = v => v == null ? '' : (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%';

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function colorForLang(lang) {
  return LANG_COLORS[lang] || '#475569';
}

function alpha(hex, opacity) {
  const clean = hex.replace('#', '');
  const r = Number.parseInt(clean.slice(0, 2), 16);
  const g = Number.parseInt(clean.slice(2, 4), 16);
  const b = Number.parseInt(clean.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${opacity})`;
}

function datasetStroke(index) {
  const cycle = [[], [6, 4], [2, 3], [8, 3], [3, 2, 1, 2]];
  return cycle[index % cycle.length];
}

function localeStats() {
  return DATA.lang_order.map((lang) => {
    const avg = DATA.lang_avgs[lang];
    const werr = avg.werr || [];
    const best = werr.length ? Math.max(...werr) : null;
    const worst = werr.length ? Math.min(...werr) : null;
    const collapseIdx = werr.findIndex((value) => value != null && value < 0);
    return {
      lang,
      best,
      worst,
      collapseIdx,
      collapseStep: collapseIdx >= 0 ? DATA.models[collapseIdx] : null,
    };
  });
}

function buildHeader() {
  const overallWerrs = DATA.overall.werr.filter((value) => value != null);
  const bestIdx = overallWerrs.reduce((best, value, index, arr) => {
    return arr[best] > value ? best : index;
  }, 0);
  const worstIdx = overallWerrs.reduce((worst, value, index, arr) => {
    return arr[worst] < value ? worst : index;
  }, 0);
  const bestModel = DATA.models[bestIdx];
  const worstModel = DATA.models[worstIdx];
  const stats = localeStats();
  const bestLocale = stats.reduce((acc, item) => (acc == null || (item.best ?? -Infinity) > (acc.best ?? -Infinity)) ? item : acc, null);
  const worstLocale = stats.reduce((acc, item) => (acc == null || (item.worst ?? Infinity) < (acc.worst ?? Infinity)) ? item : acc, null);
  document.getElementById('report-title').textContent = DATA.title;
  document.getElementById('report-meta').innerHTML = [
    `Source xlsx: <code>${escapeHtml(DATA.source)}</code>`,
    `Baseline label: <span class="hl-orange">${escapeHtml(DATA.baseline_label)}</span>`,
    `Dataset count: <strong>${DATA.dataset_count}</strong>`,
    `Locale count: <strong>${DATA.lang_order.length}</strong>`
  ].join(' · ');
  document.getElementById('report-tldr').innerHTML = [
    `Best overall model is <span class="hl-orange">${escapeHtml(bestModel)}</span> at <span class="hl-green">${fmtSigned(overallWerrs[bestIdx])}</span> WERR, while the weakest is <span class="hl-orange">${escapeHtml(worstModel)}</span> at <span class="hl-red">${fmtSigned(overallWerrs[worstIdx])}</span>.`,
    `Overall WERR spans from <span class="hl-red">${fmtSigned(Math.min(...overallWerrs))}</span> to <span class="hl-green">${fmtSigned(Math.max(...overallWerrs))}</span> across ${DATA.models.length} checkpoints/models.`,
    `${bestLocale ? `Fastest locale gain is <span class="hl-green">${escapeHtml(bestLocale.lang)}</span> at <span class="hl-green">${fmtSigned(bestLocale.best)}</span>` : ''}${bestLocale && worstLocale ? ', ' : ''}${worstLocale ? `and the sharpest collapse is <span class="hl-red">${escapeHtml(worstLocale.lang)}</span> at <span class="hl-red">${fmtSigned(worstLocale.worst)}</span>` : ''}.`
  ].join('<br>');
}

function makeChart(id, config) {
  return new Chart(document.getElementById(id), config);
}

function buildOverallCharts() {
  const labels = [DATA.baseline_label].concat(DATA.models);
  makeChart('overall-dter-chart', {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Overall DTER',
        data: DATA.overall.dter,
        borderColor: '#1a6fb5',
        backgroundColor: 'rgba(26,111,181,0.15)',
        borderWidth: 3,
        tension: 0.2,
        fill: false,
      }]
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { y: { ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` } } }
    }
  });
  makeChart('overall-werr-chart', {
    type: 'bar',
    data: {
      labels: DATA.models,
      datasets: [{
        label: 'Overall WERR',
        data: DATA.overall.werr,
        backgroundColor: DATA.overall.werr.map((value) => value >= 0 ? '#16a34a' : '#dc2626'),
      }]
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { y: { ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` } } }
    }
  });
}

function buildLocaleCharts() {
  const labels = [DATA.baseline_label].concat(DATA.models);
  makeChart('locale-dter-chart', {
    type: 'line',
    data: {
      labels,
      datasets: DATA.lang_order.map((lang) => ({
        label: lang,
        data: DATA.lang_avgs[lang].dter,
        borderColor: colorForLang(lang),
        backgroundColor: alpha(colorForLang(lang), 0.12),
        tension: 0.15,
      }))
    },
    options: {
      maintainAspectRatio: false,
      scales: { y: { ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` } } }
    }
  });
  makeChart('locale-werr-chart', {
    type: 'line',
    data: {
      labels: DATA.models,
      datasets: DATA.lang_order.map((lang) => ({
        label: lang,
        data: DATA.lang_avgs[lang].werr,
        borderColor: colorForLang(lang),
        backgroundColor: alpha(colorForLang(lang), 0.12),
        tension: 0.15,
      }))
    },
    options: {
      maintainAspectRatio: false,
      scales: { y: { ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` } } }
    }
  });
}

function buildSmallMultiples() {
  const wrap = document.getElementById('small-multiples');
  const labels = [DATA.baseline_label].concat(DATA.models);
  DATA.lang_order.forEach((lang, langIndex) => {
    const panel = document.createElement('div');
    panel.className = 'panel';
    const title = document.createElement('h3');
    title.textContent = `${lang} (${DATA.datasets_by_lang[lang].length} datasets)`;
    const chartBox = document.createElement('div');
    chartBox.className = 'chart-box';
    chartBox.style.height = '280px';
    const canvas = document.createElement('canvas');
    canvas.id = `lang-chart-${langIndex}`;
    chartBox.appendChild(canvas);
    panel.appendChild(title);
    panel.appendChild(chartBox);
    wrap.appendChild(panel);
    const baseColor = colorForLang(lang);
    const datasets = DATA.datasets_by_lang[lang].map((row, index) => ({
      label: row.name,
      data: row.dter,
      borderColor: alpha(baseColor, Math.max(0.35, 0.85 - index * 0.1)),
      backgroundColor: alpha(baseColor, 0.08),
      tension: 0.15,
      borderDash: datasetStroke(index),
      borderWidth: 2,
    }));
    datasets.push({
      label: `${lang} avg`,
      data: DATA.lang_avgs[lang].dter,
      borderColor: baseColor,
      backgroundColor: alpha(baseColor, 0.12),
      tension: 0.15,
      borderDash: [10, 4],
      borderWidth: 3,
    });
    new Chart(canvas, {
      type: 'line',
      data: { labels, datasets },
      options: {
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 0,
            max: 1.0,
            ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` }
          }
        }
      }
    });
  });
}

function buildRankingTable() {
  const table = document.getElementById('ranking-table');
  const headers = ['Rank', 'Model', 'Direction', 'Baseline overall DTER', 'Model overall DTER', 'DTER delta', 'WERR', 'Datasets'];
  const head = `<thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join('')}</tr></thead>`;
  const body = DATA.ranking.map((row) => {
    const tagClass = row.direction === 'improve' ? 'tag-improve' : 'tag-degrade';
    return `<tr>
      <td>${row.rank}</td>
      <td>${escapeHtml(row.model)}</td>
      <td><span class="legend-tag ${tagClass}">${escapeHtml(row.direction)}</span></td>
      <td>${fmtPct(row.baseline_overall_dter)}</td>
      <td>${fmtPct(row.model_overall_dter)}</td>
      <td class="${row.dter_delta >= 0 ? 'pos' : 'neg'}">${fmtSigned(row.dter_delta)}</td>
      <td class="${row.werr >= 0 ? 'pos' : 'neg'}">${fmtSigned(row.werr)}</td>
      <td>${escapeHtml(row.datasets)}</td>
    </tr>`;
  }).join('');
  table.innerHTML = head + `<tbody>${body}</tbody>`;
}

function rowClass(kind) {
  if (kind === 'overall') return 'overall';
  if (kind === 'lang_avg') return 'lang-avg';
  return '';
}

function buildDataTables() {
  const dterHeaders = ['Dataset', DATA.baseline_label].concat(DATA.models);
  const dterHead = `<thead><tr>${dterHeaders.map((header) => `<th>${escapeHtml(header)}</th>`).join('')}</tr></thead>`;
  const dterBody = DATA.ordered_rows.map((row) => {
    return `<tr class="${rowClass(row.kind)}"><td>${escapeHtml(row.name)}</td>${row.dter.map((value) => `<td>${fmtPct(value)}</td>`).join('')}</tr>`;
  }).join('');
  document.getElementById('dter-table-wrap').innerHTML = `<div class="muted">Full DTER table</div><table>${dterHead}<tbody>${dterBody}</tbody></table>`;

  const werrHeaders = ['Dataset'].concat(DATA.models);
  const werrHead = `<thead><tr>${werrHeaders.map((header) => `<th>${escapeHtml(header)}</th>`).join('')}</tr></thead>`;
  const werrBody = DATA.ordered_rows.map((row) => {
    return `<tr class="${rowClass(row.kind)}"><td>${escapeHtml(row.name)}</td>${row.werr.map((value) => {
      const cls = value == null ? '' : (value >= 0 ? 'pos' : 'neg');
      return `<td class="${cls}">${fmtSigned(value)}</td>`;
    }).join('')}</tr>`;
  }).join('');
  document.getElementById('werr-table-wrap').innerHTML = `<div class="muted" style="margin-top:18px;">Full WERR table</div><table>${werrHead}<tbody>${werrBody}</tbody></table>`;
}

function buildTakeaways() {
  const items = [];
  const overall = DATA.overall.werr.map((value, index) => ({ model: DATA.models[index], werr: value, dter: DATA.overall.dter[index + 1] }));
  const best = overall.reduce((acc, item) => acc == null || item.werr > acc.werr ? item : acc, null);
  const worst = overall.reduce((acc, item) => acc == null || item.werr < acc.werr ? item : acc, null);
  const collapse = overall.find((item) => item.werr < 0);
  const localeSummary = localeStats().sort((a, b) => (b.best ?? -Infinity) - (a.best ?? -Infinity));
  const topLocale = localeSummary[0];
  const bottomLocale = localeSummary.reduce((acc, item) => acc == null || (item.worst ?? Infinity) < (acc.worst ?? Infinity) ? item : acc, null);
  const latest = overall[overall.length - 1];
  const previous = overall.length >= 2 ? overall[overall.length - 2] : null;
  items.push(`Best checkpoint/model is <span class="hl-orange">${escapeHtml(best.model)}</span> with overall DTER <span class="hl-green">${fmtPct(best.dter)}</span> and WERR <span class="hl-green">${fmtSigned(best.werr)}</span>.`);
  if (collapse) {
    items.push(`Collapse begins at <span class="hl-orange">${escapeHtml(collapse.model)}</span>, where overall WERR flips to <span class="hl-red">${fmtSigned(collapse.werr)}</span>.`);
  } else {
    items.push(`No checkpoint in this sweep regresses below baseline overall; every model stays at or above <span class="hl-green">0.00%</span> WERR.`);
  }
  if (topLocale) {
    items.push(`Strongest locale gain is <span class="hl-green">${escapeHtml(topLocale.lang)}</span> at <span class="hl-green">${fmtSigned(topLocale.best)}</span>; weakest locale is <span class="hl-red">${escapeHtml(bottomLocale.lang)}</span> at <span class="hl-red">${fmtSigned(bottomLocale.worst)}</span>.`);
  }
  if (previous) {
    const delta = latest.werr - previous.werr;
    items.push(`Late-stage movement from <span class="hl-orange">${escapeHtml(previous.model)}</span> to <span class="hl-orange">${escapeHtml(latest.model)}</span> is <span class="${delta >= 0 ? 'hl-green' : 'hl-red'}">${fmtSigned(delta)}</span> WERR, which indicates ${Math.abs(delta) < 0.002 ? 'a plateau' : delta > 0 ? 'continued improvement' : 'a late regression'}.`);
  }
  items.push(`Suggested next step: keep <span class="hl-orange">${escapeHtml(best.model)}</span> as the working checkpoint and inspect the locales that turn <span class="hl-red">negative</span> earliest before extending the run.`);
  document.getElementById('takeaways').innerHTML = items.slice(0, 5).map((item) => `<li>${item}</li>`).join('');
}

buildHeader();
buildOverallCharts();
buildLocaleCharts();
buildSmallMultiples();
buildRankingTable();
buildDataTables();
buildTakeaways();
  </script>
</body>
</html>
"""


def _compute_werr(baseline: Optional[float], model: Optional[float], cell_value: Any) -> Optional[float]:
    if isinstance(cell_value, (int, float)):
        return float(cell_value)
    if baseline and model is not None:
        return 1 - float(model) / float(baseline)
    return None


def _row_kind(name: str) -> str:
    if name == "overall avg":
        return "overall"
    if name.endswith(" avg"):
        return "lang_avg"
    return "data"


def _row_lang(name: str, kind: str) -> str:
    if kind == "lang_avg":
        return name[:-4]
    if kind == "overall":
        return "overall"
    return name.split("_", 1)[0]


def _load_report(xlsx_path: Path, baseline_label: str, title: str) -> Dict[str, Any]:
    wb = load_workbook(xlsx_path, data_only=False)
    ws = wb["inhouse_dter"]
    rank_ws = wb["overall_improve_degrade"]

    model_labels: List[str] = []
    col = 3
    while True:
        value = ws.cell(row=HEADER_ROW, column=col).value
        if value in (None, "WERR"):
            break
        model_labels.append(str(value))
        col += 1
    n_models = 1 + len(model_labels)
    model_cols = list(range(2, 2 + n_models))
    werr_cols = list(range(2 + n_models, 2 + n_models + len(model_labels)))

    rows: List[Dict[str, Any]] = []
    datasets_by_lang: Dict[str, List[Dict[str, Any]]] = {}
    lang_avgs: Dict[str, Dict[str, Any]] = {}
    lang_order: List[str] = []
    overall: Optional[Dict[str, Any]] = None
    row_idx = DATA_START
    dataset_count = 0

    while True:
        name = ws.cell(row=row_idx, column=1).value
        if not name:
            break
        name = str(name)
        kind = _row_kind(name)
        lang = _row_lang(name, kind)
        dter = []
        for mc in model_cols:
            value = ws.cell(row=row_idx, column=mc).value
            dter.append(float(value) if value is not None else None)
        baseline = dter[0]
        werr: List[Optional[float]] = []
        for idx, wc in enumerate(werr_cols, start=1):
            cell_value = ws.cell(row=row_idx, column=wc).value
            werr.append(_compute_werr(baseline, dter[idx], cell_value))
        row = {
            "name": name,
            "lang": lang,
            "kind": kind,
            "dter": dter,
            "werr": werr,
        }
        rows.append(row)
        if kind == "data":
            dataset_count += 1
            datasets_by_lang.setdefault(lang, []).append(row)
            if lang not in lang_order:
                lang_order.append(lang)
        elif kind == "lang_avg":
            lang_avgs[lang] = row
            if lang not in lang_order:
                lang_order.append(lang)
        else:
            overall = row
        row_idx += 1

    ranking: List[Dict[str, Any]] = []
    rank_row = 3
    while True:
        rank = rank_ws.cell(row=rank_row, column=1).value
        if rank is None:
            break
        ranking.append(
            {
                "rank": int(rank),
                "model": str(rank_ws.cell(row=rank_row, column=2).value),
                "direction": str(rank_ws.cell(row=rank_row, column=3).value),
                "baseline_overall_dter": _maybe_float(rank_ws.cell(row=rank_row, column=4).value),
                "model_overall_dter": _maybe_float(rank_ws.cell(row=rank_row, column=5).value),
                "dter_delta": _maybe_float(rank_ws.cell(row=rank_row, column=6).value),
                "werr": _maybe_float(rank_ws.cell(row=rank_row, column=7).value),
                "datasets": str(rank_ws.cell(row=rank_row, column=8).value),
            }
        )
        rank_row += 1

    if overall is None:
        raise ValueError("Missing overall avg row in inhouse_dter sheet")

    ordered_rows: List[Dict[str, Any]] = []
    for lang in lang_order:
        ordered_rows.extend(datasets_by_lang.get(lang, []))
        if lang in lang_avgs:
            ordered_rows.append(lang_avgs[lang])
    ordered_rows.append(overall)

    return {
        "title": title,
        "source": str(xlsx_path),
        "baseline_label": baseline_label,
        "baseline": baseline_label,
        "steps": [baseline_label] + model_labels,
        "models": model_labels,
        "dataset_count": dataset_count,
        "lang_order": lang_order,
        "datasets": [row for row in rows if row["kind"] == "data"],
        "datasets_by_lang": datasets_by_lang,
        "lang_avgs": lang_avgs,
        "overall": overall,
        "ranking": ranking,
        "ordered_rows": ordered_rows,
        "lang_colors": LANG_COLORS,
    }


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def build_html(data: Dict[str, Any]) -> str:
    return TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xlsx", help="Path to in-house DTER xlsx.")
    parser.add_argument("--out", help="Output html path. Defaults to <xlsx-stem>.html next to the xlsx.")
    parser.add_argument("--baseline-label", default="Qwen3.5-audio", help="Label shown for the baseline column.")
    parser.add_argument("--title", help="Report title. Defaults to the xlsx stem.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xlsx_path = Path(args.xlsx).resolve()
    out_path = Path(args.out).resolve() if args.out else xlsx_path.with_suffix(".html")
    title = args.title or xlsx_path.stem
    data = _load_report(xlsx_path, baseline_label=args.baseline_label, title=title)
    html = build_html(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    if "__DATA__" in html:
        raise RuntimeError("__DATA__ placeholder remained in output")

    overall_werr = data["overall"]["werr"]
    best_idx, best_value = max(enumerate(overall_werr), key=lambda item: item[1])
    worst_idx, worst_value = min(enumerate(overall_werr), key=lambda item: item[1])
    collapse_idx = next((idx for idx, value in enumerate(overall_werr) if value < 0), None)
    collapse_label = data["models"][collapse_idx] if collapse_idx is not None else "none"
    print(f"Wrote {out_path} {out_path.stat().st_size}")
    print(
        "Summary "
        f"best={data['models'][best_idx]}:{best_value:.4f} "
        f"worst={data['models'][worst_idx]}:{worst_value:.4f} "
        f"collapse={collapse_label}"
    )


if __name__ == "__main__":
    main()
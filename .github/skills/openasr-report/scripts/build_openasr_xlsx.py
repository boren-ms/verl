#!/usr/bin/env python3
"""Build an OpenASR + OpenASR_ML xlsx report matching ~/code/MoE/results/template.xlsx.

Metrics layout (`openasr` sheet):
  Row 2  : Header | Baseline | <model-label-1> | <model-label-2> | ... | WERR(s)
  Row 3  : Column | A        | B               | C               | ... | A->B, A->C, ...
  Row 4-11 : OpenASR datasets (ami_clean..voxpopuli-cleaned-aa)
  Row 12 : avg (numeric dataset mean)
  Row 13 : Column row repeated for the ML section
  Row 14-37 : ML datasets with per-language avg rows mixed in
  Row 38 : ml avg (numeric mean of available language means)
  Column(s) after the last model column: WERR = 1 - <model>/Baseline for each model column.
Additional sheets: `model_info` and `dataset_results`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openpyxl import Workbook, load_workbook

from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# Row-highlight colors.
HEADER_FILL = PatternFill(start_color="FFD9E1F2", end_color="FFD9E1F2", fill_type="solid")  # light blue
LANG_AVG_FILL = PatternFill(start_color="FFFFF2CC", end_color="FFFFF2CC", fill_type="solid")  # light yellow
OVERALL_AVG_FILL = PatternFill(start_color="FFE2EFDA", end_color="FFE2EFDA", fill_type="solid")  # light green

# ---------------------------------------------------------------------------
# Fixed schemas
# ---------------------------------------------------------------------------

OPENASR_DATASETS: List[str] = [
    "ami_clean",
    "earnings22-cleaned-aa-chunked",
    "gigaspeech_clean",
    "librispeech-clean",
    "librispeech-other",
    "monsoon_en_in",
    "spgispeech",
    "voxpopuli-cleaned-aa",
]

# Ordered: list of (language_code, [datasets in that language]).
OPENASR_ML_GROUPS: List[Tuple[str, List[str]]] = [
    ("de", ["de_fleurs", "de_mcv"]),
    ("es", ["es_fleurs", "es_mcv", "es_mls"]),
    ("fr", ["fr_fleurs", "fr_mcv", "fr_mls"]),
    ("hi", ["monsoon_hi_in"]),
    ("it", ["it_fleurs", "it_mcv", "it_mls"]),
    ("nl", ["nl_fleurs", "nl_mcv", "nl_mls"]),
    ("pt", ["pt_fleurs", "pt_mls"]),
]

BASELINE_LABEL = "2609r2"
# User-supplied 2609r2 column A percentages (2026-10-07), converted to fractions.
BASELINE_METRICS: Dict[str, float] = {
    # OpenASR
    "ami_clean": 0.0672,
    "earnings22-cleaned-aa-chunked": 0.0502,
    "gigaspeech_clean": 0.0758,
    "librispeech-clean": 0.0126,
    "librispeech-other": 0.0274,
    "monsoon_en_in": 0.0339,
    "spgispeech": 0.0215,
    "voxpopuli-cleaned-aa": 0.0163,
    # OpenASR_ML
    "de_fleurs": 0.0232,
    "de_mcv": 0.0190,
    "es_fleurs": 0.0271,
    "es_mcv": 0.0220,
    "es_mls": 0.0258,
    "fr_fleurs": 0.0286,
    "fr_mcv": 0.0417,
    "fr_mls": 0.0237,
    "monsoon_hi_in": 0.0878,
    "it_fleurs": 0.0130,
    "it_mcv": 0.0178,
    "it_mls": 0.0399,
    "nl_fleurs": 0.0388,
    "nl_mcv": 0.0164,
    "nl_mls": 0.0453,
    "pt_fleurs": 0.0310,
    "pt_mls": 0.0336,
}

WER_LINE_RE = re.compile(
    r"val-aux/(?P<dataset>[a-zA-Z0-9_-]+)/p_err/mean@1[:=]\s*(?P<value>[0-9.eE+-]+)"
)

ML_ALIASES = {
    f"{corpus}_{lang}": dataset
    for lang, datasets in OPENASR_ML_GROUPS
    for dataset in datasets
    for prefix, separator, corpus in [dataset.partition("_")]
    if separator and prefix == lang
}


def dataset_key(name: str) -> str:
    return ML_ALIASES.get(name, name)


def parse_wer_lines(text: str) -> Dict[str, float]:
    """Extract the latest ``val-aux/<ds>/p_err/mean@1`` value per dataset."""
    out: Dict[str, float] = {}
    for m in WER_LINE_RE.finditer(text):
        try:
            out[dataset_key(m.group("dataset"))] = float(m.group("value"))
        except ValueError:
            continue
    return out


def fetch_ray_logs(node: str, job_id: str) -> str:
    # brix ssh joins trailing args by space, so wrap the remote command in a single
    # pre-quoted string to preserve `bash -l -c "..."` argument boundaries.
    remote = f"bash -l -c 'ray job logs {job_id} 2>&1'"
    cmd = ["brix", "ssh", node, "--", remote]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(
            f"[warn] ray job logs failed for {node}/{job_id} (rc={proc.returncode}):\n"
            f"{proc.stderr[-500:]}",
            file=sys.stderr,
        )
    return proc.stdout


def load_metrics(args: argparse.Namespace) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if args.metrics:
        data = json.loads(Path(args.metrics).read_text())
        metrics.update({dataset_key(k): float(v) for k, v in data.items()})
    if args.from_text:
        for text_path in args.from_text:
            metrics.update(parse_wer_lines(Path(text_path).read_text()))
    if args.from_ray:
        for node, job_id in args.from_ray:
            metrics.update(parse_wer_lines(fetch_ray_logs(node, job_id)))
    return metrics


OPENASR_START = 4  # row of first OpenASR dataset
OPENASR_END = OPENASR_START + len(OPENASR_DATASETS) - 1  # 11
OPENASR_AVG = OPENASR_END + 1  # 12

ML_HEADER = OPENASR_AVG + 1  # 13
ML_START = ML_HEADER + 1  # 14

MODEL_INFO_FIELDS = [
    ("model_path", "Model path"),
    ("config", "Config"),
    ("node", "Node"),
    ("ray_job_id", "Ray job ID"),
    ("wandb_url", "W&B URL"),
    ("code_snapshot", "Code snapshot"),
]
MODEL_INFO_HEADER = ["Model", *[title for _, title in MODEL_INFO_FIELDS]]
DATASET_RESULTS_HEADER = ["Model", "Dataset", "Result filepath"]


def _load_string_mapping(path: Path) -> Dict[str, str]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ValueError(f"{path}: expected a JSON object mapping strings to strings")
    return data


def _write_metadata_sheets(wb, columns, model_info, dataset_results) -> None:
    unknown_models = (set(model_info) | set(dataset_results)) - {label for label, _ in columns}
    if unknown_models:
        raise ValueError(f"Metadata references unknown models: {sorted(unknown_models)}")
    info_sheet = wb.create_sheet("model_info")
    results_sheet = wb.create_sheet("dataset_results")
    info_sheet.append(MODEL_INFO_HEADER)
    results_sheet.append(DATASET_RESULTS_HEADER)
    datasets = OPENASR_DATASETS + [ds for _, group in OPENASR_ML_GROUPS for ds in group]
    for label, _ in columns:
        info = model_info.get(label, {})
        unknown = set(info) - {key for key, _ in MODEL_INFO_FIELDS}
        if unknown:
            raise ValueError(f"{label}: unknown model information fields: {sorted(unknown)}")
        results = dataset_results.get(label, {})
        unknown = set(results) - set(datasets)
        if unknown:
            raise ValueError(f"{label}: unknown dataset result names: {sorted(unknown)}")
        info_sheet.append([label, *[info.get(key) for key, _ in MODEL_INFO_FIELDS]])
        for dataset in datasets:
            results_sheet.append([label, dataset, results.get(dataset)])
    for sheet in (info_sheet, results_sheet):
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True)
            cell.fill = HEADER_FILL
        for row in sheet:
            for cell in row:
                if isinstance(cell.value, str):
                    cell.data_type = "s"
                cell.alignment = Alignment(horizontal="center", vertical="center")
        for column in sheet.columns:
            width = min(90, max(18, max(len(str(cell.value or "")) for cell in column) + 2))
            sheet.column_dimensions[get_column_letter(column[0].column)].width = width


def read_existing_metadata(path: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    info: Dict[str, Dict[str, str]] = {}
    results: Dict[str, Dict[str, str]] = {}
    wb = load_workbook(path, data_only=True)
    try:
        for name, header in (("model_info", MODEL_INFO_HEADER), ("dataset_results", DATASET_RESULTS_HEADER)):
            if name in wb.sheetnames and [cell.value for cell in wb[name][1]] != header:
                raise ValueError(f"{path}: incompatible {name} header")
        if "model_info" in wb.sheetnames:
            for row in wb["model_info"].iter_rows(min_row=2, values_only=True):
                label, *values = row
                info[label] = {
                    key: value for (key, _), value in zip(MODEL_INFO_FIELDS, values) if value is not None
                }
        if "dataset_results" in wb.sheetnames:
            for label, dataset, result_path in wb["dataset_results"].iter_rows(min_row=2, values_only=True):
                if result_path is not None:
                    results.setdefault(label, {})[dataset] = result_path
    finally:
        wb.close()
    return info, results


def _ml_layout() -> Tuple[List[Tuple[str, str]], int]:
    """Return ((row_kind, label), ...) starting at row ML_START and the ml-avg row.

    row_kind is 'data' for dataset rows or 'lang_avg' for per-language avg rows.
    """
    layout: List[Tuple[str, str]] = []
    for lang, datasets in OPENASR_ML_GROUPS:
        for ds in datasets:
            layout.append(("data", ds))
        layout.append(("lang_avg", f"{lang} avg"))
    return layout, ML_START + len(layout)  # ml avg row index


def _write_dataset_werr(ws, row: int, column: int, baseline_col: int, model_col: int) -> None:
    cell = ws.cell(row=row, column=column)
    baseline = ws.cell(row=row, column=baseline_col)
    model = ws.cell(row=row, column=model_col)
    if baseline.value is not None and model.value is not None and baseline.value > 0:
        cell.value = f"=1-{model.coordinate}/{baseline.coordinate}"


def _mean_error(metrics: Dict[str, float], groups: List[List[str]]) -> Optional[float]:
    means = []
    for datasets in groups:
        values = [metrics[ds] for ds in datasets if ds in metrics]
        if values:
            means.append(sum(values) / len(values))
    return sum(means) / len(means) if means else None


def _write_average_werr(
    cell, groups: List[List[str]], baseline: Dict[str, float], model: Dict[str, float]
) -> None:
    common = [[ds for ds in datasets if ds in baseline and ds in model] for datasets in groups]
    cell.font = Font(bold=True)
    baseline_mean = _mean_error(baseline, common)
    model_mean = _mean_error(model, common)
    if baseline_mean is not None and baseline_mean > 0 and model_mean is not None:
        cell.value = 1 - model_mean / baseline_mean


def build_workbook(
    columns: List[Tuple[str, Dict[str, float]]],
    out_path: Path,
    model_info: Optional[Dict[str, Dict[str, str]]] = None,
    dataset_results: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    """``columns`` is a list of (label, metrics_dict). The first is the baseline (column B)."""

    labels = [label for label, _ in columns]
    if len(labels) != len(set(labels)):
        raise ValueError("Model labels must be unique to associate metadata unambiguously")
    for label, metrics in columns:
        for dataset, value in metrics.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label}/{dataset}: expected a finite, nonnegative error rate, got {value}")

    wb = Workbook()
    ws = wb.active
    ws.title = "openasr"

    n_models = len(columns)  # baseline + new model(s)
    model_cols = list(range(2, 2 + n_models))  # spreadsheet columns 2..(1+n_models)
    werr_cols = list(range(2 + n_models, 2 + 2 * n_models - 1 + 1))
    # WERR is computed relative to baseline (column B), so we make one WERR column per
    # non-baseline model column.
    werr_cols = list(range(2 + n_models, 2 + n_models + (n_models - 1)))

    bold = Font(bold=True)

    # Header row 2
    ws.cell(row=2, column=1, value="Header").font = bold
    ws.cell(row=2, column=2, value=columns[0][0]).font = bold
    for i, (label, _) in enumerate(columns[1:], start=0):
        ws.cell(row=2, column=3 + i, value=label).font = bold
    for j, mc in enumerate(model_cols[1:]):
        ws.cell(row=2, column=werr_cols[j], value="WERR").font = bold

    # Column row 3
    ws.cell(row=3, column=1, value="Column").font = bold
    for i, mc in enumerate(model_cols):
        ws.cell(row=3, column=mc, value=chr(ord("A") + i)).font = bold
    for j, wc in enumerate(werr_cols):
        target_letter = chr(ord("A") + 1 + j)  # B, C, D...
        ws.cell(row=3, column=wc, value=f"A->{target_letter}").font = bold

    # OpenASR section (rows 4..11)
    for r, ds in enumerate(OPENASR_DATASETS, start=OPENASR_START):
        ws.cell(row=r, column=1, value=ds)
        for i, (_, metrics) in enumerate(columns):
            v = metrics.get(ds)
            if v is not None:
                ws.cell(row=r, column=model_cols[i], value=float(v))
        for j, wc in enumerate(werr_cols):
            _write_dataset_werr(ws, r, wc, model_cols[0], model_cols[1 + j])

    # OpenASR avg row 12 (computed numeric values so cells are always filled)
    r = OPENASR_AVG
    ws.cell(row=r, column=1, value="avg").font = bold
    for i, (_, metrics) in enumerate(columns):
        avg = _mean_error(metrics, [OPENASR_DATASETS])

        if avg is not None:
            ws.cell(row=r, column=model_cols[i], value=float(avg)).font = bold
    for j, wc in enumerate(werr_cols):
        _write_average_werr(ws.cell(r, wc), [OPENASR_DATASETS], columns[0][1], columns[1 + j][1])

    # ML header row 13
    ws.cell(row=ML_HEADER, column=1, value="Column").font = bold
    for i, mc in enumerate(model_cols):
        ws.cell(row=ML_HEADER, column=mc, value=chr(ord("A") + i)).font = bold
    for j, wc in enumerate(werr_cols):
        target_letter = chr(ord("A") + 1 + j)
        ws.cell(row=ML_HEADER, column=wc, value=f"A->{target_letter}").font = bold

    # ML datasets / per-language avg rows
    layout, ml_avg_row = _ml_layout()
    current_lang_start: Optional[int] = None
    for offset, (kind, label) in enumerate(layout):
        r = ML_START + offset
        ws.cell(row=r, column=1, value=label)
        if kind == "data":
            if current_lang_start is None:
                current_lang_start = r
            for i, (_, metrics) in enumerate(columns):
                v = metrics.get(label)
                if v is not None:
                    ws.cell(row=r, column=model_cols[i], value=float(v))
            for j, wc in enumerate(werr_cols):
                _write_dataset_werr(ws, r, wc, model_cols[0], model_cols[1 + j])
        else:  # lang_avg
            ws.cell(row=r, column=1).font = bold
            assert current_lang_start is not None
            # Datasets contributing to this language group.
            group_datasets = [
                lbl for k, lbl in layout[offset - (r - current_lang_start):offset] if k == "data"
            ]

            for i, (_, metrics) in enumerate(columns):
                avg = _mean_error(metrics, [group_datasets])

                if avg is not None:
                    ws.cell(row=r, column=model_cols[i], value=float(avg)).font = bold
            for j, wc in enumerate(werr_cols):
                _write_average_werr(ws.cell(r, wc), [group_datasets], columns[0][1], columns[1 + j][1])
            current_lang_start = None

    # Average unrounded language means so each represented language has equal weight.
    r = ml_avg_row
    ws.cell(row=r, column=1, value="ml avg").font = bold
    ml_groups = [datasets for _, datasets in OPENASR_ML_GROUPS]

    for i, (_, metrics) in enumerate(columns):
        avg = _mean_error(metrics, ml_groups)

        if avg is not None:
            ws.cell(row=r, column=model_cols[i], value=float(avg)).font = bold
    for j, wc in enumerate(werr_cols):
        _write_average_werr(ws.cell(r, wc), ml_groups, columns[0][1], columns[1 + j][1])

    # Cell formatting: percentages for numeric data, column widths.
    pct_fmt = "0.00%"
    for r in list(range(OPENASR_START, OPENASR_AVG + 1)) + list(range(ML_START, ml_avg_row + 1)):
        for mc in model_cols + werr_cols:
            ws.cell(row=r, column=mc).number_format = pct_fmt

    # Row highlights.
    all_cols = [1] + model_cols + werr_cols
    header_rows = [2, 3, ML_HEADER]
    lang_avg_rows = [
        ML_START + offset for offset, (kind, _) in enumerate(layout) if kind == "lang_avg"
    ]
    overall_avg_rows = [OPENASR_AVG, ml_avg_row]
    for r in header_rows:
        for c in all_cols:
            ws.cell(row=r, column=c).fill = HEADER_FILL
    for r in lang_avg_rows:
        for c in all_cols:
            ws.cell(row=r, column=c).fill = LANG_AVG_FILL
    for r in overall_avg_rows:
        for c in all_cols:
            ws.cell(row=r, column=c).fill = OVERALL_AVG_FILL

    ws.column_dimensions["A"].width = 18
    for mc in model_cols + werr_cols:
        ws.column_dimensions[get_column_letter(mc)].width = 18

    # Center-align every cell except column A across the whole used range.
    center = Alignment(horizontal="center", vertical="center")
    for r in range(2, ml_avg_row + 1):
        for c in model_cols + werr_cols:
            ws.cell(row=r, column=c).alignment = center

    # Conditional 3-color scale on WERR columns: red for negative (regression),
    # white at 0, green for positive (WER reduction). Midpoint fixed at 0.
    for wc in werr_cols:
        col_letter = get_column_letter(wc)
        rng = f"{col_letter}{OPENASR_START}:{col_letter}{ml_avg_row}"
        rule = ColorScaleRule(
            start_type="num", start_value=-1, start_color="FFF8696B",  # red
            mid_type="num", mid_value=0, mid_color="FFFFFFFF",          # white
            end_type="num", end_value=1, end_color="FF63BE7B",          # green
        )
        ws.conditional_formatting.add(rng, rule)

    _write_metadata_sheets(wb, columns, model_info or {}, dataset_results or {})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("label", help="New model label (column header).")
    p.add_argument("--from-ray", nargs=2, metavar=("NODE", "JOB_ID"), action="append", default=[])
    p.add_argument("--from-text", action="append", default=[])
    p.add_argument("--metrics", help="JSON file: {dataset: wer_fraction}.")
    p.add_argument("--model-info", help="Candidate model information JSON with string-valued fields.")
    p.add_argument("--dataset-results", help="Candidate result paths JSON: {dataset: result_filepath}.")
    p.add_argument("--baseline", help="Override baseline metrics JSON.")
    p.add_argument("--baseline-label", default=BASELINE_LABEL)
    p.add_argument(
        "--extend-xlsx",
        help=(
            "Existing xlsx to extend by appending the new model as the next column. "
            "The sheet layout must match the template."
        ),
    )
    p.add_argument("--out", help="Output xlsx path (default: tmp/openasr_report/<label>.xlsx).")
    return p.parse_args()


def read_existing_xlsx(path: Path) -> List[Tuple[str, Dict[str, float]]]:
    wb = load_workbook(path, data_only=True)
    ws = wb["openasr"] if "openasr" in wb.sheetnames else wb.active

    # Discover model columns: row 3 cells with single letter values starting at B.
    model_cols: List[int] = []
    for c in range(2, ws.max_column + 1):
        v = ws.cell(row=3, column=c).value
        if isinstance(v, str) and len(v) == 1 and v.isalpha():
            model_cols.append(c)
        else:
            break

    labels: List[str] = []
    for mc in model_cols:
        v = ws.cell(row=2, column=mc).value
        labels.append(str(v) if v is not None else f"col{mc}")
    metrics_per_col: List[Dict[str, float]] = [dict() for _ in model_cols]
    layout, ml_avg_row = _ml_layout()
    rows: List[Tuple[int, str]] = []
    for r, ds in enumerate(OPENASR_DATASETS, start=OPENASR_START):
        rows.append((r, ds))
    for offset, (kind, label) in enumerate(layout):
        if kind == "data":
            rows.append((ML_START + offset, label))
    for r, ds in rows:
        if ws.cell(row=r, column=1).value != ds:
            raise ValueError(
                f"{path}: incompatible dataset layout at row {r}: expected {ds!r}, "
                f"found {ws.cell(row=r, column=1).value!r}; rebuild with the current schema."
            )
        for i, mc in enumerate(model_cols):
            v = ws.cell(row=r, column=mc).value
            if isinstance(v, (int, float)):
                metrics_per_col[i][ds] = float(v)

    return [(label, m) for label, m in zip(labels, metrics_per_col)]


def main() -> int:
    args = parse_args()

    new_metrics = load_metrics(args)
    if not new_metrics:
        print("[error] no metrics collected; pass --from-ray/--from-text/--metrics.", file=sys.stderr)
        return 2

    if args.extend_xlsx:
        columns = read_existing_xlsx(Path(args.extend_xlsx))
        model_info, dataset_results = read_existing_metadata(Path(args.extend_xlsx))
        if not columns:
            print(f"[warn] could not parse {args.extend_xlsx}; using baseline only.", file=sys.stderr)
    else:
        columns = []
        model_info, dataset_results = {}, {}

    if not columns:
        baseline_metrics = BASELINE_METRICS
        if args.baseline:
            baseline_metrics = {
                dataset_key(k): float(v)
                for k, v in json.loads(Path(args.baseline).read_text()).items()
            }
        columns = [(args.baseline_label, baseline_metrics)]

    columns.append((args.label, new_metrics))
    if args.model_info:
        model_info[args.label] = _load_string_mapping(Path(args.model_info))
    if args.dataset_results:
        dataset_results[args.label] = {
            dataset_key(key): value for key, value in _load_string_mapping(Path(args.dataset_results)).items()
        }

    out_path = Path(
        args.out
        or f"tmp/openasr_report/{args.label.replace('/', '_').replace('@', '_')}.xlsx"
    )
    build_workbook(columns, out_path, model_info=model_info, dataset_results=dataset_results)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

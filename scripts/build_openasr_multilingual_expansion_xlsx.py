"""Append verified multilingual results to a copy of an existing OpenASR workbook."""

import argparse
from copy import copy
import json
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.workbook.properties import CalcProperties

LABELS = ("2609r2", "2609r2b", "2611r2")
SOURCES = (
    ("zh_fleurs", "Mandarin FLEURS"),
    ("zh_aishell1", "Mandarin AISHELL-1"),
    ("vi_fleurs", "Vietnamese FLEURS"),
    ("vi_mcv", "Vietnamese Common Voice"),
    ("eslo_telephony_fr", "French ESLO telephony"),
    ("accueil_ubs_fr", "French Accueil UBS"),
)
BLUE = "D9EAF7"
GREEN = "E2F0D9"


def build(comparison, template, output):
    report = json.loads(Path(comparison).read_text())
    models = report["models"]
    for label in LABELS:
        model = models[label]
        if model["ray_status"] != "SUCCEEDED" or len(model["datasets"]) != 14:
            raise ValueError(f"Incomplete evaluation: {label}")
        if sum(x["scored_parents"] for x in model["datasets"].values()) != 10693:
            raise ValueError(f"Incomplete recording coverage: {label}")
        if any(x["include_in_headline_average"] for x in model["datasets"].values()):
            raise ValueError(f"Supplementary rows enter headline averages: {label}")
    wb = load_workbook(template)
    if not {"openasr", "model_info", "dataset_results"}.issubset(wb.sheetnames):
        raise ValueError("Template must contain the three canonical OpenASR sheets")
    if "multilingual_expansion" in wb.sheetnames:
        raise ValueError("Template already contains the multilingual expansion")
    ws = wb.create_sheet("multilingual_expansion")
    names = {label: label + " [multilingual expansion]" for label in LABELS}
    ws.merge_cells("A1:H1")
    ws["A1"] = "OpenASR multilingual expansion — verified evaluation, 2026-11-07"
    ws["A1"].font = Font(bold=True, size=14)
    headers = ["Dataset", "Metric", "Recordings", *names.values(),
               "2609r2b error reduction vs 2609r2", "2611r2 error reduction vs 2609r2"]

    def header(row):
        for col, value in enumerate(headers, 1):
            cell = ws.cell(row, col, value)
            cell.font = Font(bold=True)
            cell.fill = PatternFill("solid", fgColor=BLUE)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.row_dimensions[row].height = 48

    def result(row, title, values, recordings):
        metrics = {x["metric"] for x in values}
        if len(metrics) != 1:
            raise ValueError("Metric units disagree between models")
        ws.cell(row, 1, title)
        ws.cell(row, 2, next(iter(metrics)))
        ws.cell(row, 3, recordings)
        for col, item in enumerate(values, 4):
            counts = item.get("counts", item)
            rate = counts["n_err"] / counts["n_ref"]
            if abs(rate - item["error_rate"]) > 1e-12:
                raise ValueError("Metric differs from saved corpus counts")
            ws.cell(row, col, rate)
        for col, candidate in [(7, "E"), (8, "F")]:
            ws.cell(row, col, f'=IF(D{row}>0,1-{candidate}{row}/D{row},"")')
        for col in range(2, 9):
            ws.cell(row, col).alignment = Alignment(horizontal="center", vertical="center")
        for col in range(4, 9):
            ws.cell(row, col).number_format = "0.00%"

    header(2)
    for row, (source, title) in enumerate(SOURCES, 3):
        values = [models[label]["datasets"][source] for label in LABELS]
        result(row, title, values, values[0]["scored_parents"])
    result(9, "CORAAL, eight locations pooled", [models[x]["coraal_aggregate"] for x in LABELS], 271)
    for cell in ws[9]:
        cell.fill = PatternFill("solid", fgColor=GREEN)
        cell.font = Font(bold=True)
    ws.merge_cells("A11:H11")
    ws["A11"] = "CORAAL location diagnostics"
    ws["A11"].font = Font(bold=True)
    header(12)
    for row, location in enumerate(("ATL", "DCA", "DCB", "DTA", "LES", "PRV", "ROC", "VLD"), 13):
        values = [models[label]["datasets"]["coraal_" + location.lower()] for label in LABELS]
        result(row, location, values, values[0]["scored_parents"])
    for column in ("G", "H"):
        for start, end in ((3, 9), (13, 20)):
            ws.conditional_formatting.add(f"{column}{start}:{column}{end}", ColorScaleRule(
                start_type="num", start_value=-1, start_color="F8696B",
                mid_type="num", mid_value=0, mid_color="FFFFFF",
                end_type="num", end_value=1, end_color="63BE7B"))
    for col, width in enumerate((37, 10, 13, 27, 27, 27, 25, 25), 1):
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.freeze_panes = "D3"
    ws.auto_filter.ref = "A2:H9"
    ws.sheet_view.showGridLines = False
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.orientation = "landscape"
    ws.page_setup.paperSize = ws.PAPERSIZE_A3
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 1
    ws.print_options.horizontalCentered = True
    ws.print_area = "A1:H20"
    for label in LABELS:
        model = models[label]
        wb["model_info"].append([
            names[label], model["model_uri"], model["entrypoint"],
            f"{model['cluster']} / {model['namespace']} / {model['pool']} (2 × 8 H100)",
            model["ray_id"], model["wandb_url"], model["code_revision"],
        ])
        for source, value in model["datasets"].items():
            wb["dataset_results"].append([names[label], source, value["detail_path"]])
    for name in ("model_info", "dataset_results"):
        sheet = wb[name]
        first_new = sheet.max_row - (3 if name == "model_info" else 42) + 1
        for row in sheet.iter_rows(min_row=first_new):
            for cell in row:
                cell._style = copy(sheet.cell(2, cell.column)._style)
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        sheet.auto_filter.ref = sheet.dimensions
    wb.active = wb.sheetnames.index(ws.title)
    wb.calculation = CalcProperties(calcId=191029, fullCalcOnLoad=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)
    print(output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build(args.comparison, args.template, args.out)

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from openpyxl import load_workbook


SCRIPT = (
    Path(__file__).parents[2]
    / ".github/skills/eval-2607-benchmark-report/scripts/build_2607_report.py"
)


def _load_report_module():
    spec = importlib.util.spec_from_file_location("build_2607_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inhouse_avg_rows_use_excel_arithmetic_formulas(tmp_path, monkeypatch):
    source = tmp_path / "measures"
    measures = {
        "enus_conv_fy21q1": {"dter": 0.10, "dter_n_err": 1, "dter_n_ref": 10},
        "enus_conv_om_fy25q3": {"dter": 0.30, "dter_n_err": 90, "dter_n_ref": 300},
    }
    for corpus, values in measures.items():
        corpus_dir = source / corpus
        corpus_dir.mkdir(parents=True)
        (corpus_dir / "measures.json").write_text(json.dumps(values))

    output = tmp_path / "report.xlsx"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--label",
            "candidate",
            "--inhouse-dter",
            str(source),
            "--out",
            str(output),
        ],
    )

    report = _load_report_module()
    assert report.main() == 0

    workbook = load_workbook(output, data_only=False)
    sheet = workbook["inhouse_dter"]
    overall = next(row for row in sheet.iter_rows(values_only=True) if row[0] == "overall avg")
    baseline = report.BENCHMARKS["inhouse_dter"]["embedded_baseline"]
    baseline_values = [
        baseline[key]["dter"]
        for _, datasets in report.INHOUSE_GROUPS
        for key, _ in datasets
    ]
    assert sheet["B7"].value == "=AVERAGE(B4:B6)"
    assert sheet["C7"].value == "=AVERAGE(C4:C6)"
    assert sheet["D7"].value == "=1-C7/B7"
    assert overall[1] == "=AVERAGE(B4:B6,B8:B10,B12:B14,B16:B18,B20:B22,B24:B26)"
    assert overall[2] == "=AVERAGE(C4:C6,C8:C10,C12:C14,C16:C18,C20:C22,C24:C26)"
    assert overall[3] == "=1-C28/B28"
    assert workbook["summary"].cell(2, 3).value == pytest.approx(
        sum(baseline_values) / len(baseline_values)
    )
    assert workbook["summary"].cell(2, 4).value == pytest.approx(0.20)
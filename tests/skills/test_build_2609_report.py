import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from openpyxl import Workbook, load_workbook


SCRIPT = (
    Path(__file__).parents[2]
    / ".github/skills/eval-2609-benchmark-report/scripts/build_2609_report.py"
)


def _load_report_module():
    spec = importlib.util.spec_from_file_location("build_2609_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_updated_openasr_baseline_and_report(tmp_path, monkeypatch):
    report = _load_report_module()
    supplied = {
        "de_fleurs": (1.83, 2.24), "de_mcv": (2.01, 1.90),
        "es_fleurs": (1.75, 2.81), "es_mcv": (2.36, 2.16), "es_mls": (2.79, 2.67),
        "fr_fleurs": (2.48, 2.65), "fr_mcv": (4.32, 4.18), "fr_mls": (2.62, 2.43),
        "monsoon_hi_in": (13.32, 9.63),
        "it_fleurs": (0.87, 1.27), "it_mcv": (1.82, 1.85), "it_mls": (4.58, 4.10),
        "nl_fleurs": (2.82, 3.65), "nl_mcv": (1.88, 1.81), "nl_mls": (4.34, 3.99),
        "pt_fleurs": (2.08, 3.41), "pt_mls": (3.65, 3.27),
    }
    assert report.OPENASR_ML_BASELINE == pytest.approx(
        {key: values[0] / 100 for key, values in supplied.items()}
    )
    assert {key for _, datasets in report.OPENASR_ML_GROUPS for key, _ in datasets} == set(supplied)

    source = tmp_path / "candidate.json"
    source.write_text(json.dumps({key: values[1] / 100 for key, values in supplied.items()}))
    output = tmp_path / "report.xlsx"
    monkeypatch.setattr(
        sys, "argv",
        [str(SCRIPT), "--label", "B", "--openasr-ml", str(source), "--out", str(output)],
    )
    assert report.main() == 0

    workbook = load_workbook(output)
    sheet = workbook["openasr_ml"]
    assert sheet["B2"].value == "2609v0"
    seen = set()
    averages = set()
    for row in sheet.iter_rows():
        name = row[0].value
        if name in supplied:
            seen.add(name)
            assert row[1].value == pytest.approx(supplied[name][0] / 100)
            assert row[2].value == pytest.approx(supplied[name][1] / 100)
            assert row[3].value == f"=1-C{row[0].row}/B{row[0].row}"
            assert all(cell.number_format == "0.00%" for cell in row[1:4])
        elif isinstance(name, str) and name.endswith(" avg"):
            averages.add(name)
            assert row[1].value.startswith("=AVERAGE(")
            assert row[2].value.startswith("=AVERAGE(")
    assert seen == set(supplied)
    assert averages == {"de avg", "es avg", "fr avg", "hi avg", "it avg", "nl avg", "pt avg", "overall avg"}
    assert sheet["B28"].value == "=AVERAGE(B4:B5,B7:B9,B11:B13,B15:B15,B17:B19,B21:B23,B25:B26)"
    summary = workbook["summary"]
    assert summary["C2"].value == pytest.approx(55.52 / 1700)
    assert summary["D2"].value == pytest.approx(54.02 / 1700)
    assert summary["E2"].value == pytest.approx(1 - 54.02 / 55.52)
    assert summary["J2"].value == "embedded baseline: 2609v0"
    workbook.close()

    merged_path = tmp_path / "merged.xlsx"
    subprocess.run(
        [sys.executable, str(SCRIPT.with_name("merge_2609_reports.py")),
         "--model-label", "candidate", "--report", f"step10={output}",
         "--out", str(merged_path)],
        check=True, capture_output=True, text=True,
    )
    merged = load_workbook(merged_path)
    assert merged.sheetnames == ["summary", "step10_openasr_ml"]
    copied = merged["step10_openasr_ml"]
    assert copied["B2"].value == "2609v0"
    assert {row[0].value for row in copied.iter_rows() if row[0].value in supplied} == set(supplied)
    assert copied["D15"].value == "=1-C15/B15"
    assert merged["summary"]["D4"].value == pytest.approx(55.52 / 1700)
    assert merged["summary"]["K4"].value == "embedded baseline: 2609v0"
    merged.close()


def test_openml_report_matches_composed_2609_eval_config():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    report = _load_report_module()
    config_path = SCRIPT.parents[4] / report.BENCHMARKS["openasr_ml"]["config"]
    with initialize_config_dir(config_dir=str(config_path.parent), version_base=None):
        config = compose(config_name=config_path.stem)
    rows = OmegaConf.to_container(config.data.val_data, resolve=True)
    sources = [row["post_process"]["add_field"]["fields"]["data_source"] for row in rows]
    assert len(sources) == len(set(sources)) == 17
    assert set(sources) == set(report.OPENASR_ML_BASELINE)
    assert set(sources) == {
        key for _, datasets in report.OPENASR_ML_GROUPS for key, _ in datasets
    }
    assert config.val_reward.custom_reward_function.name == "openasr_eval"
    hindi = rows[sources.index("monsoon_hi_in")]
    assert "lattice" in hindi["post_process"]["verl_format"]["extra_keys"]


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
    summary = workbook["summary"]
    assert [str(item.sqref) for item in summary.conditional_formatting] == ["E2"]
    rule = next(iter(summary.conditional_formatting)).rules[0]
    assert [value.val for value in rule.colorScale.cfvo] == [-0.1, 0.0, 0.1]
    assert [color.rgb for color in rule.colorScale.color] == [
        "FFF8696B",
        "FFFFFFFF",
        "FF63BE7B",
    ]
    assert len(summary._charts) == 1
    chart = summary._charts[0]
    assert len(chart._charts) == 1
    assert len(chart.series) == 1
    assert chart.title is None
    assert chart.visible_cells_only is False
    assert chart.anchor._from.col == 0
    assert chart.anchor._from.row == 4
    assert chart.series[0].tx.strRef.f.endswith("!X2")
    assert chart.series[0].cat.numRef.f.endswith("!$W$3")
    assert chart.series[0].invertIfNegative is False
    assert len(chart.series[0].dPt) == 1
    assert chart.series[0].dPt[0].invertIfNegative is False
    assert chart.series[0].dPt[0].graphicalProperties.solidFill.srgbClr == "4F81BD"
    assert chart.series[0].dPt[0].graphicalProperties.line.solidFill.srgbClr == "4F81BD"
    assert chart.y_axis.majorGridlines is None
    baseline_overall = sum(baseline_values) / len(baseline_values)
    delta = 1 - 0.20 / baseline_overall
    assert chart.y_axis.scaling.min == pytest.approx(delta - abs(delta) * 0.15)
    assert chart.y_axis.scaling.max == 0
    assert chart.x_axis.delete is False
    assert chart.x_axis.axPos == "b"
    assert chart.x_axis.title is None
    assert chart.x_axis.tickLblPos == "low"
    assert chart.x_axis.majorTickMark is None
    assert chart.x_axis.spPr.ln.solidFill.srgbClr == "595959"
    assert chart.x_axis.majorGridlines is not None
    assert chart.x_axis.majorGridlines.spPr.ln.solidFill.srgbClr == "D9D9D9"
    assert chart.y_axis.delete is False
    assert chart.y_axis.axPos == "l"
    assert chart.y_axis.title is None
    assert chart.y_axis.crossesAt == 0
    assert chart.legend is not None
    assert chart.legend.position == "t"
    assert chart.dLbls.showVal is True
    assert chart.dLbls.showLegendKey is False
    assert chart.dLbls.numFmt == "0.00%"
    assert chart.dLbls.position == "outEnd"
    assert chart.overlap == -40
    assert chart.gapWidth == 260
    assert summary["W3"].value == "inhouse_dter"


def test_default_report_path_uses_model_subdirectory():
    report = _load_report_module()

    assert report.default_report_path(
        "remax_2609@step560",
        "az://orngwus2cresco/data/boren/outputs/ver_2609/remax_2609/global_step_560/qwen_hf/",
    ) == Path(
        "tmp/eval_2609_reports/remax_2609/remax_2609_step560.xlsx"
    )
    assert report.default_report_path(
        "remax_2609_step560",
        "",
    ) == Path(
        "tmp/eval_2609_reports/remax_2609/remax_2609_step560.xlsx"
    )


def test_collect_reads_remote_json_metric_source(monkeypatch):
    report = _load_report_module()
    source = "az://orngwus2cresco/data/results/key_metrics.json"
    payload = {"de_fleurs": {"wer": 0.025}, "de_mcv": {"wer": 0.031}}

    def fake_run(command, **kwargs):
        assert command == ["bbb", "cat", source]
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(report.subprocess, "run", fake_run)

    assert report.collect(source, ["wer"]) == payload


def test_default_output_writes_checkpoint_report_under_model(tmp_path, monkeypatch):
    source = tmp_path / "measures"
    corpus_dir = source / "enus_conv_fy21q1"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "measures.json").write_text(
        json.dumps({"dter": 0.10, "dter_n_err": 1, "dter_n_ref": 10})
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--label",
            "remax_2609_step10",
            "--candidate-model-path",
            "az://orngwus2cresco/data/boren/outputs/ver_2609/remax_2609/global_step_10/qwen_hf/",
            "--inhouse-dter",
            str(source),
        ],
    )

    report = _load_report_module()
    assert report.main() == 0
    assert (
        tmp_path
        / "tmp/eval_2609_reports/remax_2609/remax_2609_step10.xlsx"
    ).is_file()


def test_merged_summary_chart_clusters_steps_by_dataset():
    report = _load_report_module()
    workbook = Workbook()
    summary = workbook.active
    summary.title = "summary"
    summary.append(["Model", "candidate"])
    summary.append([])
    summary.append(["Checkpoint", "Benchmark", "Metric", "Baseline", "Candidate", "Delta"])
    for checkpoint, values in (
        ("step10", (("inhouse_dter", 0.18, 0.17), ("openasr_ml", 0.03, 0.031))),
        ("step20", (("inhouse_dter", 0.18, 0.19), ("openasr_ml", 0.03, 0.028))),
    ):
        for benchmark, baseline, candidate in values:
            summary.append(
                [checkpoint, benchmark, "DTER", baseline, candidate, 1 - candidate / baseline]
            )

    assert report.apply_summary_charts(summary) == 1

    marker = next(cell.column for cell in summary[1] if cell.value == "__chart_data__")
    assert [summary.cell(row, marker).value for row in range(3, 5)] == [
        "inhouse_dter",
        "openasr_ml",
    ]
    assert [summary.cell(2, column).value for column in range(marker + 1, marker + 3)] == [
        "step10",
        "step20",
    ]
    chart = summary._charts[0]
    assert len(chart.series) == 2
    assert all(len(series.dPt) == 2 for series in chart.series)
    assert [series.graphicalProperties.solidFill.srgbClr for series in chart.series] == [
        "4F81BD",
        "C0504D",
    ]
    assert all(
        point.graphicalProperties.solidFill.srgbClr == "4F81BD"
        for point in chart.series[0].dPt
    )
    assert all(
        point.graphicalProperties.solidFill.srgbClr == "C0504D"
        for point in chart.series[1].dPt
    )
    assert chart.legend.position == "t"


def test_merged_summary_chart_preserves_multiple_metrics_per_benchmark():
    report = _load_report_module()
    workbook = Workbook()
    summary = workbook.active
    summary.title = "summary"
    summary.append(["Model", "candidate"])
    summary.append([])
    summary.append(["Checkpoint", "Benchmark", "Metric", "Baseline", "Candidate", "Delta"])
    for checkpoint, cer, wer in (
        ("step10", (0.01, 0.02), (0.10, 0.11)),
        ("step20", (0.01, 0.015), (0.10, 0.09)),
    ):
        for metric, (baseline, candidate) in (("CER", cer), ("WER", wer)):
            summary.append(
                [
                    checkpoint,
                    "digits_enus",
                    metric,
                    baseline,
                    candidate,
                    1 - candidate / baseline,
                ]
            )

    assert report.apply_summary_charts(summary) == 1

    marker = next(cell.column for cell in summary[1] if cell.value == "__chart_data__")
    assert [summary.cell(row, marker).value for row in range(3, 5)] == [
        "digits_enus (CER)",
        "digits_enus (WER)",
    ]
    chart = summary._charts[0]
    assert len(chart.series) == 2
    assert all(len(series.dPt) == 2 for series in chart.series)


def test_single_summary_chart_colors_each_dataset():
    report = _load_report_module()
    workbook = Workbook()
    summary = workbook.active
    summary.title = "summary"
    summary.append(["Benchmark", "Metric", "Baseline", "Candidate (step100)", "delta"])
    summary.append(["inhouse_dter", "DTER", 0.18, 0.17, 1 - 0.17 / 0.18])
    summary.append(["openasr_ml", "WER", 0.03, 0.029, 1 - 0.029 / 0.03])
    summary.append(["mixlang", "DTER", 0.21, 0.16, 1 - 0.16 / 0.21])

    assert report.apply_summary_charts(summary) == 1

    chart = summary._charts[0]
    assert len(chart.series) == 1
    assert [point.graphicalProperties.solidFill.srgbClr for point in chart.series[0].dPt] == [
        "4F81BD",
        "C0504D",
        "9BBB59",
    ]
    assert chart.legend.position == "t"
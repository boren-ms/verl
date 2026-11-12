import importlib.util
import json
import sys
from pathlib import Path

import pytest
from openpyxl import load_workbook


SCRIPT = Path(__file__).parents[1] / "scripts/build_openasr_xlsx.py"
SUPPLIED = {
    "ami_clean": (6.72, 6.78),
    "earnings22-cleaned-aa-chunked": (5.02, 5.37),
    "gigaspeech_clean": (7.58, 7.52),
    "librispeech-clean": (1.26, 1.27),
    "librispeech-other": (2.74, 2.88),
    "monsoon_en_in": (3.39, 3.41),
    "spgispeech_clean": (2.15, 2.13),
    "voxpopuli-cleaned-aa": (1.63, 1.69),
    "urgent2024": (9.57, 9.00),
    "de_fleurs": (2.32, 2.24), "de_mcv": (1.90, 1.90),
    "es_fleurs": (2.71, 2.81), "es_mcv": (2.20, 2.16), "es_mls": (2.58, 2.67),
    "fr_fleurs": (2.86, 2.65), "fr_mcv": (4.17, 4.18), "fr_mls": (2.37, 2.43),
    "monsoon_hi_in": (8.78, 9.63),
    "it_fleurs": (1.30, 1.27), "it_mcv": (1.78, 1.85), "it_mls": (3.99, 4.10),
    "nl_fleurs": (3.88, 3.65), "nl_mcv": (1.64, 1.81), "nl_mls": (4.53, 3.99),
    "pt_fleurs": (3.10, 3.41), "pt_mls": (3.36, 3.27),
    "earnings21": (9.0999, 7.0),
    "earnings22": (9.8599, 11.0),
    "coraal": (14.7193, 14.0),
}


@pytest.fixture
def report():
    spec = importlib.util.spec_from_file_location("build_openasr_xlsx", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("source", ["json", "core_text", "aux_text", "ray"])
def test_canonical_english_names_populate_rows(report, tmp_path, monkeypatch, source):
    values = {"ami_clean": 0.06745362563237774, "gigaspeech_clean": 0.07368434305282773,
              "spgispeech_clean": 0.021066400716908046}
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(values))
    namespace = "aux" if source == "aux_text" else "core"
    text = "\n".join(f"val-{namespace}/{name}/p_err/mean@1:{value}" for name, value in values.items())
    logs = tmp_path / "driver.log"
    logs.write_text(text)
    paths = {name: f"az://account/container/results/{name}/0.jsonl" for name in values}
    result_file = tmp_path / "results.json"
    result_file.write_text(json.dumps(paths))
    output = tmp_path / "report.xlsx"
    if source == "json":
        inputs = ["--metrics", str(metrics)]
    elif source == "ray":
        monkeypatch.setattr(report, "fetch_ray_logs", lambda node, sid: text)
        inputs = ["--from-ray", "scoped-node", "exact-submission"]
    else:
        inputs = ["--from-text", str(logs)]
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "candidate", *inputs,
                        "--baseline", str(metrics),
                        "--dataset-results", str(result_file), "--out", str(output)])
    assert report.main() == 0
    workbook = load_workbook(output)
    sheet = workbook["openasr"]
    for row, name in [(4, "ami_clean"), (6, "gigaspeech_clean"), (10, "spgispeech_clean")]:
        assert sheet.cell(row, 1).value == name
        assert sheet.cell(row, 3).value == pytest.approx(values[name])
        assert sheet.cell(row, 4).value == f"=1-C{row}/B{row}"
    assert sheet["C13"].value == pytest.approx(sum(values.values()) / len(values))
    _, results = report.read_existing_metadata(output)
    assert results["candidate"] == paths
    workbook.close()


def test_core_p_err_wins_over_aux_regardless_of_line_order(report):
    core = "val-core/ami_clean/p_err/mean@1:0.067"
    aux = "val-aux/ami_clean/p_err/mean@1:0.08"
    for text in [core + "\n" + aux, aux + "\n" + core]:
        assert report.parse_wer_lines(text) == {"ami_clean": 0.067}


@pytest.mark.parametrize("name", ["ami", "gigaspeech", "spgispeech"])
def test_english_names_are_not_aliased(report, name):
    assert report.dataset_key(name) == name
    assert report.parse_wer_lines(f"val-core/{name}/p_err/mean@1:0.07") == {name: 0.07}


def test_baseline_and_full_report(report, tmp_path, monkeypatch):
    assert report.BASELINE_LABEL == "r2611v1"
    baseline = {key: values[0] / 100 for key, values in SUPPLIED.items()}
    candidate = {key: values[1] / 100 for key, values in SUPPLIED.items()}
    assert set(report.OPENASR_DATASETS) | {
        key for _, datasets in report.OPENASR_ML_GROUPS for key in datasets
    } | set(report.OPENLONG_DATASETS) == set(SUPPLIED)
    baseline_source = tmp_path / "historical_baseline.json"
    baseline_source.write_text(json.dumps(baseline))
    source = tmp_path / "candidate.json"
    source.write_text(json.dumps(candidate))
    output = tmp_path / "report.xlsx"
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "B", "--metrics", str(source), "--baseline", str(baseline_source),
         "--baseline-label", "2609r2", "--out", str(output)],
    )
    assert report.main() == 0
    workbook = load_workbook(output)
    assert workbook.sheetnames == ["openasr", "model_info", "dataset_results"]
    sheet = workbook["openasr"]
    assert sheet["A3"].value == "OpenEN"
    assert sheet["A14"].value == "OpenML"
    seen = set()
    for row in sheet.iter_rows():
        name = row[0].value
        if name in SUPPLIED:
            seen.add(name)
            assert row[1].value == pytest.approx(baseline[name])
            assert row[2].value == pytest.approx(candidate[name])
            assert row[3].value == f"=1-C{row[0].row}/B{row[0].row}"
            assert all(cell.number_format == "0.00%" for cell in row[1:4])
    assert seen == set(SUPPLIED)
    assert sheet["B13"].value == pytest.approx(40.06 / 900)
    assert sheet["C13"].value == pytest.approx(40.05 / 900)
    assert sheet["B39"].value == pytest.approx(3.6366666666666667 / 100)
    assert sheet["A39"].value == "avg"
    assert f"{sheet['B39'].value:.2%}" == "3.64%"
    candidate_ml_mean = sum(
        sum(candidate[ds] for ds in datasets) / len(datasets)
        for _, datasets in report.OPENASR_ML_GROUPS
    ) / len(report.OPENASR_ML_GROUPS)
    assert sheet["C39"].value == pytest.approx(candidate_ml_mean)
    assert sheet["D39"].value == pytest.approx(1 - candidate_ml_mean / sheet["B39"].value)
    assert sheet["B44"].value == pytest.approx(0.11226366666666667)
    assert sheet["C44"].value == pytest.approx(0.32 / 3)
    assert sheet["D44"].value == pytest.approx(1 - 0.32 / 0.336791)
    assert sheet["A44"].value == "avg"
    summaries = {
        "avg": "4.45%", "de avg": "2.11%", "es avg": "2.50%",
        "fr avg": "3.13%", "hi avg": "8.78%", "it avg": "2.36%",
        "nl avg": "3.35%", "pt avg": "3.23%",
    }
    for row in sheet.iter_rows(max_row=38):
        if row[0].value in summaries:
            assert f"{row[1].value:.2%}" == summaries[row[0].value]
    workbook.close()
    columns = report.read_existing_xlsx(output)
    assert columns[0][0] == "2609r2"
    assert columns[0][1] == pytest.approx(baseline)
    assert columns[1][1] == pytest.approx(candidate)


def test_log_names_and_latest_value(report):
    text = "\n".join([
        "val-aux/earnings22-cleaned-aa-chunked/p_err/mean@1:0.0537",
        "val-aux/fleurs_nl/p_err/mean@1:0.04",
        "val-aux/nl_fleurs/p_err/mean@1:0.0365",
        "val-aux/monsoon_hi_in/p_err/mean@1:0.0963",
        "val-aux/urgent2024/p_err/mean@1:0.09",
        "val-aux/urgent2024_clean/p_err/mean@1:0.04",
    ])
    assert report.parse_wer_lines(text) == {
        "earnings22-cleaned-aa-chunked": 0.0537,
        "nl_fleurs": 0.0365,
        "monsoon_hi_in": 0.0963,
        "urgent2024": 0.09,
        "urgent2024_clean": 0.04,
    }


def test_json_aliases_and_baseline_override(report, tmp_path, monkeypatch):
    source = tmp_path / "metrics.json"
    source.write_text(json.dumps({"fleurs_nl": 0.10, "mcv_nl": 0.20, "mls_nl": 0.30}))
    output = tmp_path / "override.xlsx"
    monkeypatch.setattr(
        sys, "argv",
        [str(SCRIPT), "B", "--metrics", str(source), "--baseline", str(source),
         "--baseline-label", "explicit", "--out", str(output)],
    )
    assert report.main() == 0
    columns = report.read_existing_xlsx(output)
    assert columns[0][0] == "explicit"
    assert columns[0][1] == columns[1][1] == {"nl_fleurs": 0.10, "nl_mcv": 0.20, "nl_mls": 0.30}


def test_rejects_old_dataset_layout(report, tmp_path):
    output = tmp_path / "old.xlsx"
    report.build_workbook([(report.BASELINE_LABEL, report.BASELINE_METRICS)], output)
    workbook = load_workbook(output)
    workbook["openasr"]["A4"] = "ami"
    workbook.save(output)
    workbook.close()
    with pytest.raises(ValueError, match="incompatible dataset layout at row 4"):
        report.read_existing_xlsx(output)


def test_partial_coverage_deltas_use_matched_datasets(report, tmp_path):
    baseline = {"ami_clean": 0.10, "spgispeech_clean": 0.40, "de_fleurs": 0.10, "de_mcv": 0.30}
    candidate = {"ami_clean": 0.05, "de_fleurs": 0.20}
    output = tmp_path / "partial.xlsx"
    report.build_workbook([("baseline", baseline), ("candidate", candidate)], output)
    workbook = load_workbook(output)
    sheet = workbook["openasr"]
    assert sheet["D4"].value == "=1-C4/B4"
    assert sheet["D13"].value == pytest.approx(0.50)
    assert sheet["B13"].value == pytest.approx(0.25)
    assert sheet["C13"].value == pytest.approx(0.05)
    assert sheet["D13"].comment is None
    assert sheet["D15"].value == "=1-C15/B15"
    assert sheet["D17"].value == pytest.approx(-1.0)
    assert sheet["D39"].value == pytest.approx(-1.0)
    assert sheet["D39"].comment is None
    for row in (5, 10, 12, 16):
        assert sheet.cell(row, 4).value is None
        assert sheet.cell(row, 4).comment is None
    assert sheet.max_row == 52
    assert all(cell.comment is None for row in sheet for cell in row)
    assert all(cell.value is None for cell in sheet[1])
    workbook.close()


def test_zero_baseline_and_zero_candidate(report, tmp_path):
    output = tmp_path / "zeros.xlsx"
    report.build_workbook([
        ("baseline", {"ami_clean": 0.0, "spgispeech_clean": 0.10, "de_fleurs": 0.0}),
        ("candidate", {"ami_clean": 0.05, "spgispeech_clean": 0.0, "de_fleurs": 0.0}),
    ], output)
    workbook = load_workbook(output)
    sheet = workbook["openasr"]
    assert sheet["D4"].value is None
    assert sheet["D4"].comment is None
    assert sheet["D10"].value == "=1-C10/B10"
    assert sheet["C10"].value == 0
    assert sheet["D13"].value == pytest.approx(0.5)
    assert sheet["D39"].value is None
    assert sheet["D39"].comment is None
    workbook.close()


def test_ml_macro_mean_and_werr_use_matched_language_coverage(report, tmp_path):
    baseline = {
        "de_fleurs": 0.10, "de_mcv": 0.30,
        "fr_fleurs": 0.60, "fr_mcv": 0.80, "fr_mls": 0.90,
        "monsoon_hi_in": 0.90,
    }
    candidate = {"de_fleurs": 0.05, "fr_fleurs": 0.15, "fr_mcv": 0.20}
    output = tmp_path / "macro.xlsx"
    report.build_workbook([("baseline", baseline), ("candidate", candidate)], output)
    workbook = load_workbook(output)
    sheet = workbook["openasr"]
    assert sheet["B39"].value == pytest.approx((0.20 + 2.30 / 3 + 0.90) / 3)
    assert sheet["C39"].value == pytest.approx((0.05 + 0.175) / 2)
    assert sheet["D39"].value == pytest.approx(1 - 0.1125 / 0.40)
    assert sheet["D39"].value != pytest.approx(1 - 0.40 / 1.50)
    workbook.close()


def test_no_common_ml_language_leaves_werr_blank(report, tmp_path):
    output = tmp_path / "unmatched.xlsx"
    report.build_workbook([
        ("baseline", {"de_fleurs": 0.10}),
        ("candidate", {"fr_fleurs": 0.20}),
    ], output)
    workbook = load_workbook(output)
    sheet = workbook["openasr"]
    assert sheet["B39"].value == pytest.approx(0.10)
    assert sheet["C39"].value == pytest.approx(0.20)
    assert sheet["D39"].value is None
    workbook.close()


def test_every_model_gets_its_own_delta(report, tmp_path, monkeypatch):
    prior = tmp_path / "prior.xlsx"
    baseline = {"ami_clean": 0.10}
    report.build_workbook([("baseline", baseline), ("better", {"ami_clean": 0.05})], prior)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"ami_clean": 0.20}))
    output = tmp_path / "extended.xlsx"
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "worse", "--metrics", str(candidate),
        "--extend-xlsx", str(prior), "--out", str(output),
    ])
    assert report.main() == 0
    workbook = load_workbook(output)
    sheet = workbook["openasr"]
    assert sheet["E4"].value == "=1-C4/B4"
    assert sheet["F4"].value == "=1-D4/B4"
    assert sheet["E13"].value == pytest.approx(0.5)
    assert sheet["F13"].value == pytest.approx(-1.0)
    assert len(sheet.conditional_formatting) == 2
    for key in sheet.conditional_formatting:
        rules = sheet.conditional_formatting[key]
        assert rules[0].colorScale.cfvo[1].val == 0
    workbook.close()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1])
def test_invalid_error_rate_is_rejected(report, tmp_path, value):
    with pytest.raises(ValueError, match="finite, nonnegative"):
        report.build_workbook([("baseline", {"ami_clean": value})], tmp_path / "invalid.xlsx")


def test_metadata_round_trip_and_extension(report, tmp_path, monkeypatch):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({
        "ami_clean": 0.05, "de_fleurs": 0.02,
        "urgent2024": 0.09,
    }))
    model_path = tmp_path / "model.json"
    result_path = tmp_path / "results.json"
    prior = tmp_path / "prior.xlsx"
    first = {
        "model_path": "az://account/container/model1/qwen_hf",
        "config": "recipe/phimm/config/eval/example.yaml",
        "node": "verl-n4-i4",
        "ray_job_id": "raysubmit_first",
        "wandb_url": "https://wandb.example/project/runs/first",
        "code_snapshot": "abc123",
    }
    first_results = {
        "ami_clean": "az://account/container/model1/ami/0.jsonl",
        "de_fleurs": "/results/model1/de_fleurs/result_details.jsonl",
        "urgent2024": "/results/model1/urgent2024/0.jsonl",
    }
    model_path.write_text(json.dumps(first))
    result_path.write_text(json.dumps({**first_results, "fleurs_de": first_results["de_fleurs"]}))
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "first", "--metrics", str(metrics),
        "--model-info", str(model_path), "--dataset-results", str(result_path), "--out", str(prior),
    ])
    assert report.main() == 0
    info, results = report.read_existing_metadata(prior)
    assert info["first"] == first
    assert info["r2611v1"] == {}
    assert results == {"first": first_results}
    second = {"model_path": "/models/second", "ray_job_id": "raysubmit_second"}
    second_results = {"ami_clean": "/results/second/ami/0.jsonl"}
    model_path.write_text(json.dumps(second))
    result_path.write_text(json.dumps(second_results))
    output = tmp_path / "extended.xlsx"
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "second", "--metrics", str(metrics), "--extend-xlsx", str(prior),
        "--model-info", str(model_path), "--dataset-results", str(result_path), "--out", str(output),
    ])
    assert report.main() == 0
    info, results = report.read_existing_metadata(output)
    assert info == {"r2611v1": {}, "first": first, "second": second}
    assert results == {"first": first_results, "second": second_results}
    workbook = load_workbook(output)
    assert workbook.sheetnames == ["openasr", "model_info", "dataset_results"]
    assert workbook["openasr"].max_row == 52
    assert workbook["model_info"].max_row == 4
    assert workbook["dataset_results"].max_row == 100
    assert all(cell.comment is None for sheet in workbook for row in sheet for cell in row)
    workbook.close()


def test_metric_only_workbook_metadata_is_optional(report, tmp_path):
    path = tmp_path / "legacy.xlsx"
    report.build_workbook([("baseline", {"ami_clean": 0.1})], path)
    workbook = load_workbook(path)
    del workbook["model_info"]
    del workbook["dataset_results"]
    workbook.save(path)
    workbook.close()
    assert report.read_existing_metadata(path) == ({}, {})
    assert report.read_existing_xlsx(path) == [("baseline", {"ami_clean": 0.1})]


def test_unknown_metadata_is_rejected(report, tmp_path):
    with pytest.raises(ValueError, match="unknown model information fields"):
        report.build_workbook([("candidate", {})], tmp_path / "bad.xlsx",
                              model_info={"candidate": {"model_paht": "wrong key"}})
    with pytest.raises(ValueError, match="unknown dataset result names"):
        report.build_workbook([("candidate", {})], tmp_path / "bad.xlsx",
                              dataset_results={"candidate": {"wrong_dataset": "/some/result.jsonl"}})


def test_openlong_logs_metadata_and_extension(report, tmp_path, monkeypatch):
    baseline = {ds: round(SUPPLIED[ds][0] / 100, 8) for ds in report.OPENLONG_DATASETS}
    candidate = {ds: value / 2 for ds, value in baseline.items()}
    baseline_source = tmp_path / "baseline.json"
    baseline_source.write_text(json.dumps(baseline))
    logs = tmp_path / "eval.log"
    logs.write_text("\n".join(
        f"val-aux/{ds}/p_err/mean@1:{value}" for ds, value in candidate.items()
    ))
    results = {
        ds: f"az://account/container/candidate/val_data_gen/{ds}/0.jsonl"
        for ds in candidate
    }
    result_file = tmp_path / "results.json"
    result_file.write_text(json.dumps(results))
    prior = tmp_path / "openlong.xlsx"
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "candidate", "--from-text", str(logs),
        "--baseline", str(baseline_source),
        "--dataset-results", str(result_file), "--out", str(prior),
    ])
    assert report.main() == 0
    assert report.read_existing_metadata(prior)[1] == {"candidate": results}
    assert report.read_existing_xlsx(prior)[1] == ("candidate", candidate)
    output = tmp_path / "extended.xlsx"
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "next", "--from-text", str(logs),
        "--extend-xlsx", str(prior), "--out", str(output),
    ])
    assert report.main() == 0
    assert report.read_existing_metadata(output)[1] == {"candidate": results}
    workbook = load_workbook(output)
    sheet = workbook["openasr"]
    assert sheet["A40"].value == "OpenLong"
    assert sheet["A44"].value == "avg"
    assert sheet["B44"].value == pytest.approx(sum(baseline.values()) / 3)
    for row in range(41, 44):
        assert sheet.cell(row, 5).value == f"=1-C{row}/B{row}"
        assert sheet.cell(row, 6).value == f"=1-D{row}/B{row}"
    for column in range(2, 7):
        for row in (44,):
            cell = sheet.cell(row, column)
            assert cell.number_format == "0.00%"
            assert cell.alignment.horizontal == "center"
            assert cell.fill == report.OVERALL_AVG_FILL
    assert sheet["E44"].value == pytest.approx(0.5)
    assert sheet["F44"].value == pytest.approx(0.5)
    assert sheet["C13"].value is None
    assert sheet["C39"].value is None
    for key in sheet.conditional_formatting:
        assert str(key.sqref).endswith("52")
        assert sheet.conditional_formatting[key][0].colorScale.cfvo[1].val == 0
    workbook.close()


@pytest.mark.parametrize("candidate, expected_werr", [
    ({"earnings21": 0.05}, 0.5),
    ({"earnings21": 0.20}, -1.0),
    ({"earnings21": 0.0}, 1.0),
    ({"coraal": 0.05}, None),
])
def test_openlong_partial_coverage_and_zero_baseline(report, tmp_path, candidate, expected_werr):
    output = tmp_path / "partial.xlsx"
    report.build_workbook([
        ("baseline", {"earnings21": 0.10, "earnings22": 0.30, "coraal": 0.0}),
        ("candidate", candidate),
    ], output)
    workbook = load_workbook(output)
    sheet = workbook["openasr"]
    assert sheet["B44"].value == pytest.approx(0.40 / 3)
    assert sheet["C44"].value == pytest.approx(next(iter(candidate.values())))
    if expected_werr is None:
        assert sheet["D44"].value is None
    else:
        assert sheet["D44"].value == pytest.approx(expected_werr)
        assert sheet["D41"].value == "=1-C41/B41"
    assert sheet["D42"].value is None
    assert sheet["D43"].value is None
    workbook.close()


def test_extend_pre_openlong_report_preserves_existing_values(report, tmp_path, monkeypatch):
    prior = tmp_path / "legacy.xlsx"
    columns = [("baseline", {"ami_clean": 0.1}), ("first", {"ami_clean": 0.05})]
    report.build_workbook(columns, prior)
    workbook = load_workbook(prior)
    workbook["openasr"].delete_rows(40, 16)
    workbook.save(prior)
    workbook.close()
    assert report.read_existing_xlsx(prior) == columns
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"earnings21": 0.05}))
    output = tmp_path / "extended.xlsx"
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "long", "--metrics", str(metrics),
        "--extend-xlsx", str(prior), "--out", str(output),
    ])
    assert report.main() == 0
    assert report.read_existing_xlsx(output) == columns + [("long", {"earnings21": 0.05})]
    workbook = load_workbook(output)
    assert workbook["openasr"]["B41"].value is None
    assert workbook["openasr"]["F41"].value is None
    workbook.close()


def test_rejects_mismatched_openlong_layout(report, tmp_path):
    output = tmp_path / "bad.xlsx"
    report.build_workbook([("baseline", report.BASELINE_METRICS)], output)
    workbook = load_workbook(output)
    workbook["openasr"]["A43"] = "earnings22-cleaned-aa-chunked"
    workbook.save(output)
    workbook.close()
    with pytest.raises(ValueError, match="incompatible dataset layout at row 43"):
        report.read_existing_xlsx(output)


@pytest.mark.parametrize("candidate, expected_mean, expected_werr", [
    ({"earnings22_cleaned_aa": 0.99}, None, None),
    ({"earnings21": 0.05, "earnings22_cleaned_aa": 0.99}, 0.05, 0.5),
    ({"earnings21": 0.20, "earnings22": 0.0, "earnings22_cleaned_aa": 0.99}, 0.10, 0.5),
])
def test_openlong_avg_excludes_cleaned_and_matches_coverage(
    report, tmp_path, candidate, expected_mean, expected_werr
):
    output = tmp_path / "avg.xlsx"
    baseline = {"earnings21": 0.10, "earnings22": 0.30, "coraal": 0.20, "earnings22_cleaned_aa": 0.01}
    report.build_workbook([("baseline", baseline), ("candidate", candidate)], output)
    workbook = load_workbook(output)
    sheet = workbook["openasr"]
    assert sheet["B44"].value == pytest.approx(0.20)
    if expected_mean is None:
        assert sheet["C44"].value is None
        assert sheet["D44"].value is None
    else:
        assert sheet["C44"].value == pytest.approx(expected_mean)
        assert sheet["D44"].value == pytest.approx(expected_werr)
    workbook.close()


def test_extend_drops_redundant_avg0_and_preserves_metrics_and_metadata(report, tmp_path):
    prior = tmp_path / "prior.xlsx"
    columns = [("baseline", report.BASELINE_METRICS), ("candidate", {"earnings21": 0.05})]
    info = {"baseline": {}, "candidate": {"model_path": "az://account/container/candidate"}}
    results = {"candidate": {"earnings21": "az://account/container/results/earnings21/0.jsonl"}}
    report.build_workbook(columns, prior, model_info=info, dataset_results=results)
    workbook = load_workbook(prior)
    workbook["openasr"].append(["avg0", 0.1076, 0.05])
    workbook.save(prior)
    workbook.close()
    loaded = report.read_existing_xlsx(prior)
    loaded_info, loaded_results = report.read_existing_metadata(prior)
    assert loaded == columns
    assert (loaded_info, loaded_results) == (info, results)
    output = tmp_path / "updated.xlsx"
    report.build_workbook(loaded, output, model_info=loaded_info, dataset_results=loaded_results)
    assert report.read_existing_xlsx(output) == columns
    assert report.read_existing_metadata(output) == (info, results)
    workbook = load_workbook(output)
    assert workbook["openasr"].max_row == 52
    assert "avg0" not in [row[0].value for row in workbook["openasr"]]
    assert workbook["openasr"]["C44"].value == pytest.approx(0.05)
    workbook.close()


def test_r2611v1_default_scores_and_historical_extension(report, tmp_path, monkeypatch):
    candidate = {"ami_clean": 0.06, "spgispeech_clean": 0.02,
                 "urgent2024_clean": 0.04, "earnings22_cleaned_aa": 0.05,
                 "earnings21": 0.09}
    metrics = tmp_path / "candidate.json"
    metrics.write_text(json.dumps(candidate))
    output = tmp_path / "current.xlsx"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "candidate", "--metrics", str(metrics),
                                    "--out", str(output)])
    assert report.main() == 0
    workbook = load_workbook(output)
    sheet = workbook["openasr"]
    assert sheet["B2"].value == "r2611v1"
    assert sheet["B4"].value == pytest.approx(0.0668)
    assert sheet["B10"].value == pytest.approx(0.0210)
    assert sheet["D10"].value == "=1-C10/B10"
    assert sheet["B5"].value == pytest.approx(0.0526)
    assert "earnings22_cleaned_aa" not in [row[0].value for row in sheet]
    assert sheet["B13"].value == pytest.approx(0.3965 / 9)
    assert sheet["D13"].value == pytest.approx(1 - 0.08 / 0.0878)
    assert sheet["B39"].value == pytest.approx(3.8195238095238095 / 100)
    assert sheet["B44"].value == pytest.approx(0.1076)
    assert sheet["D44"].value == pytest.approx(1 - 0.09 / 0.0921)
    assert report.BASELINE_SUPPLEMENTARY_METRICS == {"hy_fleurs": 0.0638, "hy_mcv": 0.119}
    workbook.close()

    historical = tmp_path / "historical.xlsx"
    old_baseline = {key: values[0] / 100 for key, values in SUPPLIED.items()}
    report.build_workbook([("2609r2", old_baseline)], historical)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "candidate", "--metrics", str(metrics),
                                    "--extend-xlsx", str(historical), "--out", str(output)])
    assert report.main() == 0
    columns = report.read_existing_xlsx(output)
    assert columns[0][0] == "2609r2"
    assert columns[0][1] == pytest.approx(old_baseline)


def test_retired_urgent_clean_is_removed_from_new_and_extended_reports(report, tmp_path, monkeypatch):
    prior = tmp_path / "prior.xlsx"
    columns = [("historical", {"urgent2024": 0.10, "urgent2024_clean": 0.90,
                               "de_fleurs": 0.03, "earnings21": 0.20})]
    info = {"historical": {"model_path": "/models/historical"}}
    paths = {"historical": {"urgent2024": "/results/urgent/0.jsonl",
                             "urgent2024_clean": "/results/clean/0.jsonl"}}
    report.build_workbook(columns, prior, model_info=info, dataset_results=paths)
    workbook = load_workbook(prior)
    sheet = workbook["openasr"]
    assert "urgent2024_clean" not in [row[0].value for row in sheet]
    assert sheet["B13"].value == pytest.approx(0.10)
    assert "urgent2024_clean" not in [row[1].value for row in workbook["dataset_results"]]

    # Simulate the former layout, including retired metrics and result metadata.
    sheet.insert_rows(13)
    sheet["A13"] = "urgent2024_clean"
    sheet["B13"] = 0.90
    sheet["B14"] = 0.50
    workbook["dataset_results"].append(["historical", "urgent2024_clean", "/results/clean/0.jsonl"])
    workbook.save(prior)
    workbook.close()
    expected = {key: value for key, value in columns[0][1].items() if key != "urgent2024_clean"}
    assert report.read_existing_xlsx(prior) == [("historical", expected)]

    metrics = tmp_path / "candidate.json"
    metrics.write_text(json.dumps({"urgent2024": 0.05, "urgent2024_clean": 0.99}))
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"urgent2024": "/candidate/urgent/0.jsonl",
                                   "urgent2024_clean": "/candidate/clean/0.jsonl"}))
    output = tmp_path / "extended.xlsx"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "candidate", "--metrics", str(metrics),
                                    "--dataset-results", str(results),
                                    "--extend-xlsx", str(prior), "--out", str(output)])
    assert report.main() == 0
    assert report.read_existing_xlsx(output) == [("historical", expected), ("candidate", {"urgent2024": 0.05})]
    final_info, final_results = report.read_existing_metadata(output)
    assert final_info["historical"] == info["historical"]
    assert final_results == {"historical": {"urgent2024": "/results/urgent/0.jsonl"},
                             "candidate": {"urgent2024": "/candidate/urgent/0.jsonl"}}
    workbook = load_workbook(output)
    sheet = workbook["openasr"]
    assert sheet.max_row == 52
    assert sheet["A13"].value == "avg"
    assert sheet["B13"].value == pytest.approx(0.10)
    assert sheet["C13"].value == pytest.approx(0.05)
    assert sheet["D13"].value == pytest.approx(0.50)
    assert sheet["B15"].value == pytest.approx(0.03)
    assert sheet["B41"].value == pytest.approx(0.20)
    workbook.close()


def test_confirmed_earnings22_cleaned_alias(report):
    assert report.dataset_key("earnings22-cleaned-aa") == "earnings22-cleaned-aa-chunked"
    assert report.canonical_mapping({"earnings22-cleaned-aa": 0.0526}) == {
        "earnings22-cleaned-aa-chunked": 0.0526}
    assert report.parse_wer_lines("val-core/earnings22-cleaned-aa/p_err/mean@1:0.0526") == {
        "earnings22-cleaned-aa-chunked": 0.0526}
    with pytest.raises(ValueError, match="Conflicting inputs"):
        report.canonical_mapping({"earnings22-cleaned-aa": 0.05,
                                  "earnings22-cleaned-aa-chunked": 0.06})


def test_extend_removes_cleaned_earnings22_from_openlong(report, tmp_path, monkeypatch):
    prior = tmp_path / "prior.xlsx"
    report.build_workbook([("r2611v1", report.BASELINE_METRICS)], prior)
    workbook = load_workbook(prior)
    sheet = workbook["openasr"]
    sheet.insert_rows(44)
    sheet["A44"] = "earnings22_cleaned_aa"
    sheet["B44"] = 0.0526
    workbook["dataset_results"].append(["r2611v1", "earnings22_cleaned_aa", "/old/cleaned/0.jsonl"])
    workbook.save(prior)
    workbook.close()
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"earnings21": 0.09, "earnings22_cleaned_aa": 0.99,
                                   "earnings22-cleaned-aa-chunked": 0.05}))
    paths = tmp_path / "paths.json"
    paths.write_text(json.dumps({"earnings22_cleaned_aa": "/candidate/old/0.jsonl",
                                 "earnings22-cleaned-aa-chunked": "/candidate/english/0.jsonl"}))
    output = tmp_path / "new.xlsx"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "candidate", "--metrics", str(metrics),
                                    "--dataset-results", str(paths),
                                    "--extend-xlsx", str(prior), "--out", str(output)])
    assert report.main() == 0
    workbook = load_workbook(output)
    sheet = workbook["openasr"]
    assert sheet.max_row == 52
    assert sheet["B5"].value == pytest.approx(0.0526)
    assert sheet["C5"].value == pytest.approx(0.05)
    assert [sheet.cell(row, 1).value for row in (41, 42, 43)] == ["earnings21", "earnings22", "coraal"]
    assert "earnings22_cleaned_aa" not in [row[0].value for row in sheet]
    for row in (44,):
        assert sheet.cell(row, 2).value == pytest.approx(0.1076)
        assert sheet.cell(row, 3).value == pytest.approx(0.09)
        assert sheet.cell(row, 4).value == pytest.approx(1 - 0.09 / 0.0921)
    assert "earnings22_cleaned_aa" not in [row[1].value for row in workbook["dataset_results"]]
    workbook.close()
    assert report.read_existing_metadata(output)[1] == {
        "candidate": {"earnings22-cleaned-aa-chunked": "/candidate/english/0.jsonl"}}


@pytest.mark.parametrize("source", ["json", "logs"])
def test_librispeech_original_sources_populate_report_and_metadata(report, tmp_path, monkeypatch, source):
    values = {"ls_clean": 0.01, "ls_other": 0.02}
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(values))
    logs = tmp_path / "eval.log"
    logs.write_text("\n".join(f"val-core/{name}/p_err/mean@1:{value}" for name, value in values.items()))
    paths = {name: f"/results/{name}/0.jsonl" for name in values}
    results = tmp_path / "results.json"
    results.write_text(json.dumps(paths))
    output = tmp_path / "report.xlsx"
    inputs = ["--metrics", str(metrics)] if source == "json" else ["--from-text", str(logs)]
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "candidate", *inputs,
                                    "--dataset-results", str(results), "--out", str(output)])
    assert report.main() == 0
    workbook = load_workbook(output)
    sheet = workbook["openasr"]
    for row, name in [(7, "ls_clean"), (8, "ls_other")]:
        assert sheet.cell(row, 1).value == report.dataset_key(name)
        assert sheet.cell(row, 3).value == pytest.approx(values[name])
        assert sheet.cell(row, 4).value == f"=1-C{row}/B{row}"
    assert sheet["C13"].value == pytest.approx(0.015)
    assert report.read_existing_metadata(output)[1]["candidate"] == {
        report.dataset_key(name): path for name, path in paths.items()}
    workbook.close()


@pytest.mark.parametrize("source", ["json", "logs"])
def test_openext_inputs_baseline_averages_and_metadata(report, tmp_path, monkeypatch, source):
    # Unequal coverage distinguishes language-weighted from dataset-weighted means.
    values = {"fleurs_zh": 0.01, "fleurs_vi": 0.02, "mcv_vi": 0.04,
              "eslo_telephony_fr": 0.20, "accueil_ubs_fr": 0.40}
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(values))
    logs = tmp_path / "eval.log"
    logs.write_text("\n".join(f"val-core/{ds}/p_err/mean@1:{v}" for ds, v in values.items()))
    paths = {ds: f"/results/{ds}/0.jsonl" for ds in values}
    results = tmp_path / "results.json"
    results.write_text(json.dumps(paths))
    output = tmp_path / "openext.xlsx"
    inputs = ["--metrics", str(metrics)] if source == "json" else ["--from-text", str(logs)]
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "candidate", *inputs,
                                    "--dataset-results", str(results), "--out", str(output)])
    assert report.main() == 0
    workbook = load_workbook(output)
    sheet = workbook["openasr"]
    assert [sheet.cell(r, 1).value for r in range(45, 53)] == [
        "OpenExt", "zh_fleurs", "zh_aishell1", "zh avg", "vi_fleurs", "vi_mcv",
        "vi avg", "avg"]
    for row, expected in {46: 0.0297, 47: 0.0845, 48: 0.0571,
                          49: 0.0419, 50: 0.0812, 51: 0.06155, 52: 0.059325}.items():
        assert sheet.cell(row, 2).value == pytest.approx(expected)
    assert sheet["C52"].value == pytest.approx((0.01 + 0.03) / 2)
    assert sheet["D52"].value == pytest.approx(1 - 0.02 / ((0.0297 + 0.06155) / 2))
    assert sheet["D48"].value == pytest.approx(1 - 0.01 / 0.0297)
    for row in (46, 49, 50):
        assert sheet.cell(row, 4).value == f"=1-C{row}/B{row}"
    assert sheet["C47"].value is None
    assert sheet["D47"].value is None
    assert sheet.max_row == 52
    assert "eslo_telephony_fr" not in [row[0].value for row in sheet]
    assert "accueil_ubs_fr" not in [row[0].value for row in sheet]
    for row in range(46, 53):
        for col in range(2, 5):
            cell = sheet.cell(row, col)
            assert cell.number_format == "0.00%"
            assert cell.alignment.horizontal == "center"
    assert sheet["A45"].fill == report.HEADER_FILL
    assert sheet["A48"].fill == report.LANG_AVG_FILL
    assert sheet["A52"].fill == report.OVERALL_AVG_FILL
    workbook.close()
    assert report.read_existing_metadata(output)[1] == {
        "candidate": {k: v for k, v in report.canonical_mapping(paths).items()
                      if k not in {"eslo_telephony_fr", "accueil_ubs_fr"}}}
    assert report.read_existing_xlsx(output)[1] == ("candidate", {k: v for k, v in report.canonical_mapping(values).items()
                       if k not in {"eslo_telephony_fr", "accueil_ubs_fr"}})


@pytest.mark.parametrize("baseline, candidate, expected", [
    ({"zh_fleurs": 0.10}, {"zh_fleurs": 0.0}, 1.0),
    ({"zh_fleurs": 0.10}, {"zh_fleurs": 0.20}, -1.0),
    ({"zh_fleurs": 0.0}, {"zh_fleurs": 0.10}, None),
    ({"zh_fleurs": 0.10}, {"vi_fleurs": 0.10}, None),
])
def test_openext_missing_zero_and_negative_werr(report, tmp_path, baseline, candidate, expected):
    output = tmp_path / "edge.xlsx"
    report.build_workbook([("baseline", baseline), ("candidate", candidate)], output)
    workbook = load_workbook(output)
    value = workbook["openasr"]["D52"].value
    assert value is None if expected is None else value == pytest.approx(expected)
    workbook.close()


@pytest.mark.parametrize("legacy", [False, True])
def test_openext_extension_preserves_metrics_metadata_and_missing_baseline(report, tmp_path, monkeypatch, legacy):
    prior = tmp_path / "prior.xlsx"
    baseline = {"ami_clean": 0.1}
    first = {"ami_clean": 0.05} if legacy else {"zh_fleurs": 0.02}
    info = {"first": {"model_path": "/models/first"}}
    results = {"first": {next(iter(first)): "/results/first/0.jsonl"}}
    report.build_workbook([("historical", baseline), ("first", first)], prior,
                          model_info=info, dataset_results=results)
    if legacy:
        workbook = load_workbook(prior)
        workbook["openasr"].delete_rows(45, 11)
        workbook.save(prior)
        workbook.close()
    metrics = tmp_path / "candidate.json"
    metrics.write_text(json.dumps({"zh_fleurs": 0.01, "vi_mcv": 0.03}))
    output = tmp_path / "extended.xlsx"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "next", "--metrics", str(metrics),
                                    "--extend-xlsx", str(prior), "--out", str(output)])
    assert report.main() == 0
    assert report.read_existing_xlsx(output) == [
        ("historical", baseline), ("first", first), ("next", {"zh_fleurs": 0.01, "vi_mcv": 0.03})]
    loaded_info, loaded_results = report.read_existing_metadata(output)
    assert loaded_info["first"] == info["first"]
    assert loaded_results == results
    workbook = load_workbook(output)
    sheet = workbook["openasr"]
    assert sheet["B46"].value is None
    assert sheet["B52"].value is None
    assert sheet["E52"].value is None
    assert sheet["F52"].value is None
    assert sheet["D52"].value == pytest.approx(0.02)
    workbook.close()

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[2]
    / ".github/skills/eval-mix-openml-aa-report/scripts/build_report.py"
)
SPEC = importlib.util.spec_from_file_location("build_mix_openml_aa_report", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

NAME_SCRIPT = (
    Path(__file__).parents[2]
    / ".github/skills/eval-mix-openml-aa-report/scripts/make_experiment_name.py"
)
NAME_SPEC = importlib.util.spec_from_file_location("make_mix_openml_aa_experiment_name", NAME_SCRIPT)
NAME_MODULE = importlib.util.module_from_spec(NAME_SPEC)
assert NAME_SPEC.loader is not None
NAME_SPEC.loader.exec_module(NAME_MODULE)


def test_load_metrics_and_build_workbook(tmp_path):
    baseline_values = {}
    candidate_values = {}
    for index, (dataset, metric) in enumerate(MODULE.EXPECTED_METRIC.items(), start=1):
        baseline_values[f"val-core/{dataset}/{metric}/mean@1"] = index / 100
        candidate_values[f"val-core/{dataset}/{metric}/mean@1"] = index / 200

    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline_values), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate_values), encoding="utf-8")

    baseline = MODULE.load_metrics(baseline_path)
    candidate = MODULE.load_metrics(candidate_path)
    workbook = MODULE.build_workbook(
        baseline, candidate, "reference", "step100", "az://candidate", str(baseline_path), str(candidate_path)
    )

    sheet = workbook["results"]
    assert len(baseline) == 34
    assert sheet["B2"].value == "mixlang_fy26q2"
    assert sheet["F2"].value == "=1-E2/D2"
    assert sheet.max_row == 40
    aa_rows = [
        row
        for row in sheet.iter_rows(min_row=2, values_only=True)
        if row[0] == "AA" and not str(row[1]).endswith(" avg")
    ]
    assert [row[1] for row in aa_rows] == ["earnings22_cleaned_aa", "voxpopuli_cleaned_aa"]
    assert workbook["provenance"]["B2"].value == MODULE.CONFIG_PATH


def test_load_metrics_rejects_incomplete_input(tmp_path):
    source = tmp_path / "incomplete.log"
    source.write_text("val-core/de_fleurs/p_err/mean@1: 0.1", encoding="utf-8")

    try:
        MODULE.load_metrics(source)
    except ValueError as error:
        assert "missing 33 datasets" in str(error)
    else:
        raise AssertionError("incomplete metrics should be rejected")


def test_load_metrics_accepts_quoted_ray_log_keys(tmp_path):
    source = tmp_path / "ray_job.log"
    lines = []
    for index, (dataset, metric) in enumerate(MODULE.EXPECTED_METRIC.items(), start=1):
        lines.append(f"'val-core/{dataset}/{metric}/mean@1': {index}e-3")
    source.write_text("\n".join(lines), encoding="utf-8")

    metrics = MODULE.load_metrics(source)

    assert metrics["earnings22_cleaned_aa"] > 0
    assert metrics["voxpopuli_cleaned_aa"] > 0
    assert len(metrics) == 34


def test_different_checkpoints_get_different_experiment_names():
    step100 = NAME_MODULE.make_experiment_name(
        "remax 2607", "az://models/run/global_step_100/qwen_hf/", "candidate"
    )
    step200 = NAME_MODULE.make_experiment_name(
        "remax 2607", "az://models/run/global_step_200/qwen_hf/", "candidate"
    )

    assert "step100" in step100
    assert "step200" in step200
    assert step100 != step200
    assert step100.startswith("mix-openml-aa-remax-2607-step100-candidate-")
    assert NAME_MODULE.make_experiment_name(
        "remax 2607", "az://models/run/global_step_100/qwen_hf/", "candidate", attempt=2
    ).endswith("-try2")
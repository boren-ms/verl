"""Result acceptance rejects partial or mismatched longform generations."""

import json

import pytest
from recipe.phimm.tools.inhouse_entity_report import audit_rows


def result_fixture():
    expected = [
        {
            "id": "u",
            "data_source": "eer_test",
            "audio_path": ["a", "b"],
            "text": "display",
            "entity_transcription": "<NE> entity </NE>",
        }
    ]
    detail = {
        "parent_info": {"id": "u", "entity_transcription": expected[0]["entity_transcription"]},
        "segment_rows": [{"audio_path": "a"}, {"audio_path": "b"}],
        "segment_outputs": ["first", "second"],
        "previous_contexts": ["", ""],
    }
    row = {
        "asr_data_source": "eer_test",
        "asr_detail": json.dumps(detail),
        "gts": "display",
        "eer_n_err": 1,
        "eer_n_ref": 3,
        "dter_n_err": 2,
        "dter_n_ref": 10,
    }
    return expected, row, detail


def test_complete_counts():
    expected, row, _ = result_fixture()
    assert audit_rows([row], expected) == {
        "parents": 1,
        "segments": 2,
        "eer_n_err": 1,
        "eer_n_ref": 3,
        "dter_n_err": 2,
        "dter_n_ref": 10,
    }


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "reference", "order", "context", "nonfinite"])
def test_incomplete_or_mismatched_results_rejected(corruption):
    expected, row, detail = result_fixture()
    rows = [row]
    if corruption == "missing":
        rows = []
    elif corruption == "duplicate":
        rows.append(row)
    elif corruption == "reference":
        row["gts"] = "wrong reference"
    elif corruption == "order":
        detail["segment_rows"].reverse()
    elif corruption == "context":
        detail["previous_contexts"][1] = "ground truth leak"
    else:
        row["eer_n_err"] = float("nan")
    row["asr_detail"] = json.dumps(detail)
    with pytest.raises(ValueError):
        audit_rows(rows, expected)


def text_history_fixture():
    expected, row, detail = result_fixture()
    expected[0]["audio_path"] = ["a", "b", "c"]
    detail["segment_rows"] = [{"audio_path": path} for path in expected[0]["audio_path"]]
    detail["segment_outputs"] = [
        "<src=English><tgt=English>\n<TXT>First segment.</TXT>",
        "<nonspeech>",
        "new output",
    ]
    detail["previous_contexts"] = ["", "Previous context: First segment.", ""]
    row["asr_detail"] = json.dumps(detail)
    return expected, row, detail


def test_text_h1_audits_cleaned_previous_output_and_empty_turn_reset():
    expected, row, _ = text_history_fixture()
    assert audit_rows([row], expected, context_mode="text_h1")["segments"] == 3


@pytest.mark.parametrize(
    "contexts",
    [
        ["leaked recording", "Previous context: First segment.", ""],
        ["", "wrong text", ""],
        ["", "Previous context: First segment.", "Previous context: First segment."],
        ["", "Previous context: First segment."],
    ],
)
def test_text_h1_rejects_leaked_changed_stale_or_missing_context(contexts):
    expected, row, detail = text_history_fixture()
    detail["previous_contexts"] = contexts
    row["asr_detail"] = json.dumps(detail)
    with pytest.raises(ValueError, match="Unexpected decoding context"):
        audit_rows([row], expected, context_mode="text_h1")


@pytest.mark.parametrize("prefix", ["Previous segment transcript (for context):", "", "  Custom context:  "])
def test_text_h1_audits_custom_prefix(prefix):
    expected, row, detail = text_history_fixture()
    detail["previous_contexts"][1] = f"{prefix.strip()} First segment.".strip()
    row["asr_detail"] = json.dumps(detail)
    assert audit_rows([row], expected, context_mode="text_h1", prefix=prefix)["segments"] == 3
    with pytest.raises(ValueError, match="Unexpected decoding context"):
        audit_rows([row], expected, context_mode="text_h1")


def test_text_h1_rejects_raw_context_without_instruction():
    expected, row, detail = text_history_fixture()
    detail["previous_contexts"][1] = "First segment."
    row["asr_detail"] = json.dumps(detail)
    with pytest.raises(ValueError, match="Unexpected decoding context"):
        audit_rows([row], expected, context_mode="text_h1")


def test_text_context_is_rejected_by_default_independent_audit():
    expected, row, _ = text_history_fixture()
    with pytest.raises(ValueError, match="Unexpected decoding context"):
        audit_rows([row], expected)


def test_ter_only_audit_does_not_require_entity_counts():
    expected, row, _ = result_fixture()
    del row["eer_n_err"], row["eer_n_ref"]
    counts = audit_rows([row], expected, include_eer=False)
    assert counts == dict(parents=1, segments=2, dter_n_err=2, dter_n_ref=10)


@pytest.mark.parametrize("suite", ["eer", "ter"])
def test_report_uses_selected_suite_and_equal_dataset_weight(tmp_path, monkeypatch, suite):
    import sys

    from recipe.phimm.data.inhouse_dtest_enus import DATA_SOURCES, SUITES
    from recipe.phimm.tools import inhouse_entity_report as report

    manifests = tmp_path / "manifests"
    results = tmp_path / "results"
    output = tmp_path / "report"
    manifests.mkdir()
    results.mkdir()
    metrics = {}
    for i, domain in enumerate(SUITES[suite], start=1):
        expected, row, detail = result_fixture()
        source = DATA_SOURCES[suite][domain]
        expected[0]["data_source"] = row["asr_data_source"] = source
        if suite == "eer":
            row["eer_n_ref"] = i
            metrics[f"val-core/{source}/eer_p_err/mean@1"] = 1 / i
        else:
            del row["eer_n_err"], row["eer_n_ref"]
            row["dter_n_err"], row["dter_n_ref"] = 1, i
        metrics[f"val-core/{source}/dter_p_err/mean@1"] = row["dter_n_err"] / row["dter_n_ref"]
        (manifests / (domain + ".jsonl")).write_text(json.dumps(expected[0]) + "\n")
        (results / source).mkdir()
        (results / source / "0.jsonl").write_text(json.dumps(row) + "\n")
    (results / "0.metrics.json").write_text(json.dumps({"metrics": metrics}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report",
            "--suite",
            suite,
            "--manifest-root",
            str(manifests),
            "--result-root",
            str(results),
            "--output-dir",
            str(output),
        ],
    )
    report.main()
    summary = json.loads((output / "summary.json").read_text())
    metric = "eer" if suite == "eer" else "dter"
    assert summary["primary_" + metric] == pytest.approx(sum(1 / i for i in range(1, 6)) / 5)
    assert summary["total"][metric] == pytest.approx(5 / 15)
    assert len(summary["datasets"]) == 5
    assert (output / "report.md").exists() and (output / "datasets.csv").exists()
    if suite == "ter":
        assert "eer" not in summary["total"]


def test_language_hint_audits_every_segment_base_prompt():
    expected, row, detail = result_fixture()
    from recipe.phimm.data.prompts import get_task_prompt

    prompt = get_task_prompt(style="display", language="English", lang_hint=True)
    for segment in detail["segment_rows"]:
        segment["prompt"] = [{"role": "user", "content": prompt}]
    row["asr_detail"] = json.dumps(detail)
    assert audit_rows([row], expected, expected_prompt=prompt)["segments"] == 2


@pytest.mark.parametrize("bad_prompt", [None, "Transcribe the audio clip into text.<audio>", "The language is German."])
def test_missing_or_changed_language_hint_is_rejected(bad_prompt):
    expected, row, detail = result_fixture()
    from recipe.phimm.data.prompts import get_task_prompt

    prompt = get_task_prompt(style="display", language="English", lang_hint=True)
    detail["segment_rows"][0]["prompt"] = [{"role": "user", "content": prompt}]
    detail["segment_rows"][1]["prompt"] = [{"role": "user", "content": bad_prompt}]
    row["asr_detail"] = json.dumps(detail)
    with pytest.raises(ValueError, match="base prompt/language hint"):
        audit_rows([row], expected, expected_prompt=prompt)

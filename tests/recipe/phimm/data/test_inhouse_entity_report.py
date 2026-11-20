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
    detail["previous_contexts"] = ["", "First segment.", ""]
    row["asr_detail"] = json.dumps(detail)
    return expected, row, detail


def test_text_h1_audits_cleaned_previous_output_and_empty_turn_reset():
    expected, row, _ = text_history_fixture()
    assert audit_rows([row], expected, context_mode="text_h1")["segments"] == 3


@pytest.mark.parametrize(
    "contexts",
    [
        ["leaked recording", "First segment.", ""],
        ["", "wrong text", ""],
        ["", "First segment.", "First segment."],
        ["", "First segment."],
    ],
)
def test_text_h1_rejects_leaked_changed_stale_or_missing_context(contexts):
    expected, row, detail = text_history_fixture()
    detail["previous_contexts"] = contexts
    row["asr_detail"] = json.dumps(detail)
    with pytest.raises(ValueError, match="Unexpected decoding context"):
        audit_rows([row], expected, context_mode="text_h1")


def test_text_context_is_rejected_by_default_independent_audit():
    expected, row, _ = text_history_fixture()
    with pytest.raises(ValueError, match="Unexpected decoding context"):
        audit_rows([row], expected)

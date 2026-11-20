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

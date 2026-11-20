"""Rescoring consumes the exact saved recording outputs and original references."""

import json
from unittest.mock import Mock

import pytest
from recipe.phimm.tools import inhouse_entity_rescore as rescore


def fixture():
    expected = dict(
        id="recording",
        data_source="banking",
        audio_path=["a", "b"],
        text="Hello <NE> John Smith </NE>.",
        entity_transcription="original entities",
    )
    detail = dict(
        parent_info=dict(id="recording", entity_transcription="original entities"),
        segment_rows=[dict(audio_path="a"), dict(audio_path="b")],
        segment_outputs=["<src=English><tgt=English>\n<TXT>Hello</TXT>", "John Smith."],
        previous_contexts=["", "Previous context: Hello"],
    )
    row = dict(asr_data_source="banking", gts=expected["text"], asr_detail=json.dumps(detail), eer_n_err=1, eer_n_ref=1)
    return row, expected, detail


def test_rescore_joins_saved_segments_and_scores_parent_once(monkeypatch):
    row, expected, _ = fixture()
    saved = dict(row)
    scorer = Mock(return_value=(0, 1, 0.0))
    monkeypatch.setattr(rescore.measure, "_compute_eer", scorer)
    result = rescore.rescore_row(row, expected, "/pack")
    scorer.assert_called_once_with(
        expected["text"], "Hello John Smith.", "/pack", strict=True, mode="nondisfluency", locale="en-us"
    )
    assert row == saved
    assert result["old_eer_n_err"] == 1 and result["eer_n_err"] == 0
    assert result["eer_n_ref"] == 1 and result["segments"] == 2
    assert len(result["hypothesis_sha256"]) == 64


@pytest.mark.parametrize("corruption", ["id", "source", "reference", "entity", "order", "missing"])
def test_rescore_rejects_changed_or_incomplete_inputs(monkeypatch, corruption):
    row, expected, detail = fixture()
    if corruption == "id":
        detail["parent_info"]["id"] = "other"
    elif corruption == "source":
        row["asr_data_source"] = "other"
    elif corruption == "reference":
        row["gts"] = "other"
    elif corruption == "entity":
        detail["parent_info"]["entity_transcription"] = "other"
    elif corruption == "order":
        detail["segment_rows"].reverse()
    else:
        detail["segment_outputs"].pop()
    row["asr_detail"] = json.dumps(detail)
    scorer = Mock()
    monkeypatch.setattr(rescore.measure, "_compute_eer", scorer)
    with pytest.raises(ValueError):
        rescore.rescore_row(row, expected, "/pack")
    scorer.assert_not_called()

"""Rescoring consumes the exact saved recording outputs and original references."""

import json
import sys
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


def test_main_accumulates_details_across_datasets(monkeypatch, tmp_path):
    row, expected, detail = fixture()
    compact_detail = {
        "extra_info": detail["parent_info"],
        "asr_segments": detail["segment_rows"],
        "segment_outputs": detail["segment_outputs"],
        "previous_contexts": detail["previous_contexts"],
    }
    row["asr_detail"] = json.dumps(compact_detail)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(expected) + "\n")
    detail_path = tmp_path / "0.jsonl"
    detail_path.write_text(json.dumps(row) + "\n")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "suite": "eer",
                "result_root": str(tmp_path),
                "model_label": "test",
                "prompt_style": "display",
                "context_mode": "independent",
                "language_hint": "English",
                "primary_eer": 1.0,
                "total": {"parents": 1, "segments": 2},
                "datasets": [
                    {
                        "dataset": "banking",
                        "manifest": str(manifest),
                        "detail_path": str(detail_path),
                        "parents": 1,
                        "segments": 2,
                        "eer_n_err": 1,
                        "eer_n_ref": 1,
                        "eer": 1.0,
                    }
                ],
            }
        )
    )
    out_dir = tmp_path / "out"
    monkeypatch.setattr(rescore.measure, "ensure_pack_dir", lambda path: path)
    monkeypatch.setattr(rescore.measure, "_compute_eer", lambda *args, **kwargs: (0, 1, 0.0))
    monkeypatch.setattr(
        sys, "argv", ["inhouse_entity_rescore", "--summary", str(summary_path), "--output-dir", str(out_dir)]
    )
    rescore.main()
    saved_details = [json.loads(line) for line in (out_dir / "details.jsonl").read_text().splitlines()]
    saved_summary = json.loads((out_dir / "summary.json").read_text())
    assert len(saved_details) == 1 and saved_details[0]["id"] == "recording"
    assert saved_summary["changed_recordings"] == 1

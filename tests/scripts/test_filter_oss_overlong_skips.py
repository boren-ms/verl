import argparse
import json
import sqlite3

import pytest

from scripts.filter_oss_2611r2_errors import run


def inputs(tmp_path):
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps([{"audio_key": "audio", "count": 2, "language": "en",
                                    "family": "test", "split_status": "train"}]))
    generation = tmp_path / "generation"
    generation.mkdir()
    skipped = {"audio_chunk": "audio:2:0", "text": "original reference",
               "generation_status": "skipped_overlong_input", "input_tokens": 9,
               "max_input_tokens": 8, "n_err": None, "n_ref": None,
               "raw_response": None, "response": None}
    decoded = {"audio_chunk": "audio:2:1", "text": "other original reference",
               "n_err": 3, "n_ref": 4, "selection_scorer": "oss_error_mining_v1",
               "selection_metric": "wer"}
    args = argparse.Namespace(sources=sources, generation=str(generation), out=tmp_path / "out",
                              eligibility=None, allow_overlong_skips=True)
    return args, skipped, decoded


def test_explicit_skips_complete_accounting_but_never_become_scored_errors(tmp_path):
    args, skipped, decoded = inputs(tmp_path)
    (tmp_path / "generation/part-000.jsonl").write_text("\n".join(map(json.dumps, [skipped, decoded])))
    run(args)
    coverage = json.loads((args.out / "coverage.json").read_text())
    assert coverage["observed_unique"] == 2
    assert coverage["decoded"] == coverage["skipped_overlong_inputs"] == 1
    assert coverage["n_err_positive_before_split_audit"] == 1
    with sqlite3.connect(args.out / "scores.sqlite") as db:
        assert db.execute("SELECT id FROM scores WHERE n_err>0").fetchall() == [("audio:2:1",)]
        assert db.execute("SELECT n_err FROM scores WHERE id='audio:2:0'").fetchone() == (None,)


@pytest.mark.parametrize("change,allow,message", [
    ({}, False, "Explicit --allow-overlong-skips"),
    ({"input_tokens": 8}, True, "Invalid overlong-input"),
    ({"input_tokens": True}, True, "Invalid overlong-input"),
    ({"n_err": 0}, True, "Invalid overlong-input"),
    ({"raw_response": "invented"}, True, "Invalid overlong-input"),
])
def test_filter_rejects_unapproved_or_malformed_skip_records(tmp_path, change, allow, message):
    args, skipped, decoded = inputs(tmp_path)
    args.allow_overlong_skips = allow
    skipped.update(change)
    (tmp_path / "generation/part-000.jsonl").write_text("\n".join(map(json.dumps, [skipped, decoded])))
    with pytest.raises(ValueError, match=message):
        run(args)

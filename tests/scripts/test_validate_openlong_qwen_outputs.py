import json

import pytest

from scripts import validate_openlong_qwen_outputs as validator


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_checks_coverage_hypotheses_and_reference_words(tmp_path, monkeypatch):
    monkeypatch.setitem(validator.EXPECTED, "test", (2, 1))
    source = [{"id": f"s{i}", "parent_audio_path": "p", "seg_index": i,
               "seg_start": i, "seg_end": i + 1, "n_segments": 2,
               "audio_path": f"audio#{i}", "baseline_hypothesis": "hyp", "text": text}
              for i, text in enumerate(("one two", "three"))]
    output = [dict(row, prior_inferred_text=row["text"], alignment_method="qwen3_forced_aligner_0.6b",
                   alignment_status="qwen_candidate", alignment_flags=[]) for row in source]
    source_path, output_path = tmp_path / "source", tmp_path / "output"
    write_rows(source_path, source)
    write_rows(output_path, output)
    assert validator.validate(source_path, output_path, "test")["segments"] == 2
    output[0]["text"] = "one"
    write_rows(output_path, output)
    with pytest.raises(ValueError, match="words lost"):
        validator.validate(source_path, output_path, "test")

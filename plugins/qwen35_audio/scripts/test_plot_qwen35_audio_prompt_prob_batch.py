import json
import math

import pytest

from plot_qwen35_audio_prompt_prob_batch import (
    orange_audio_path,
    read_examples,
    token_probability_rows,
    write_report,
)


def test_orange_audio_path_maps_librispeech_suffix():
    source = "/datablob1/users/example/DATA/LibriSpeech/test-other/1/2/1-2-3.flac"
    assert orange_audio_path(source).endswith("/LibriSpeech/test-other/1/2/1-2-3.flac")


def test_read_examples_selects_first_valid_records(tmp_path):
    manifest = tmp_path / "input.jsonl"
    records = [
        {"id": "one", "audio_path": "/x/LibriSpeech/test-other/a.flac", "text": "alpha", "keywords": ["alpha"]},
        {"id": "two", "audio_path": "/x/LibriSpeech/test-other/b.flac", "text": "beta", "keywords": ["beta"]},
    ]
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    examples = read_examples(manifest, 2)
    assert [example["id"] for example in examples] == ["one", "two"]
    assert examples[0]["line_number"] == 1


def test_token_probability_rows_include_only_txt_content():
    rows = token_probability_rows(
        ["<TXT>", "wrong", " word", "</TXT>"],
        [-1.0, math.log(0.8), math.log(0.6), -1.0],
        [-1.0, math.log(0.2), math.log(0.9), -1.0],
    )
    assert [row["text"] for row in rows] == ["wrong", " word"]
    assert rows[0]["teacher_minus_student_probability"] == pytest.approx(-0.6)
    assert rows[1]["teacher_minus_student_probability"] == pytest.approx(0.3)


def test_write_report_creates_artifacts(tmp_path):
    output = tmp_path / "report.png"
    report = {
        "examples": [
            {
                "id": "one",
                "keywords": ["alpha"],
                "student_transcript": "alfa",
                "tokens": [
                    {"text": "al", "teacher_minus_student_probability": -0.4},
                    {"text": "fa", "teacher_minus_student_probability": 0.2},
                ],
            }
        ]
    }
    json_path = write_report(output, report)
    assert output.stat().st_size > 0
    assert json.loads(json_path.read_text())["examples"][0]["id"] == "one"
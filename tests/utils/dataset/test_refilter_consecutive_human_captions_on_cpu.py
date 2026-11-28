# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "consecutive", Path(__file__).resolve().parents[3] / "scripts/refilter_consecutive_human_captions.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def parent(durations):
    return {
        "id": "chunk:0",
        "chunk_name": "chunk",
        "audio_chunk": "az://account/data/chunk.audio:4:0",
        "text": " ".join(f"word{i}" for i in range(len(durations))),
        "alignment_audio_duration": len(durations) * 7.0,
        "transcription_source": "display_human_caption_adjusted",
        "word_timestamps": [
            {"text": f"word{i}", "start": i * 7.0, "end": i * 7.0 + d} for i, d in enumerate(durations)
        ],
    }


@pytest.mark.parametrize(
    "durations,passed,max_zero,max_long",
    [
        ([0, 0, 0, 1], True, 3, 0),
        ([0, 0, 0, 0], False, 4, 0),
        ([6, 6, 6, 1], True, 0, 3),
        ([6, 6, 6, 6], False, 0, 4),
        ([0, 0, 0, 6, 6, 6], True, 3, 3),
        ([0, 6, 0, 6, 0, 6, 0, 6], True, 1, 1),
        ([0, 0, 0, 5, 0, 0, 0], True, 3, 0),
        ([6, 6, 6, 5, 6, 6, 6], True, 0, 3),
    ],
)
def test_separate_maximal_runs_and_exact_five_boundary(durations, passed, max_zero, max_long):
    old = parent(durations)
    r = runner.filter_consecutive_words(old, maximum_consecutive=3, long_seconds=5)
    q = r["word_duration_filter"]
    assert q["passed"] == passed
    assert q["maximum_runs"] == {"zero": max_zero, "above_5": max_long}
    assert r["word_timestamps"] == old["word_timestamps"]


def test_decimal_endpoint_duration_does_not_turn_five_seconds_into_long_word():
    old = parent([1] * 4)
    for w in old["word_timestamps"]:
        w.update(start=4.94, end=9.94)
    q = runner.filter_consecutive_words(old, maximum_consecutive=3, long_seconds=5)["word_duration_filter"]
    assert q["maximum_runs"]["above_5"] == 0


def test_retained_alignment_error_is_never_recovered_by_duration_policy():
    old = parent([1, 1])
    old["alignment_error"] = "existing bounds failure"
    assert not runner.filter_consecutive_words(old, maximum_consecutive=3, long_seconds=5)["word_duration_filter"][
        "passed"
    ]


def test_derivation_keeps_source_bytes_word_coverage_and_all_decisions(tmp_path):
    raw = tmp_path / "raw_segments.jsonl"
    rows = []
    for i, durations in enumerate([[1] * 10, [0] * 3 + [1] * 7, [0] * 4 + [1] * 6]):
        r = parent(durations)
        r["id"] = f"chunk:{i}"
        rows.append(runner.alignment.filter_word_durations(runner.alignment.segment_row(r)))
    with raw.open("w") as f:
        for row in rows:
            runner.alignment.write_row(f, row)
    before = raw.read_bytes()
    receipt = tmp_path / "ALIGNMENT_COMPLETE.json"
    receipt.write_text(
        json.dumps(
            {
                "rows": 3,
                "sha256": runner.alignment.fingerprint(raw),
                "runner_sha256": runner.alignment.fingerprint(Path(runner.alignment.__file__)),
            }
        )
    )
    out = tmp_path / "v3"
    runner.derive(
        SimpleNamespace(
            input=raw, source_receipt=receipt, output_dir=out, maximum_long_word_seconds=5, maximum_consecutive_words=3
        )
    )
    assert raw.read_bytes() == before
    c = json.loads((out / "COMPLETE.json").read_text())
    assert c["counts"]["filtered_originals"] == 2
    assert c["counts"]["reject_zero_run"] == 1
    with gzip.open(out / "filtered_originals.jsonl.gz", "rt") as f:
        accepted = [json.loads(line) for line in f]
    assert accepted[1]["word_duration_filter"]["maximum_runs"]["zero"] == 3
    stats = json.loads((out / "RUN_STATISTICS.json").read_text())
    assert stats["categories"]["zero"]["run_length_histogram"] == {"3": 1, "4": 1}
    assert stats["categories"]["zero"]["word_count_by_run_length"] == {"3": 3, "4": 4}
    with gzip.open(out / "pairs.jsonl.gz", "rt") as f:
        for line in f:
            pair = json.loads(line)
            assert len(pair["audio_path"]) == len(pair["segment_reference_texts"]) == 2
            assert pair["text"] == " ".join(pair["segment_reference_texts"])
    assert not list(out.glob("*.wav"))

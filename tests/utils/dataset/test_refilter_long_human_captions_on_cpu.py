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
    "refilter", Path(__file__).resolve().parents[3] / "scripts/refilter_long_human_captions.py"
)
refilter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refilter)


def parent(durations):
    return {
        "id": "chunk:0",
        "chunk_name": "chunk",
        "audio_chunk": "az://account/data/chunk.audio:3:0",
        "text": " ".join(f"word{i}" for i in range(len(durations))),
        "alignment_audio_duration": len(durations) * 6.0,
        "transcription_source": "display_human_caption_adjusted",
        "word_timestamps": [
            {"text": f"word{i}", "start": i * 6.0, "end": i * 6.0 + duration} for i, duration in enumerate(durations)
        ],
    }


@pytest.mark.parametrize(
    "durations,passed,count", [([0, 1], True, 1), ([5, 1], True, 1), ([0, 5], False, 2), ([0.001, 4.999], True, 0)]
)
def test_open_bounds_and_two_outlier_rejection(durations, passed, count):
    row = refilter.filter_words(parent(durations), minimum=0, maximum=5, maximum_outliers=1)
    assert row["word_duration_filter"]["passed"] == passed
    assert row["word_duration_filter"]["outlier_count"] == count
    assert row["word_timestamps"] == parent(durations)["word_timestamps"]


def test_alignment_failure_and_incomplete_words_cannot_be_recovered():
    row = parent([1, 1])
    row["alignment_error"] = "unaligned human reference"
    assert not refilter.filter_words(row, minimum=0, maximum=5, maximum_outliers=1)["word_duration_filter"]["passed"]
    del row["alignment_error"]
    row["word_timestamps"].pop()
    assert not refilter.filter_words(row, minimum=0, maximum=5, maximum_outliers=1)["word_duration_filter"]["passed"]


def test_derived_versions_preserve_original_raw_and_pair_word_coverage(tmp_path):
    rows = []
    for index, durations in enumerate([[1] * 10, [0] + [1] * 9]):
        row = parent(durations)
        row["id"] = f"chunk:{index}"
        row = refilter.alignment.filter_word_durations(refilter.alignment.segment_row(row))
        rows.append(row)
    failure = {**parent([1, 1]), "id": "chunk:2", "alignment_error": "retained failure", "segments": []}
    rows.append(refilter.alignment.filter_word_durations(failure))
    raw = tmp_path / "raw_segments.jsonl"
    with raw.open("w") as stream:
        for row in rows:
            refilter.alignment.write_row(stream, row)
    original_bytes = raw.read_bytes()
    receipt = tmp_path / "ALIGNMENT_COMPLETE.json"
    receipt.write_text(
        json.dumps(
            {
                "rows": 3,
                "sha256": refilter.alignment.fingerprint(raw),
                "runner_sha256": refilter.alignment.fingerprint(Path(refilter.alignment.__file__)),
            }
        )
    )
    output = tmp_path / "new-version"
    refilter.derive(
        SimpleNamespace(
            input=raw,
            source_receipt=receipt,
            output_dir=output,
            minimum_word_seconds=0,
            maximum_word_seconds=5,
            maximum_outlier_words=1,
        )
    )
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert complete["counts"]["filtered_originals"] == 2
    assert complete["counts"]["newly_accepted_originals"] == 1
    assert complete["counts"]["alignment_failures"] == 1
    assert complete["accepted_outlier_count_histogram"] == {"0": 1, "1": 1}
    assert raw.read_bytes() == original_bytes
    assert not list(output.glob("*.wav"))
    with gzip.open(output / "pairs.jsonl.gz", "rt") as stream:
        pairs = [json.loads(line) for line in stream]
    assert pairs
    for pair in pairs:
        assert len(pair["audio_path"]) == len(pair["segment_reference_texts"]) == 2
        assert pair["text"] == " ".join(pair["segment_reference_texts"])
        assert pair["parent_id"] in {"chunk:0", "chunk:1"}
    with gzip.open(output / "quality_decisions.jsonl.gz", "rt") as stream:
        decisions = [json.loads(line) for line in stream]
    assert len(decisions) == 3
    assert decisions[1]["word_duration_filter"]["outliers"][0]["duration"] == 0

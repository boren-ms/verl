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

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "alignment", Path(__file__).resolve().parents[3] / "scripts/align_long_human_captions.py"
)
alignment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alignment)


def original():
    words = [f"word{i}" for i in range(65)]
    return {
        "id": "chunk:0",
        "audio_chunk": "az://account/data/chunk.audio:100:0",
        "text": " ".join(words),
        "transcription_source": "display_human_caption_adjusted",
        "word_timestamps": [{"text": word, "start": float(i), "end": i + 0.5} for i, word in enumerate(words)],
        "alignment_audio_duration": 65.0,
    }


def test_segments_preserve_words_and_time_selectors_without_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    row = alignment.segment_row(original())
    assert len(row["segments"]) == 3
    assert " ".join(part[1] for part in row["segments"]) == row["text"]
    assert row["segments"][0][0].endswith("chunk.audio:100:0#0.000:29.500")
    assert all(0 < meta["end"] - meta["start"] <= 30 for meta in row["segment_metadata"])
    assert list(tmp_path.iterdir()) == []


def test_sliding_pairs_include_last_segment_and_exact_own_labels():
    row = alignment.filter_word_durations(alignment.segment_row(original()))
    pairs = list(alignment.paired_rows(row))
    assert len(pairs) == 2
    assert [pair["segment_indices"] for pair in pairs] == [[0, 1], [1, 2]]
    for pair in pairs:
        assert len(pair["audio_path"]) == len(pair["segment_reference_texts"]) == 2
        assert pair["text"] == " ".join(pair["segment_reference_texts"])
        assert pair["parent_id"] == row["id"]


@pytest.mark.parametrize("duration,reason", [(0.0, "too_short"), (0.03, "too_short"), (3.001, "too_long")])
def test_one_bad_word_excludes_entire_original_and_all_pairs(duration, reason):
    row = alignment.segment_row(original())
    row["word_timestamps"][61]["end"] = 61 + duration
    filtered = alignment.filter_word_durations(row)
    assert not filtered["word_duration_filter"]["passed"]
    assert filtered["word_duration_filter"]["outliers"][0]["word_index"] == 61
    assert filtered["word_duration_filter"]["outliers"][0]["reason"] == reason
    assert len(filtered["segments"]) == 3  # Raw segments and timestamps remain available.
    assert list(alignment.paired_rows(filtered)) == []


@pytest.mark.parametrize("duration", [0.04, 3.0])
def test_duration_limits_are_inclusive(duration):
    row = original()
    row["word_timestamps"][1]["end"] = 1 + duration
    assert alignment.filter_word_durations(row)["word_duration_filter"]["passed"]


@pytest.mark.parametrize("problem", ["overlap", "mismatch", "outside", "nonfinite", "overlong"])
def test_bad_alignment_is_rejected(problem):
    row = original()
    if problem == "overlap":
        row["word_timestamps"][1]["start"] = 0.1
    elif problem == "mismatch":
        row["word_timestamps"][1]["text"] = "other"
    elif problem == "outside":
        row["word_timestamps"][-1]["end"] = 66.0
    elif problem == "nonfinite":
        row["word_timestamps"][0]["start"] = float("nan")
    else:
        row["word_timestamps"][0]["end"] = 31.0
    with pytest.raises(ValueError):
        alignment.segment_row(row)


def test_long_caption_windows_cover_exact_reference_despite_normalization_change():
    captions = [
        {
            "start_time": i * 10.0,
            "end_time": (i + 1) * 10.0,
            "transcription": f"These are words{i} with item{i} numbers.",
        }
        for i in range(40)
    ]
    text = " ".join(part["transcription"].lower().replace(".", "") for part in captions)
    text = text.replace("item2", "replacement")
    windows = alignment.alignment_windows(text, 400.0, captions)
    assert all(window["end"] - window["start"] <= 180 for window in windows)
    assert windows[0]["word_start"] == 0 and windows[-1]["word_end"] == len(text.split())
    assert all(a["word_end"] == b["word_start"] for a, b in zip(windows, windows[1:], strict=False))


def test_unreliable_caption_mapping_is_rejected_instead_of_dividing_text_evenly():
    with pytest.raises(ValueError, match="agreement"):
        alignment.alignment_windows(
            "different known human words",
            500.0,
            [{"start_time": 0, "end_time": 500, "transcription": "unrelated caption"}],
        )


def test_model_window_context_is_checked_but_every_human_word_is_emitted_once():
    captions = [
        {"start_time": i * 10, "end_time": (i + 1) * 10, "transcription": f"first{i} middle{i} last{i}"}
        for i in range(8)
    ]
    reference = " ".join(caption["transcription"] for caption in captions).split()
    windows = alignment.alignment_windows(" ".join(reference), 80, captions, maximum=42)
    timestamps = []
    for window in windows:
        assert window["end"] - window["start"] <= 42
        if window["word_start"]:
            assert window["align_word_start"] < window["word_start"]
        if window["word_end"] < len(reference):
            assert window["align_word_end"] > window["word_end"]
        result = [
            SimpleNamespace(text=reference[i], start_time=i - window["start"], end_time=i + 0.5 - window["start"])
            for i in range(window["align_word_start"], window["align_word_end"])
        ]
        timestamps.extend(alignment.window_timestamps(reference, window, result, window["start"]))
    assert [item["text"] for item in timestamps] == reference
    assert [(item["start"], item["end"]) for item in timestamps] == [(i, i + 0.5) for i in range(len(reference))]
    result[0].text = "wrong context word"
    with pytest.raises(ValueError, match="different alignment word"):
        alignment.window_timestamps(reference, windows[-1], result, windows[-1]["start"])


def test_caption_context_that_cannot_fit_is_retained_as_mapping_failure():
    captions = [
        {"start_time": i * 25, "end_time": (i + 1) * 25, "transcription": f"first{i} middle{i} last{i}"}
        for i in range(4)
    ]
    with pytest.raises(ValueError, match="with context"):
        alignment.alignment_windows(" ".join(c["transcription"] for c in captions), 100, captions, maximum=42)


def test_stitching_selects_real_predictions_without_clamping_overlap():
    windows = [
        {"align_word_start": 0, "align_word_end": 5, "word_start": 0},
        {"align_word_start": 2, "align_word_end": 7, "word_start": 4},
    ]
    left = [{"text": f"word{i}", "start": i, "end": i + 0.6} for i in range(5)]
    right = [{"text": f"word{i}", "start": i - 0.5, "end": i + 0.1} for i in range(2, 7)]
    # The planned seam overlaps. The preceding shared seam is ordered.
    right[1]["start"] = 3.0
    merged, seams = alignment.stitch_window_timestamps(windows, [left, right])
    assert seams[0]["word_boundary"] == 3
    assert merged == left[:3] + right[1:]
    assert [x["text"] for x in merged] == [f"word{i}" for i in range(7)]
    assert right[2]["start"] == 3.5 and left[3]["end"] == 3.6
    for item in right:
        item["start"] = 0.0
    with pytest.raises(ValueError, match="No ordered model seam"):
        alignment.stitch_window_timestamps(windows, [left, right])


def test_owned_cache_releases_only_completed_chunks_and_preserves_other_data(tmp_path):
    root, output = tmp_path / "cache", tmp_path / "output"
    output.mkdir()
    cache = alignment.OwnedChunkCache(root, output, "input-hash")
    cache.visit("az://orngwus2cresco/data/task/first.audio:2:0")
    first = root / "task/first.audio"
    first.parent.mkdir()
    first.write_bytes(b"whole chunk")
    other = root / "task/keep.txt"
    other.write_bytes(b"preserve")
    cache.visit("az://orngwus2cresco/data/task/first.audio:2:1")
    assert first.exists()
    cache.visit("az://orngwus2cresco/data/task/second.audio:1:0")
    assert not first.exists() and other.read_bytes() == b"preserve"
    ledger = list(alignment.rows(output / "chunk_cache_releases.jsonl"))
    assert ledger[0]["released_bytes"] == len(b"whole chunk")
    with pytest.raises(ValueError, match="revisits"):
        cache.visit("az://orngwus2cresco/data/task/first.audio:2:0")
    cache.lock.close()
    with pytest.raises(ValueError, match="different alignment"):
        alignment.OwnedChunkCache(root, output, "other-input")


def test_owned_cache_refuses_unowned_data_and_escaped_source(tmp_path):
    unowned = tmp_path / "unowned"
    unowned.mkdir()
    (unowned / "keep.audio").write_bytes(b"keep")
    with pytest.raises(ValueError, match="without alignment ownership"):
        alignment.OwnedChunkCache(unowned, tmp_path, "input")
    cache = alignment.OwnedChunkCache(tmp_path / "owned", tmp_path, "input")
    with pytest.raises(ValueError, match="escapes"):
        cache.visit("az://orngwus2cresco/data/../../keep.audio:1:0")
    assert (unowned / "keep.audio").read_bytes() == b"keep"


def test_failed_original_never_produces_pairs():
    row = alignment.filter_word_durations({**original(), "alignment_error": "model error"})
    assert list(alignment.paired_rows(row)) == []


def test_versions_keep_raw_input_and_exclude_all_segments_of_bad_original(tmp_path):
    good = alignment.segment_row(original())
    bad = alignment.segment_row({**original(), "id": "bad"})
    bad["word_timestamps"][63]["end"] = 63.02
    source = tmp_path / "raw.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in [good, bad]))
    original_hash = alignment.fingerprint(source)
    output = tmp_path / "versions"
    alignment.pairs(
        SimpleNamespace(input=source, output_dir=output, minimum_word_seconds=0.04, maximum_word_seconds=3.0)
    )
    assert alignment.fingerprint(source) == original_hash
    originals = list(alignment.rows(output / "filtered_originals.jsonl.gz"))
    segments = list(alignment.rows(output / "segments.jsonl.gz"))
    pairs = list(alignment.rows(output / "pairs.jsonl.gz"))
    assert len(originals) == len(segments) == 1 and len(pairs) == 2
    assert originals[0]["audio_chunk"] == good["audio_chunk"] and "segments" not in originals[0]
    assert segments[0]["segments"] == good["segments"]
    exclusions = list(alignment.rows(output / "pair_exclusions.jsonl"))
    assert exclusions[0]["id"] == "bad" and exclusions[0]["reason"] == "unexpected_word_duration"
    counts = json.loads((output / "COMPLETE.json").read_text())
    assert counts["parents"] == 2 and counts["word_duration_exclusions"] == 1

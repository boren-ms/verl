import pytest

from scripts import prepare_openlong_qwen_alignment as planner


def make_rows(count):
    return [{"id": f"s{i}", "parent_audio_path": "parent", "seg_index": i,
             "n_segments": count, "seg_start": i * 30.0, "seg_end": i * 30.0 + 29.0,
             "audio_path": f"audio.zip!10:100#{i * 30}:{i * 30 + 29}", "text": f"word{i}"}
            for i in range(count)]


def test_core_coverage_context_and_word_offsets(monkeypatch):
    monkeypatch.setitem(planner.EXPECTED, "test", (7, 1))
    windows = planner.build_windows("test", make_rows(7), core_segments=3, context_segments=1)
    assert [w["core_segment_indexes"] for w in windows] == [[0, 1, 2], [3, 4, 5], [6]]
    assert windows[1]["context_segment_indexes"] == [2, 3, 4, 5, 6]
    assert [segment["id"] for segment in windows[1]["segments"]] == ["s2", "s3", "s4", "s5", "s6"]
    assert windows[1]["seed_text"].split()[windows[1]["core_word_start"]:windows[1]["core_word_end"]] == [
        "word3", "word4", "word5"
    ]
    assert all(w["audio_end"] - w["audio_start"] <= 150 for w in windows)
    assert [sid for w in windows for sid in w["core_segment_ids"]] == [f"s{i}" for i in range(7)]


def test_long_gap_shrinks_windows_without_losing_segments(monkeypatch):
    monkeypatch.setitem(planner.EXPECTED, "test", (3, 1))
    rows = make_rows(3)
    rows[2]["seg_start"], rows[2]["seg_end"] = 400.0, 429.0
    windows = planner.build_windows("test", rows, max_seconds=150)
    assert [i for w in windows for i in w["core_segment_indexes"]] == [0, 1, 2]
    assert all(w["audio_end"] - w["audio_start"] <= 150 for w in windows)


def test_rejects_missing_parent_segment(monkeypatch):
    monkeypatch.setitem(planner.EXPECTED, "test", (3, 1))
    rows = make_rows(3)
    rows[1]["seg_index"] = 2
    with pytest.raises(ValueError, match="Incomplete segment group"):
        planner.build_windows("test", rows)

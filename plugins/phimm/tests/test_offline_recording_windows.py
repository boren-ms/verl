"""Bounded recording windows retain each ordered short clip and its reference."""

import copy
import json

import pytest
from recipe.phimm.tools.offline_recording_windows import recording_windows


def example():
    rows = [{"audio_path": f"clip{i}.wav", "text": f"word{i}", "keywords": [str(i)]} for i in range(3)]
    parent = {
        "id": "recording",
        "audio_path": [row["audio_path"] for row in rows],
        "text": "word0 word1 word2",
        "source_audio_lengths": [301.0, 299.0, 1.0],
        "source_indices": [3, 0, 9],
        "source_ids": ["s0", "s1", "s2"],
        "source_metadata_json": [json.dumps(row) for row in rows],
        "merge_parent_audio_path": "original.wav",
        "source_order_by": "timestamp",
        "source_start_seconds": [0, 302, 605],
    }
    return parent, rows


def test_exact_cap_order_reference_and_provenance_without_mutating_inputs():
    parent, rows = example()
    before = copy.deepcopy((parent, rows))
    output = recording_windows(parent, rows)
    assert (parent, rows) == before
    assert [row["audio_length"] for row in output] == [600, 1]
    assert [row["audio_path"] for row in output] == [["clip0.wav", "clip1.wav"], ["clip2.wav"]]
    assert [row["text"] for row in output] == ["word0 word1", "word2"]
    assert [i for row in output for i in row["source_indices"]] == [3, 0, 9]
    assert [s for row in output for s in row["source_metadata_json"]] == parent["source_metadata_json"]
    assert [row["source_start_seconds"] for row in output] == [[0, 302], [605]]
    assert output[0]["keywords"] == ["0", "1"]
    assert len({row["id"] for row in output}) == 2
    assert all(row["merge_parent_audio_path"] == "original.wav" for row in output)


@pytest.mark.parametrize("duration", [601, 0, float("nan")])
def test_invalid_or_overlong_clip_requires_attention_instead_of_dropping_audio(duration):
    parent, rows = example()
    parent["source_audio_lengths"][0] = duration
    with pytest.raises(ValueError, match="cannot fit"):
        recording_windows(parent, rows)


def test_reordered_or_changed_reference_is_rejected():
    parent, rows = example()
    with pytest.raises(ValueError, match="order and reference"):
        recording_windows(parent, list(reversed(rows)))

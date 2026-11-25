"""Regression cases for zero-duration filtering and exact reference slices."""

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "src/recipe/phimm/tools/align_mix4_reference.py"
spec = importlib.util.spec_from_file_location("align_mix4_reference", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
remove = module.remove_zero_duration_words


def word(text, start, end):
    return {"text": text, "start": start, "end": end}


def test_leading_middle_and_trailing_words_are_removed_without_reappearance():
    text = "First, keep! middle? Straße. last..."
    timestamps = [
        word("First", 0, 0),
        word("keep", 0, 0.5),
        word("middle", 0.5, 0.5),
        word("Straße", 0.6, 1),
        word("last", 1, 1),
    ]
    result = remove(text, timestamps, 1)
    assert result["reference_text"] == "keep! Straße. "
    assert [w["text"] for w in result["word_timestamps"]] == ["keep", "Straße"]
    assert result["unaltered_word_timestamps"] == timestamps
    assert result["zero_duration_word_count"] == 3
    for removed in result["removed_zero_duration_words"]:
        lo, hi = removed["character_span"]
        assert text[lo:hi] == removed["reference_text"]


def test_positive_duration_including_tiny_spans_is_preserved_exactly():
    text = "Hello, world!"
    result = remove(text, [word("Hello", 0, 1e-12), word("world", 0.1, 1)], 1)
    assert result["reference_text"] == text
    assert result["removed_zero_duration_words"] == []


def test_all_zero_words_have_empty_reference_and_explicit_status():
    result = remove("Missing words.", [word("Missing", 1, 1), word("words", 1, 1)], 1)
    assert result["reference_text"] == ""
    assert result["word_timestamps"] == []
    assert result["alignment_status"] == "NO_POSITIVE_DURATION_REFERENCE"


def test_punctuation_only_reference_is_preserved():
    result = remove(".", [], 1)
    assert result["reference_text"] == "."
    assert result["alignment_status"] == "NO_LEXICAL_REFERENCE"


def test_standalone_qwen_punctuation_is_mapped_and_filtered():
    timestamps = [word("Partners", 0, 0.5), word("'", 0.5, 0.5), word("growth", 0.5, 1)]
    result = remove("Partners ' growth", timestamps, 1)
    assert result["reference_text"] == "Partners growth"
    assert result["removed_zero_duration_words"][0]["reference_text"] == "' "
    timestamps[1]["end"] = 0.6
    assert remove("Partners ' growth", timestamps, 1)["reference_text"] == "Partners ' growth"


@pytest.mark.parametrize("bad", [word("tail", 2, 2), word("tail", 0.8, 0.7), word("tail", float("nan"), 1)])
def test_invalid_words_are_rejected_before_filtering(bad):
    with pytest.raises(ValueError, match="Timestamp outside"):
        remove("tail", [bad], 1)


def test_transcript_mismatch_is_not_hidden_by_removal():
    with pytest.raises(ValueError, match="differ from source"):
        remove("different", [word("tail", 0, 0)], 1)


def test_nonmonotonic_alignment_is_rejected():
    with pytest.raises(ValueError, match="Nonmonotonic"):
        remove("one two", [word("one", 0.7, 0.9), word("two", 0.1, 0.2)], 1)


def parent(paths=None):
    return {
        "audio_path": "parent.wav",
        "text": "Hello, missing world!",
        "original_segment_audio_paths": paths or ["parent.wav#0%:50%", "parent.wav#50%:100%"],
    }


def test_full_parent_words_are_assigned_once_to_original_segments():
    result = module.assign_parent_alignment(
        parent(), [word("Hello", 0, 0.4), word("missing", 0.5, 0.5), word("world", 0.6, 1)], 1
    )
    assert result["segment_reference_texts"] == ["Hello, ", "world!"]
    assert result["word_segment_indices"] == [0, 1]
    assert result["alignment_scope"] == "full_parent_audio_then_original_segments"


def test_crossing_word_at_boundary_is_assigned_once_and_flagged():
    row = parent()
    row["text"] = "crossing"
    result = module.assign_parent_alignment(row, [word("crossing", 0.4, 0.6)], 1)
    assert result["segment_reference_texts"] == ["", "crossing"]
    assert result["reference_boundary_crossings"][0]["assigned_segment"] == 2


@pytest.mark.parametrize("path", ["parent.wav#0%:50%", ["parent.wav"], "one.wav;two.wav"])
def test_segment_audio_cannot_be_aligned_with_full_reference(path):
    row = parent()
    row["audio_path"] = path
    with pytest.raises(ValueError, match="full parent"):
        module.validate_parent_audio(row)


@pytest.mark.parametrize(
    "paths",
    [["other.wav#0%:100%"], ["parent.wav#0%:40%", "parent.wav#50%:100%"], ["parent.wav#0%:70%", "parent.wav#50%:100%"]],
)
def test_wrong_parent_gaps_and_overlaps_are_rejected(paths):
    with pytest.raises(ValueError):
        module.assign_parent_alignment(parent(paths), [], 1)

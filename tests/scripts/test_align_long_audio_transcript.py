from scripts.align_long_audio_transcript import assign_reference_words


def test_assign_reference_words_preserves_reference_at_segment_boundary():
    reference = "Hello, WORLD. missing boundary Next segment!"
    hypotheses = ["hello world", "next segment"]

    aligned = assign_reference_words(reference, hypotheses)

    assert aligned == ["Hello, WORLD. missing", "boundary Next segment!"]
    assert " ".join(aligned) == reference


def test_assign_reference_words_handles_empty_hypotheses():
    aligned = assign_reference_words("one two three", ["", ""])

    assert aligned == ["one two three", ""]
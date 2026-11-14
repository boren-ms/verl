"""Preserve ASR alignment and operation-specific accuracies after extraction."""

import pytest

from recipe.phimm.reward.asr_metrics import ErrorCounts, measure, normalize_text


@pytest.mark.parametrize(
    ("hyp", "ref", "unit", "expected"),
    [
        ("hello world", "hello world", "word", ErrorCounts(n_hit=2)),
        ("hello", "hello world", "word", ErrorCounts(n_del=1, n_hit=1)),
        ("hello brave world", "hello world", "word", ErrorCounts(n_ins=1, n_hit=2)),
        ("hello there", "hello world", "word", ErrorCounts(n_sub=1, n_hit=1)),
        ("one two three", "red blue", "word", ErrorCounts(n_sub=2, n_ins=1)),
        ("red blue", "one two three", "word", ErrorCounts(n_sub=2, n_del=1)),
        ("", "hello world", "word", ErrorCounts(n_del=2)),
        ("hello world", "", "word", ErrorCounts(n_ins=2)),
        ("", "", "word", ErrorCounts()),
        ("你 好", "你好", "char", ErrorCounts(n_hit=2)),
        ("你们", "你好", "char", ErrorCounts(n_sub=1, n_hit=1)),
    ],
)
def test_alignment_counts(hyp, ref, unit, expected):
    counts = measure(hyp, ref, unit=unit, text_norm="identity")
    assert counts == expected
    assert counts.n_ref == expected.n_sub + expected.n_del + expected.n_hit
    assert counts.n_err == expected.n_sub + expected.n_del + expected.n_ins


def test_operation_specific_accuracies_and_empty_reference_insertions():
    counts = ErrorCounts(n_sub=1, n_del=1, n_ins=2, n_hit=2)
    assert counts.accuracy() == 0
    assert counts.accuracy(sub=0, ins=0) == 0.75
    assert counts.accuracy(sub=0, **{"del": 0}) == 0.5
    assert ErrorCounts(n_ins=2).accuracy() == -1
    assert ErrorCounts().accuracy() == 1


def test_long_repeated_sequences_keep_alignment_without_autojunk():
    ref = " ".join(["a"] * 200 + ["b"] + ["a"] * 200)
    hyp = " ".join(["a"] * 200 + ["c"] + ["a"] * 200)
    assert measure(hyp, ref, text_norm="identity") == ErrorCounts(n_sub=1, n_hit=400)


def test_normalization_keeps_language_default_and_explicit_override():
    text = " Hello, World! "
    assert normalize_text(text, lang="english") == "hello world"
    assert normalize_text(text, name="identity", lang="english") == "Hello, World!"


def test_unsupported_alignment_unit_fails_explicitly():
    with pytest.raises(ValueError, match="Unsupported error measure unit"):
        measure("hello", "hello", unit="sentence")

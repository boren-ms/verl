"""Tests for digits-only validation scoring."""

import pytest

from recipe.phimm.reward.digits_measure import compute_score, eval_score, score_digits


def test_score_digits_returns_error_and_reference_counts():
    assert score_digits("020", "zero two nine") == {
        "wer": pytest.approx(1.0 / 3.0),
        "n_err": 1,
        "n_ref": 3,
        "cer": pytest.approx(1.0 / 3.0),
        "nc_err": 1,
        "nc_ref": 3,
    }


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("The code is AB020X.", "The code is AB029X.", (1, 3, 1, 3)),
        ("Call 020-394-1446 today.", "Call 020-394-1447 today.", (1, 10, 1, 10)),
        ("Order #12 ships in 3 days.", "Order #12 ships in 4 days.", (1, 3, 1, 3)),
        ("Use zero two zero for access.", "Use zero two nine for access.", (1, 3, 1, 3)),
        ("IDs 02 and 94 are active.", "IDs 02 and 95 are active.", (1, 4, 1, 4)),
    ],
)
def test_score_digits_extracts_digits_from_strings(reference, hypothesis, expected):
    n_err, n_ref, nc_err, nc_ref = expected
    result = score_digits(reference, hypothesis)
    assert result["n_err"] == n_err
    assert result["n_ref"] == n_ref
    assert result["nc_err"] == nc_err
    assert result["nc_ref"] == nc_ref


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("0203", "zero two zero"),
        ("0203", "02039"),
        ("0203", "0293"),
    ],
)
def test_score_digits_reports_partial_sequence_errors(reference, hypothesis):
    assert score_digits(reference, hypothesis) == {
        "wer": pytest.approx(1.0 / 4.0),
        "n_err": 1,
        "n_ref": 4,
        "cer": pytest.approx(1.0 / 4.0),
        "nc_err": 1,
        "nc_ref": 4,
    }


def test_eval_score_matches_spoken_digit_words():
    result = eval_score(
        "Audio Language: English.\n<ASR><lang=English><TXT>zero two zero three</TXT></ASR>",
        "0203",
    )

    assert result["score"] == 1.0
    assert result["wer"] == 0.0
    assert result["cer"] == 0.0
    assert result["n_err"] == 0
    assert result["n_ref"] == 4
    assert result["nc_err"] == 0
    assert result["nc_ref"] == 4
    assert set(result) == {"score", "wer", "n_err", "n_ref", "nc_err", "nc_ref", "cer"}


def test_eval_score_ignores_non_digit_words():
    result = eval_score(
        "<ASR><lang=English><TXT>please call 0 2, 9 now</TXT></ASR>",
        "zero two nine",
    )

    assert result["score"] == 1.0
    assert result["wer"] == 0.0
    assert result["n_ref"] == 3
    assert result["nc_ref"] == 3


def test_eval_score_reports_digit_substitution():
    result = eval_score("<TXT>zero two nine</TXT>", "020")

    assert result["cer"] == pytest.approx(1.0 / 3.0)
    assert result["wer"] == pytest.approx(1.0 / 3.0)
    assert result["score"] == pytest.approx(2.0 / 3.0)
    assert result["n_err"] == 1
    assert result["n_ref"] == 3
    assert result["nc_err"] == 1
    assert result["nc_ref"] == 3
    assert set(result) == {"score", "wer", "n_err", "n_ref", "nc_err", "nc_ref", "cer"}


def test_compute_score_reports_digit_accuracy():
    result = compute_score(
        "Audio Language: English.\n<ASR><lang=English><TXT>zero two nine</TXT></ASR>",
        "020",
        measures={"digit_char": {"beta": 1.0}},
        reduce="mean",
    )

    assert result["digit_char"] == pytest.approx(2.0 / 3.0)
    assert result["digit_word"] == pytest.approx(2.0 / 3.0)
    assert result["score"] == pytest.approx(2.0 / 3.0)
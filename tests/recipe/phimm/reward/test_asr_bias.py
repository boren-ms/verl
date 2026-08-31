import pytest

from recipe.phimm.reward.asr_bias import compute_score, eval_score


@pytest.mark.parametrize("score_fn", [compute_score, eval_score])
def test_score_extracts_hypothesis_from_asr_response(score_fn):
    result = score_fn(
        "<src=English><tgt=English>\nthe quick blue fox",
        "the quick brown fox",
        extra_info={"keywords": ["brown"]},
    )

    assert result["n_err"] == 1
    assert result["n_ref"] == 4
    assert result["nu_err"] == 0
    assert result["nu_ref"] == 3
    assert result["nb_err"] == 1
    assert result["nb_ref"] == 1


def test_eval_score_preserves_raw_hypothesis():
    expected = eval_score(
        "<src=English><tgt=English>\nthe quick brown fox",
        "the quick brown fox",
        extra_info={"keywords": ["brown"]},
    )

    assert eval_score(
        "the quick brown fox",
        "the quick brown fox",
        extra_info={"keywords": ["brown"]},
    ) == expected
import pytest
from omegaconf import OmegaConf

from recipe.phimm.reward import asr_measure, asr_measure_v0, asr_vote


def test_voting_reuses_shared_delta_helpers():
    assert asr_vote.compute_delta is asr_measure.compute_delta


@pytest.mark.parametrize("score_type", ["acc", "err"])
@pytest.mark.parametrize("text", ["cat", "cat dog", "cat dog extra extra extra extra"])
def test_score_type_converts_all_raw_measures(score_type, text):
    kwargs = {"version": 2609, "text_norm": "identity"}
    response = f"<TXT>{text}</TXT>"
    accuracies = asr_measure._parse_response(response, "cat dog", **kwargs)
    result = asr_vote.compute_score(
        response, "cat dog",
        measures={name: {"score": score_type} for name in accuracies}, **kwargs,
    )
    expected = {name: value - 1.0 if score_type == "err" else value for name, value in accuracies.items()}
    assert result == {"score": 0.0, **expected}


@pytest.mark.parametrize("score_type", ["acc", "err"])
@pytest.mark.parametrize("delta_mode", ["abs", "rel"])
@pytest.mark.parametrize("vote_mode", ["delta", "thr"])
@pytest.mark.parametrize("accuracy", [0.625, 0.75, 0.875])
def test_score_type_changes_relative_denominator(monkeypatch, score_type, delta_mode, vote_mode, accuracy):
    monkeypatch.setattr(asr_vote, "_parse_response", lambda *args, **kwargs: {"word": accuracy})
    baseline = -0.25 if score_type == "err" else 0.75
    value = accuracy - 1.0 if score_type == "err" else accuracy
    delta = accuracy - 0.75
    if delta_mode == "rel":
        delta /= abs(baseline)
    vote = delta if vote_mode == "delta" else 1.0 if delta > 0.25 else -1.0 if delta < -0.25 else 0.0
    result = asr_vote.compute_score(
        "", "",
        measures={
            "word": {"score": score_type, "delta": delta_mode, "vote": vote_mode, "tol": [-0.25, 0.25], "beta": 0.5}
        },
        extra_info={"greedy_reward": {"word": baseline}},
    )
    assert result == pytest.approx(
        {"score": 0.5 * vote, "word": value, "d_word": delta, "v_word": vote}
    )


@pytest.mark.parametrize("baseline_scorer", [asr_vote.compute_score, asr_measure_v0.compute_score])
def test_error_score_type_full_parser_round_trip(baseline_scorer):
    kwargs = {
        "version": 2609,
        "text_norm": "identity",
        "measures": {"word": {"score": "err", "delta": "rel", "vote": "delta", "beta": 0.5}},
    }
    baseline = baseline_scorer("<TXT>cat</TXT>", "cat dog cat dog", **kwargs)
    assert baseline["word"] == -0.75
    original = baseline.copy()
    result = asr_vote.compute_score(
        "<TXT>cat dog</TXT>", "cat dog cat dog", extra_info={"greedy_reward": baseline}, **kwargs
    )
    assert result == pytest.approx({"score": 1 / 6, "word": -0.5, "d_word": 1 / 3, "v_word": 1 / 3})
    assert baseline == original


def test_error_score_type_perfect_baseline_clips_relative_vote():
    kwargs = {
        "version": 2609, "text_norm": "identity",
        "measures": {"word": {"score": "err", "delta": "rel", "vote": "delta", "beta": 0.5}},
    }
    baseline = asr_vote.compute_score("<TXT>cat dog</TXT>", "cat dog", **kwargs)
    assert baseline == {"score": 0.0, "word": 0.0}
    result = asr_vote.compute_score(
        "<TXT>cat</TXT>", "cat dog", extra_info={"greedy_reward": baseline}, **kwargs
    )
    assert result == {"score": -5.0, "word": -0.5, "d_word": -5e7, "v_word": -10.0}


def test_accuracy_score_type_is_default():
    kwargs = {"version": 2609, "text_norm": "identity", "measures": {"word": {"delta": "rel", "vote": "delta"}}}
    baseline = asr_vote.compute_score("<TXT>cat</TXT>", "cat dog", **kwargs)
    kwargs["extra_info"] = {"greedy_reward": baseline}
    default_result = asr_vote.compute_score("<TXT>cat dog</TXT>", "cat dog", **kwargs)
    kwargs["measures"]["word"]["score"] = "acc"
    assert asr_vote.compute_score("<TXT>cat dog</TXT>", "cat dog", **kwargs) == default_result


@pytest.mark.parametrize("score_type", [None, "", "error", "negative_error", "neg_er", "neg_err", True, [], {}])
def test_invalid_score_type_fails_even_without_baseline(monkeypatch, score_type):
    monkeypatch.setattr(asr_vote, "_parse_response", lambda *args, **kwargs: {"word": 0.5})
    with pytest.raises(ValueError, match="Unsupported ASR vote score type"):
        asr_vote.compute_score("", "", measures={"word": {"score": score_type}})


def test_score_type_rejects_shared_configuration():
    with pytest.raises(ValueError, match="Configure ASR vote score per measure"):
        asr_vote.compute_score("", "", score="err")


def test_mixed_measure_score_types_baseline_and_reduction(monkeypatch):
    accuracies = {
        "baseline": {"word": 0.75, "word_del": 0.5, "lang": 0.25, "word_ins": 0.8},
        "sample": {"word": 0.875, "word_del": 0.25, "lang": 0.5, "word_ins": 0.9},
    }
    monkeypatch.setattr(asr_vote, "_parse_response", lambda solution, **kwargs: accuracies[solution])
    measures = OmegaConf.create({
        "word": {"score": "err", "delta": "rel", "vote": "delta", "beta": 2.0, "clip": [-0.25, 0.25]},
        "word_del": {"score": "acc", "vote": "thr", "tol": [-0.1, 0.1], "beta": 0.5},
        "lang": {"score": "err", "vote": "delta"},
        "word_ins": None,
    })
    baseline = asr_vote.compute_score("baseline", "", measures=measures)
    assert baseline == {"score": 0.0, "word": -0.25, "word_del": 0.5, "lang": -0.75, "word_ins": 0.8}
    original = baseline.copy()
    result = asr_vote.compute_score(
        "sample", "", measures=measures,
        reduce={"mode": "average", "total": 4}, extra_info={"greedy_reward": baseline},
    )
    assert result == pytest.approx({
        "score": 0.3125,
        "word": -0.125, "word_del": 0.25, "lang": -0.5, "word_ins": 0.9,
        "d_word": 0.5, "d_word_del": -0.25, "d_lang": 0.25, "d_word_ins": 0.1,
        "v_word": 0.25, "v_word_del": -1.0, "v_lang": 0.25, "v_word_ins": 1.0,
    })
    assert baseline == original


@pytest.mark.parametrize(
    ("delta", "expected"),
    [(None, 0.0), (-0.02, -1.0), (-0.01, 0.0), (0.0, 0.0), (0.01, 0.0), (0.02, 1.0)],
)
def test_compute_vote(delta, expected):
    assert asr_vote.compute_vote(delta) == expected


def test_compute_vote_custom_thresholds_and_missing_baseline():
    assert asr_vote.compute_vote(0.5, low=0.25, high=0.5) == 0.0
    assert asr_vote.compute_vote(0.0, low=0.25, high=0.5) == -1.0
    assert asr_vote.compute_vote(0.75, low=0.25, high=0.5) == 1.0
    assert asr_vote.compute_vote(None, low=0.25, high=0.5) == 0.0


@pytest.mark.parametrize(
    ("delta", "low", "high"),
    [
        (0.0, 0.5, -0.5),
        (0.0, float("nan"), 0.1),
        (None, -0.1, float("inf")),
        (float("nan"), -0.1, 0.1),
        (float("inf"), -0.1, 0.1),
    ],
)
def test_compute_vote_rejects_invalid_values(delta, low, high):
    with pytest.raises(ValueError, match="finite"):
        asr_vote.compute_vote(delta, low=low, high=high)


@pytest.mark.parametrize(
    ("baseline", "cfg", "expected"),
    [
        (0.5, None, (1.0, 0.0625, 1.0)),
        (0.5, {"delta": "abs", "beta": 2.0}, (2.0, 0.0625, 1.0)),
        (0.5, {"delta": "rel", "beta": 2.0}, (2.0, 0.125, 1.0)),
        (0.75, {"delta": "abs", "beta": 0.5}, (-0.5, -0.1875, -1.0)),
        (0.5, {"delta": "rel", "tol": [-0.01, 0.125]}, (0.0, 0.125, 0.0)),
        (0.0, {"delta": "rel"}, (1.0, 56250000.0, 1.0)),
        (0.5, {"delta": "rel", "beta": 0.0}, (0.0, 0.125, 1.0)),
    ],
)
def test_compute_measure_vote(baseline, cfg, expected):
    assert asr_vote.compute_measure_vote("word", 0.5625, baseline, cfg) == expected


def test_measure_vote_does_not_modify_configuration():
    cfg = {"delta": "rel", "beta": 2.0, "tol": [-0.25, 0.25]}
    original = cfg.copy()
    assert asr_vote.compute_measure_vote("word", 0.75, 0.5, cfg) == (2.0, 0.5, 1.0)
    assert cfg == original


@pytest.mark.parametrize("tol", [None, [], [0], [0, 1, 2], "01", 1, [False, 1], [0, "1"], {0: 0, 1: 1}])
def test_tol_rejects_invalid_bounds(tol):
    with pytest.raises(ValueError, match="tol"):
        asr_vote.compute_measure_vote("word", 1.0, 0.0, {"tol": tol})


@pytest.mark.parametrize(
    ("delta", "expected"),
    [(-0.03, -1.0), (-0.02, 0.0), (0.0, 0.0), (0.01, 0.0), (0.02, 1.0)],
)
def test_tol_supports_hydra_and_asymmetric_thresholds(delta, expected):
    cfg = OmegaConf.create({"tol": [-0.02, 0.01], "beta": 0.25})
    assert asr_vote.compute_measure_vote("word", delta, 0.0, cfg) == (0.25 * expected, delta, expected)


@pytest.mark.parametrize("delta_mode", ["abs", "absolute", "rel", "relative"])
@pytest.mark.parametrize("baseline", [-0.5, 0.0, 0.5])
@pytest.mark.parametrize("value", [-1.0, 0.0, 0.55, 2.0])
@pytest.mark.parametrize("beta", [0.0, 0.25, -2.0])
def test_direct_delta_vote(delta_mode, baseline, value, beta):
    cfg = {"delta": delta_mode, "vote": "delta", "beta": beta, "tol": [1.0, -1.0]}
    original = cfg.copy()
    expected = asr_measure.compute_delta(value, baseline, mode=delta_mode)
    expected_vote = min(max(expected, -10), 10)
    expected_score = beta * expected_vote
    assert asr_vote.compute_measure_vote("word", value, baseline, cfg) == (expected_score, expected, expected_vote)
    assert cfg == original


@pytest.mark.parametrize("vote_mode", ["unknown", None, True])
def test_invalid_vote_mode(vote_mode):
    with pytest.raises(ValueError, match="Unsupported ASR vote mode"):
        asr_vote.compute_measure_vote("word", 1.0, 0.5, {"vote": vote_mode})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_direct_delta_vote_rejects_nonfinite_delta(value):
    with pytest.raises(ValueError, match="finite delta"):
        asr_vote.compute_measure_vote("word", value, 0.5, {"vote": "delta"})


@pytest.mark.parametrize("vote_mode", ["thr", "threshold"])
def test_mixed_delta_and_threshold_votes(monkeypatch, vote_mode):
    monkeypatch.setattr(asr_vote, "_parse_response", lambda *args, **kwargs: {"word": 0.55, "lang": 0.0})
    result = asr_vote.compute_score(
        "",
        "",
        measures={
            "word": {"delta": "rel", "vote": "delta", "beta": 2.0},
            "lang": {"vote": vote_mode, "beta": 0.5},
        },
        reduce={"mode": "average", "total": 2.5},
        extra_info={"greedy_reward": {"word": 0.5, "lang": 1.0}},
    )
    assert result == pytest.approx(
        {"word": 0.55, "lang": 0.0, "d_word": 0.1, "d_lang": -1.0,
         "v_word": 0.1, "v_lang": -1.0, "score": -0.12}
    )


def test_direct_delta_full_parser_baseline_round_trip():
    kwargs = {
        "version": 2609,
        "text_norm": "identity",
        "measures": {"word": {"vote": "delta", "delta": "rel", "beta": 0.5}},
    }
    baseline = asr_vote.compute_score("<TXT>cat</TXT>", "cat dog", **kwargs)
    assert baseline == {"score": 0.0, "word": 0.5}
    result = asr_vote.compute_score(
        "<TXT>cat dog</TXT>", "cat dog", extra_info={"greedy_reward": baseline}, **kwargs
    )
    assert result == {"score": 0.5, "word": 1.0, "d_word": 1.0, "v_word": 1.0}


@pytest.mark.parametrize("value", [-100.0, -10.0, -2.0, 0.0, 2.0, 10.0, 100.0])
@pytest.mark.parametrize("beta", [0.25, -0.25, 2.0])
def test_score_clipping_defaults_before_beta(value, beta):
    cfg = {"vote": "delta", "beta": beta}
    expected_vote = min(max(value, -10), 10)
    expected = beta * expected_vote
    assert asr_vote.compute_measure_vote("word", value, 0.0, cfg) == (expected, value, expected_vote)
    assert "clip" not in cfg


def test_score_clipping_default_relative_zero_baseline(monkeypatch):
    monkeypatch.setattr(asr_vote, "_parse_response", lambda *args, **kwargs: {"word": 0.5})
    result = asr_vote.compute_score(
        "", "",
        measures={"word": {"delta": "rel", "vote": "delta", "beta": 0.25}},
        extra_info={"greedy_reward": {"word": 0.0}},
    )
    assert result == {"score": 2.5, "word": 0.5, "d_word": 5e7, "v_word": 10.0}


@pytest.mark.parametrize("vote_mode", ["delta", "thr", "threshold"])
@pytest.mark.parametrize(
    ("value", "expected"), [(0.0, -0.5), (0.5, 0.0), (1.0, 1.0)]
)
def test_score_clipping_before_beta_preserves_raw_delta(vote_mode, value, expected):
    cfg = {"vote": vote_mode, "beta": 2.0, "clip": [-0.25, 0.5]}
    score, delta, vote = asr_vote.compute_measure_vote("word", value, 0.5, cfg)
    assert score == expected
    assert delta == value - 0.5
    assert vote == expected / 2.0


@pytest.mark.parametrize(
    ("bounds", "value", "expected"),
    [
        ({"clip": [-0.25, 100]}, -2.0, -0.25),
        ({"clip": [-0.25, 100]}, 2.0, 2.0),
        ({"clip": [-1000, 0.25]}, 2.0, 0.25),
        ({"clip": [-1000, 0.25]}, -2.0, -2.0),
        ({"clip": [0.0, 0.0]}, 2.0, 0.0),
        ({"clip": [-1000, 100]}, -2000.0, -1000.0),
        ({"clip": [-1000, 100]}, 200.0, 100.0),
    ],
)
def test_score_clipping_custom_and_equal_bounds(bounds, value, expected):
    cfg = {"vote": "delta", **bounds}
    original = cfg.copy()
    assert asr_vote.compute_measure_vote("word", value, 0.0, cfg) == (expected, value, expected)
    assert cfg == original


@pytest.mark.parametrize("beta", [0.0, 0.25, -0.25, 2.0])
def test_score_clipping_happens_before_weighting(beta):
    assert asr_vote.compute_measure_vote(
        "word", 2.0, 0.0, {"vote": "delta", "beta": beta, "clip": [-1.0, 1.0]}
    ) == (beta, 2.0, 1.0)


@pytest.mark.parametrize("value", [-2000.0, -1000.0, 100.0, 200.0])
def test_score_clipping_requested_bounds_before_beta(value):
    expected_vote = min(max(value, -1000.0), 100.0)
    assert asr_vote.compute_measure_vote(
        "word", value, 0.0, {"vote": "delta", "beta": 0.25, "clip": [-1000, 100]}
    ) == (0.25 * expected_vote, value, expected_vote)


@pytest.mark.parametrize(
    "bounds",
    [
        {"clip": [1.0, -1.0]},
        {"clip": [float("nan"), 1]},
        {"clip": [0, float("inf")]},
        {"clip": None},
        {"clip": [0, "1"]},
        {"clip": [True, 1]},
        {"clip": []},
        {"clip": [0]},
        {"clip": [0, 1, 2]},
        {"clip": "01"},
        {"clip": 1},
    ],
)
def test_score_clipping_rejects_invalid_bounds(bounds):
    with pytest.raises(ValueError, match="clip"):
        asr_vote.compute_measure_vote("word", 1.0, 0.0, {"vote": "delta", **bounds})


def test_score_clipping_does_not_hide_nonfinite_weighted_scores():
    with pytest.raises(ValueError, match="nonfinite weighted score"):
        asr_vote.compute_measure_vote(
            "word", 1e308, 0.0, {"vote": "delta", "beta": 2.0, "clip": [-1e308, 1e308]}
        )


def test_score_clipping_prevents_overflow_before_weighting():
    assert asr_vote.compute_measure_vote(
        "word", 1e308, 0.0, {"vote": "delta", "beta": 2.0, "clip": [-1, 1]}
    ) == (2.0, 1e308, 1.0)


def test_score_clipping_precedes_reduction_and_returns_clipped_diagnostics(monkeypatch):
    monkeypatch.setattr(asr_vote, "_parse_response", lambda *args, **kwargs: {"word": 0.5, "lang": 0.0})
    result = asr_vote.compute_score(
        "", "",
        measures={
            "word": {"delta": "rel", "vote": "delta", "beta": 0.5, "clip": [-1, 1]},
            "lang": {"vote": "thr", "beta": 0.25, "clip": [-0.1, 1]},
        },
        reduce={"mode": "average", "total": 2},
        extra_info={"greedy_reward": {"word": 0.0, "lang": 1.0}},
    )
    assert result == pytest.approx({
        "score": 0.2375, "word": 0.5, "lang": 0.0,
        "d_word": 5e7, "v_word": 1.0, "d_lang": -1.0, "v_lang": -0.1,
    })


def test_score_clipping_rejects_shared_configuration():
    with pytest.raises(ValueError, match="per measure"):
        asr_vote.compute_score("", "", clip=[-1000, 100])


def test_score_clipping_supports_hydra_config():
    cfg = OmegaConf.create({"vote": "delta", "beta": 0.25, "clip": [-1000, 100]})
    assert asr_vote.compute_measure_vote("word", 200.0, 0.0, cfg) == (25.0, 200.0, 100.0)


def test_scorer_calls_vote_helper_for_each_measure(monkeypatch):
    monkeypatch.setattr(asr_vote, "_parse_response", lambda *args, **kwargs: {"word": 0.75, "fmt": 0.25})
    calls = []
    original = asr_vote.compute_vote

    def track(delta, low, high):
        calls.append((delta, low, high))
        return original(delta, low=low, high=high)

    monkeypatch.setattr(asr_vote, "compute_vote", track)
    result = asr_vote.compute_score(
        "",
        "",
        measures={
            "word": {"delta": "relative", "beta": 2.0},
            "fmt": {"delta": "relative", "tol": [-0.25, 0.25]},
        },
        extra_info={"greedy_reward": {"word": 0.5, "fmt": 0.5}},
    )
    assert calls == [(0.5, -0.01, 0.01), (-0.5, -0.25, 0.25)]
    assert result == {
        "score": 0.5,
        "word": 0.75,
        "fmt": 0.25,
        "d_word": 0.5,
        "d_fmt": -0.5,
        "v_word": 1.0,
        "v_fmt": -1.0,
    }


@pytest.mark.parametrize(
    ("delta", "expected"),
    [(-0.02, -1.0), (-0.01, 0.0), (-0.005, 0.0), (0.0, 0.0), (0.005, 0.0), (0.01, 0.0), (0.02, 1.0)],
)
def test_default_votes_and_inclusive_neutral_interval(monkeypatch, delta, expected):
    monkeypatch.setattr(asr_vote, "_parse_response", lambda *args, **kwargs: {"word": delta})
    result = asr_vote.compute_score(
        "",
        "",
        measures={"word": {}},
        extra_info={"greedy_reward": {"word": 0.0, "score": 0.0}},
    )

    assert result == {"score": expected, "word": delta, "d_word": delta, "v_word": expected}


@pytest.mark.parametrize("beta", [0.0, 0.25, 2.0, -1.0])
def test_returned_vote_is_independent_of_beta(monkeypatch, beta):
    monkeypatch.setattr(asr_vote, "_parse_response", lambda *args, **kwargs: {"word": 1.0})
    result = asr_vote.compute_score(
        "",
        "",
        measures={"word": {"beta": beta}},
        extra_info={"greedy_reward": {"word": 0.5}},
    )
    assert result == {"score": beta, "word": 1.0, "d_word": 0.5, "v_word": 1.0}


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [
        (None, 0.5),
        ("MEAN", 0.5),
        ("sum", 1.5),
        ({"mode": "average", "total": 6}, 0.25),
        ({"total": 6}, 0.5),
        ("multiply", 0.0),
        ("geometric", 0.0),
        ("harmonic", 0.0),
    ],
)
def test_beta_weighted_votes_are_reduced(monkeypatch, reduction, expected):
    monkeypatch.setattr(
        asr_vote,
        "_parse_response",
        lambda *args, **kwargs: {"word": 0.8, "fmt": 0.0, "lang": 1.0},
    )
    kwargs = {} if reduction is None else {"reduce": reduction}
    result = asr_vote.compute_score(
        "",
        "",
        measures={"word": {"beta": 2.0}, "fmt": {"beta": 0.5}, "lang": {}},
        extra_info={"greedy_reward": {"word": 0.5, "fmt": 1.0, "lang": 1.0}},
        **kwargs,
    )

    assert result == pytest.approx(
        {
            "score": expected,
            "word": 0.8,
            "fmt": 0.0,
            "lang": 1.0,
            "d_word": 0.3,
            "d_fmt": -1.0,
            "d_lang": 0.0,
            "v_word": 1.0,
            "v_fmt": -1.0,
            "v_lang": 0.0,
        }
    )


def test_measure_specific_thresholds_and_no_accuracy_transforms(monkeypatch):
    monkeypatch.setattr(asr_vote, "_parse_response", lambda *args, **kwargs: {"word": 0.25, "fmt": 0.25})
    result = asr_vote.compute_score(
        "",
        "",
        gamma=2.0,
        scale="word",
        measures={
            "word": {"tol": [-0.5, 0.5], "gamma": 2.0, "cut": 1.0},
            "fmt": {"tol": [-0.2, 0.2], "beta": 0.5},
        },
        extra_info={"greedy_reward": {"word": 0.0, "fmt": 0.0}},
    )

    assert result["score"] == 0.25


@pytest.mark.parametrize("extra_info", [None, {}, {"greedy_reward": None}, {"greedy_reward": {}}])
@pytest.mark.parametrize("delta", ["abs", "absolute", "rel", "relative"])
@pytest.mark.parametrize(
    "reduction",
    ["mean", "sum", "multiply", "geometric", "harmonic", {"mode": "average", "total": 2}],
)
@pytest.mark.parametrize("measures", [{}, {"word": {"beta": 2.0}, "word_del": {"beta": 0.5}}])
def test_missing_baseline_collects_raw_metrics_with_neutral_score(extra_info, delta, reduction, measures):
    result = asr_vote.compute_score(
        "<TXT>cat</TXT>",
        "cat dog",
        version=2609,
        text_norm="identity",
        measures={name: {**cfg, "delta": delta} for name, cfg in measures.items()},
        reduce=reduction,
        extra_info=extra_info,
    )

    assert result == {"score": 0.0, **{name: 0.5 for name in measures}}


@pytest.mark.parametrize("baseline_scorer", [asr_vote.compute_score, asr_measure.compute_score])
def test_real_parser_baseline_round_trip(baseline_scorer):
    kwargs = {"version": 2609, "text_norm": "identity", "measures": {"word": {}, "word_del": {}}}
    baseline = baseline_scorer("<TXT>cat</TXT>", "cat dog", **kwargs)
    original = baseline.copy()
    improved = asr_vote.compute_score(
        "<TXT>cat dog</TXT>",
        "cat dog",
        extra_info={"greedy_reward": baseline},
        **kwargs,
    )
    regressed = asr_vote.compute_score(
        "<TXT></TXT>",
        "cat dog",
        extra_info={"greedy_reward": baseline},
        **kwargs,
    )

    assert improved == {
        "score": 1.0,
        "word": 1.0,
        "word_del": 1.0,
        "d_word": 0.5,
        "d_word_del": 0.5,
        "v_word": 1.0,
        "v_word_del": 1.0,
    }
    assert regressed["score"] == -1.0
    assert regressed["d_word"] == -0.5
    assert baseline == original


@pytest.mark.parametrize("measures", [None, {}])
def test_no_measures_returns_zero(measures):
    assert asr_vote.compute_score("cat", "cat", version=2609, measures=measures) == {"score": 0.0}


def test_reduction_mapping_overrides_top_level_options(monkeypatch):
    monkeypatch.setattr(asr_vote, "_parse_response", lambda *args, **kwargs: {"word": 1.0})
    result = asr_vote.compute_score(
        "",
        "",
        measures={"word": {}},
        total=10,
        reduce={"mode": "average", "total": 2},
        extra_info={"greedy_reward": {"word": 0.0}},
    )
    assert result["score"] == 0.5


@pytest.mark.parametrize(
    ("accuracy", "baseline", "expected"),
    [
        (0.5625, 0.5, 1.0),
        (0.46875, 0.5, 0.0),
        (0.4, 0.5, -1.0),
        (0.5, 0.5, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 1.0),
        (-0.5, 0.0, -1.0),
        (5e-10, 0.0, 0.0),
        (0.0, 5e-10, 0.0),
        (-0.4, -0.5, 1.0),
        (-0.6, -0.5, -1.0),
    ],
)
@pytest.mark.parametrize("mode", ["rel", "relative"])
def test_relative_delta_votes(monkeypatch, accuracy, baseline, expected, mode):
    monkeypatch.setattr(asr_vote, "_parse_response", lambda *args, **kwargs: {"word": accuracy})
    result = asr_vote.compute_score(
        "",
        "",
        measures={"word": {"delta": mode, "tol": [-0.1, 0.1]}},
        extra_info={"greedy_reward": {"word": baseline, "score": 0.25}},
    )

    assert result == pytest.approx(
        {
            "score": expected,
            "word": accuracy,
            "d_word": (accuracy - baseline) / max(abs(baseline), 1e-8),
            "v_word": expected,
        }
    )


@pytest.mark.parametrize("accuracy", [0.375, 0.625])
def test_relative_delta_threshold_equality_is_neutral(monkeypatch, accuracy):
    monkeypatch.setattr(asr_vote, "_parse_response", lambda *args, **kwargs: {"word": accuracy})
    result = asr_vote.compute_score(
        "",
        "",
        measures={"word": {"delta": "relative", "tol": [-0.25, 0.25]}},
        extra_info={"greedy_reward": {"word": 0.5}},
    )
    assert result["score"] == 0.0
    assert result["v_word"] == 0.0


def test_relative_mode_changes_votes_and_preserves_beta_and_reduction(monkeypatch):
    monkeypatch.setattr(asr_vote, "_parse_response", lambda *args, **kwargs: {"word": 0.5625, "fmt": 0.4375})
    kwargs = {
        "measures": {
            "word": {"beta": 2.0, "tol": [-0.1, 0.1]},
            "fmt": {"beta": 0.5, "tol": [-0.1, 0.1]},
        },
        "extra_info": {"greedy_reward": {"word": 0.5, "fmt": 0.5}},
    }
    assert asr_vote.compute_score("", "", **kwargs)["score"] == 0.0
    for cfg in kwargs["measures"].values():
        cfg["delta"] = "abs"
    assert asr_vote.compute_score("", "", **kwargs)["score"] == 0.0
    for cfg in kwargs["measures"].values():
        cfg["delta"] = "relative"
    assert asr_vote.compute_score("", "", **kwargs)["score"] == 0.75
    assert asr_vote.compute_score("", "", reduce="sum", **kwargs)["score"] == 1.5


@pytest.mark.parametrize(
    ("word_mode", "fmt_mode", "expected", "word_delta", "fmt_delta"),
    [
        ("relative", "abs", 1.0, 0.125, 0.0625),
        ("abs", "relative", 0.25, 0.0625, 0.125),
        ("relative", "absolute", 1.0, 0.125, 0.0625),
        ("rel", "abs", 1.0, 0.125, 0.0625),
        ("abs", "rel", 0.25, 0.0625, 0.125),
    ],
)
def test_mixed_measure_delta_modes(monkeypatch, word_mode, fmt_mode, expected, word_delta, fmt_delta):
    monkeypatch.setattr(asr_vote, "_parse_response", lambda *args, **kwargs: {"word": 0.5625, "fmt": 0.5625})
    result = asr_vote.compute_score(
        "",
        "",
        measures={
            "word": {"delta": word_mode, "beta": 2.0, "tol": [-0.1, 0.1]},
            "fmt": {"delta": fmt_mode, "beta": 0.5, "tol": [-0.1, 0.1]},
        },
        extra_info={"greedy_reward": {"word": 0.5, "fmt": 0.5, "score": 0.0}},
    )
    assert result == {
        "score": expected,
        "word": 0.5625,
        "fmt": 0.5625,
        "d_word": word_delta,
        "d_fmt": fmt_delta,
        "v_word": 1.0 if word_delta > 0.1 else 0.0,
        "v_fmt": 1.0 if fmt_delta > 0.1 else 0.0,
    }


def test_measure_without_delta_defaults_to_abs(monkeypatch):
    monkeypatch.setattr(asr_vote, "_parse_response", lambda *args, **kwargs: {"word": 0.5625, "fmt": 0.5625})
    result = asr_vote.compute_score(
        "",
        "",
        measures={"word": {"delta": "relative"}, "fmt": None},
        extra_info={"greedy_reward": {"word": 0.5, "fmt": 0.5}},
    )
    assert result == {
        "score": 1.0,
        "word": 0.5625,
        "fmt": 0.5625,
        "d_word": 0.125,
        "d_fmt": 0.0625,
        "v_word": 1.0,
        "v_fmt": 1.0,
    }


def test_relative_mode_real_parser_baseline_round_trip():
    kwargs = {"version": 2609, "text_norm": "identity", "measures": {"word": {"delta": "rel"}}}
    baseline = asr_vote.compute_score("<TXT>cat</TXT>", "cat dog", **kwargs)
    assert baseline == {"score": 0.0, "word": 0.5}
    result = asr_vote.compute_score(
        "<TXT>cat dog</TXT>",
        "cat dog",
        extra_info={"greedy_reward": baseline},
        **kwargs,
    )
    assert result == {"score": 1.0, "word": 1.0, "d_word": 1.0, "v_word": 1.0}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"measures": {"unknown": {}}}, "Unsupported ASR vote measures"),
        ({"measures": {"word": {}}, "extra_info": {"greedy_reward": {"fmt": 1.0}}}, "Missing greedy baseline"),
        ({"measures": {"word": {"tol": [0.5, -0.5]}}}, "tol low <= high"),
        ({"measures": {"word": {"tol": [-0.01, float("nan")]}}}, "finite numeric tol"),
        ({"measures": {"word": {"tol": [float("-inf"), 0.01]}}}, "finite numeric tol"),
        ({"measures": {"word": {"low": -0.01}}}, "instead of low/high"),
        ({"measures": {"word": {"high": 0.01}}}, "instead of low/high"),
        ({"tol": [-0.01, 0.01]}, "Configure ASR vote tol per measure"),
        ({"measures": {"word": {"beta": float("inf")}}}, "finite beta"),
        ({"reduce": "unknown"}, "Unsupported ASR vote reduction"),
        ({"measures": {"word": {"delta": "unknown"}}}, "Unsupported ASR vote delta mode"),
        ({"measures": {"word": {"delta": None}}}, "Unsupported ASR vote delta mode"),
        ({"delta": "relative"}, "Configure ASR vote delta per measure"),
        ({"vote": "delta"}, "Configure ASR vote mode per measure"),
        ({"reduce": {"mode": "average", "total": 0}}, "positive, finite"),
        (
            {"measures": {"word": {}}, "extra_info": {"greedy_reward": {"word": float("nan")}}},
            "finite delta",
        ),
        (
            {
                "measures": {"word": {"delta": "relative"}},
                "extra_info": {"greedy_reward": {"word": float("inf")}},
            },
            "finite delta",
        ),
    ],
)
def test_invalid_configuration_fails_explicitly(monkeypatch, kwargs, message):
    monkeypatch.setattr(asr_vote, "_parse_response", lambda *args, **kwargs: {"word": 1.0})
    kwargs = {"extra_info": {"greedy_reward": {"word": 0.5}}, **kwargs}
    with pytest.raises(ValueError, match=message):
        asr_vote.compute_score("", "", **kwargs)


@pytest.mark.parametrize("extra_info", [None, {}, {"greedy_reward": None}, {"greedy_reward": {}}])
def test_missing_baseline_returns_before_voting_and_reduction(monkeypatch, extra_info):
    monkeypatch.setattr(asr_vote, "_parse_response", lambda *args, **kwargs: {"word": 0.5, "fmt": 1.0})

    def unexpected(*args, **kwargs):
        pytest.fail("Baseline scoring must return before voting or reduction.")

    for name in ("_parse_reduction", "compute_delta", "compute_vote", "compute_measure_vote", "reduce_scores"):
        monkeypatch.setattr(asr_vote, name, unexpected)

    result = asr_vote.compute_score(
        "",
        "",
        extra_info=extra_info,
        measures={"word": {"delta": "invalid", "beta": float("nan")}},
        reduce={"mode": "average", "total": 0},
    )
    assert result == {"score": 0.0, "word": 0.5}

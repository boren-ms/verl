import ast
from pathlib import Path

import pytest

from recipe.phimm.reward import asr_measure, asr_measure_v0


def test_variant_contains_only_error_rate_wrappers():
    variant = ast.parse(Path(asr_measure_v0.__file__).read_text())
    functions = {node.name for node in variant.body if isinstance(node, ast.FunctionDef)}

    assert functions == {"_parse_response", "compute_score", "lang_score"}
    assert asr_measure_v0._parse_accuracy is asr_measure._parse_response
    assert asr_measure_v0._lang_accuracy_score is asr_measure.lang_score
    for name in ("compute_deltas", "reduce_scores", "scale_score", "signed_pow"):
        assert getattr(asr_measure_v0, name) is getattr(asr_measure, name)


def test_variant_calls_original_parser(monkeypatch):
    calls = []
    original = asr_measure._parse_response

    def track(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(asr_measure_v0, "_parse_accuracy", track)

    kwargs = {"version": 2609, "text_norm": "identity"}
    assert asr_measure_v0.compute_score(
        "<TXT>cat</TXT>", ground_truth="cat dog", measures={"word": {}}, **kwargs,
    ) == {"score": -0.5, "word": 0.5}
    assert len(calls) == 1
    assert asr_measure_v0.lang_score(
        "<src=English><tgt=English>\n<TXT>cat</TXT>", language="English", **kwargs,
    ) == {"score": 0.0, "p_lang": 0.0}


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("Hello world.", "<src=English><tgt=English>\n<TXT>Hello world.</TXT>"),
        ("Hello world.", "<TXT>hello world</TXT>"),
        ("a", "<TXT>a b c d</TXT>"),
        ("hello world", "<TXT>hello [background noise] world</TXT>"),
        ("the brown fox", "<TXT>the blue fox</TXT>"),
        ("hello", "<TXT></TXT>"),
        ("", "<TXT></TXT>"),
        ("", "<TXT>hello</TXT>"),
    ],
)
def test_every_parsed_component_is_an_error_rate(reference, hypothesis):
    kwargs = {"version": 2609, "text_norm": "identity", "extra_info": {"keywords": ["brown"]}}
    accuracies = asr_measure._parse_response(hypothesis, ground_truth=reference, **kwargs)
    errors = asr_measure_v0._parse_response(hypothesis, ground_truth=reference, **kwargs)

    assert errors == pytest.approx({name: 1.0 - value for name, value in accuracies.items()})
    assert errors["fixed"] == 0.0


@pytest.mark.parametrize(
    ("reference", "hypothesis", "keywords", "errors", "references"),
    [
        ("the brown fox", "the brown fox", ["brown"], 0, 1),
        ("the brown fox", "the blue fox", ["brown"], 1, 1),
        ("the brown fox", "the fox", ["brown"], 1, 1),
        ("the brown fox", "the brown brown fox", ["brown"], 1, 1),
        ("new york state", "new state", ["new york"], 1, 2),
        ("the brown fox", "the blue fox", None, 0, 0),
        ("the brown fox", "the blue fox", [], 0, 0),
        ("", "", ["brown"], 0, 0),
        ("", "brown brown", ["brown"], 2, 0),
    ],
)
def test_parser_returns_keyword_error_rate_from_shared_helper(
    reference, hypothesis, keywords, errors, references,
):
    expected = errors / max(len(reference.split()), 1)
    keyword_result = asr_measure.compute_kw_score(
        reference, hypothesis, keywords, text_norm="identity",
    )
    assert keyword_result == pytest.approx({
        "word_kw": 1.0 - expected,
        "kw_acc": errors / max(references, 1),
        "kw_nerr": errors,
        "kw_nref": references,
    })

    kwargs = {"version": 2609, "text_norm": "identity", "extra_info": {"keywords": keywords}}
    parsed = asr_measure_v0._parse_response(f"<TXT>{hypothesis}</TXT>", ground_truth=reference, **kwargs)
    assert parsed["word_kw"] == pytest.approx(expected)
    result = asr_measure_v0.compute_score(
        f"<TXT>{hypothesis}</TXT>", ground_truth=reference, measures={"word_kw": {"high": 10.0}}, **kwargs,
    )
    assert result == pytest.approx({"word_kw": expected, "score": -expected})


def test_keyword_alignment_is_computed_only_once(monkeypatch):
    calls = []
    original = asr_measure.compute_kw_score

    def track(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(asr_measure, "compute_kw_score", track)
    result = asr_measure_v0.compute_score(
        "<TXT>the blue fox</TXT>",
        ground_truth="the brown fox",
        version=2609,
        text_norm="identity",
        extra_info={"keywords": ["brown"]},
        measures={"word_kw": {}},
    )

    assert len(calls) == 1
    assert result == pytest.approx({"word_kw": 1 / 3, "score": -1 / 3})


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [
        ("sum", -0.75),
        ("mean", -0.375),
        ({"mode": "average", "total": 4}, -0.1875),
        ("multiply", -0.125),
        ("geometric", -(0.125 ** 0.5)),
        ("harmonic", -(2 / 6)),
    ],
)
def test_score_is_negative_reduced_error(reduction, expected):
    result = asr_measure_v0.compute_score(
        "<TXT>cat</TXT>",
        ground_truth="cat dog",
        version=2609,
        text_norm="identity",
        reduce=reduction,
        measures={"word": {"beta": 1.0}, "word_del": {"beta": 0.5}},
    )

    assert result == pytest.approx({"score": expected, "word": 0.5, "word_del": 0.5})


@pytest.mark.parametrize("measures", [None, {}, {"word": {}}, {"fixed": {"beta": 1.0}}])
def test_perfect_response_has_zero_score(measures):
    result = asr_measure_v0.compute_score(
        "<TXT>cat dog</TXT>", ground_truth="cat dog", version=2609, measures=measures,
    )

    assert result == {"score": 0.0, **{name: 0.0 for name in measures or {}}}


@pytest.mark.parametrize(("high", "expected"), [(1.0, -1.0), (10.0, -3.0)])
def test_insertion_errors_are_unclipped_in_raw_metrics(high, expected):
    result = asr_measure_v0.compute_score(
        "<TXT>a b c d</TXT>",
        ground_truth="a",
        version=2609,
        text_norm="identity",
        measures={"word_ins": {"high": high}},
    )

    assert result == {"score": expected, "word_ins": 3.0}


@pytest.mark.parametrize(("scale", "factor"), [("constant", 1.0), ("word", 0.5), ("char", 1 / 6)])
def test_weighting_gamma_and_error_scaling(scale, factor):
    result = asr_measure_v0.compute_score(
        "<TXT>ct dog</TXT>",
        ground_truth="cat dog",
        version=2609,
        text_norm="identity",
        gamma=2.0,
        scale=scale,
        measures={"word": {"beta": 2.0, "gamma": 2.0}},
    )

    assert result["word"] == 0.5
    assert result["score"] == pytest.approx(-0.25 * factor)


@pytest.mark.parametrize("cut", [-1.0, 0.0, 0.5, 0.6, 1.0])
def test_cut_does_not_override_negative_error_score(cut):
    result = asr_measure_v0.compute_score(
        "<TXT>cat</TXT>",
        ground_truth="cat dog",
        version=2609,
        measures={"word": {"cut": cut}},
    )

    assert result == {"score": -0.5, "word": 0.5}


def test_greedy_deltas_use_error_metrics_and_negative_reward():
    kwargs = {"version": 2609, "measures": {"word": {}}}
    baseline = asr_measure_v0.compute_score("<TXT>cat dog</TXT>", ground_truth="cat dog", **kwargs)
    result = asr_measure_v0.compute_score(
        "<TXT>cat</TXT>", ground_truth="cat dog", extra_info={"greedy_reward": baseline}, **kwargs,
    )

    assert result == {"score": -0.5, "word": 0.5, "d_score": -0.5, "d_word": 0.5}
    assert baseline == {"score": 0.0, "word": 0.0}


@pytest.mark.parametrize("scale", [None, "unknown", "fmt"])
def test_invalid_scale_is_rejected(scale):
    with pytest.raises(ValueError, match="Unsupported score scale"):
        asr_measure_v0.compute_score("<TXT>cat</TXT>", ground_truth="cat", scale=scale)


def test_unknown_measure_is_rejected():
    with pytest.raises(ValueError, match="Unsupported ASR error measures: missing"):
        asr_measure_v0.compute_score("<TXT>cat</TXT>", ground_truth="cat", measures={"missing": {}})


@pytest.mark.parametrize(
    ("language", "error"),
    [("English", 0.0), ("French", 1.0)],
)
def test_language_only_reward(language, error):
    result = asr_measure_v0.lang_score(
        "<src=English><tgt=English>\n<TXT>cat</TXT>", language=language, version=2609,
    )

    assert result == {"score": -error, "p_lang": error}

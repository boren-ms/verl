import sys

import pytest

from recipe.phimm.reward import asr_measure, asr_measure_v0, asr_vote
from recipe.phimm.reward.asr_edge import measure
from recipe.phimm.utils.tn import default_tn_name, text_norm


def test_text_norm_supports_hf_english():
    assert text_norm("It's Wi Fi", name="hf_english") == "it is wifi"


def test_text_norm_supports_hf_english_lists():
    assert text_norm(["It's Wi Fi", "E Mail"], name="hf_english") == ["it is wifi", "email"]


@pytest.mark.parametrize("lang", ["Hindi", "hindi", " HI ", "hi_in", "hi-IN"])
def test_hindi_defaults_to_voi_normalization(lang):
    assert default_tn_name(lang) == "voi_hi"


@pytest.mark.parametrize(
    ("lang", "expected"),
    [(None, "openasr_en"), ("English", "openasr_en"), ("German", "openasr_ml"), ("Tamil", "openasr_ml")],
)
def test_other_language_defaults_are_unchanged(lang, expected):
    assert default_tn_name(lang) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "\u0939\u093f\u0928\u094d\u0926\u0940 \u092d\u093e\u0937\u093e\u0964",
        "\u0915\u093f\u0924\u093e\u092c \u0905\u091a\u094d\u091b\u0940 \u0939\u0948\u0964",
        "\u0915\u093f",
        "\u0915\u0940",
        "\u092e\u0948\u0902 \u0939\u0942\u0901",
        "[\u0936\u094b\u0930] \u0928\u092e\u0938\u094d\u0924\u0947 (\u0938\u0902\u0917\u0940\u0924)",
        "25% 1,000",
        "Hello, WORLD! <noise>",
        "  a\tb\nc  ",
        "\u0958 \u0915\u093c \u200d \u200c",
    ],
)
def test_voi_hi_matches_evaluation_preprocessing(text):
    from voi_oiwer import oiwer, standardize_hypothesis

    expected = standardize_hypothesis(text, "hi")
    assert text_norm(text, name="voi_hi").split() == expected
    assert text_norm([text], name="voi_hi") == [" ".join(expected)]
    result = oiwer(hypothesis=text, reference_lists=[[text]], input_language="hindi")
    if text.strip():
        assert result[7][0][0].split() == expected
    else:
        assert result[7] == []


def test_hindi_vowel_distinctions_are_preserved_and_legacy_override_remains():
    short = "\u0915\u093f"
    long = "\u0915\u0940"
    assert text_norm(short, name="voi_hi") == short
    assert text_norm(long, name="voi_hi") == long
    assert measure(long, short, tgt_lang="hindi").n_sub == 1
    assert measure(long, short, tgt_lang="hindi", text_norm="openasr_ml").n_err == 0


@pytest.mark.parametrize("scorer", [asr_measure.compute_score, asr_measure_v0.compute_score, asr_vote.compute_score])
def test_hindi_reward_scorers_use_matching_word_and_keyword_normalization(scorer):
    short = "\u0915\u093f"
    long = "\u0915\u0940"
    result = scorer(
        f"<src=Hindi><tgt=Hindi>\n<TXT>{long}</TXT>",
        short,
        version=2609,
        measures={"word": {}, "word_kw": {}},
        extra_info={"language": "Hindi", "keywords": [short]},
    )
    expected = -1.0 if scorer is asr_measure_v0.compute_score else 0.0
    assert result["word"] == expected
    assert result["word_kw"] == expected


def test_voi_hi_missing_dependency_fails_explicitly(monkeypatch):
    monkeypatch.setitem(sys.modules, "voi_oiwer", None)
    with pytest.raises(ImportError, match="Install voi-oiwer==0.1.4"):
        text_norm("hello", name="voi_hi")
    assert text_norm("Hello", name="openasr_ml") == "hello"
import pytest

from recipe.phimm.utils.open_asr_normalizer import EnglishTextNormalizer
from recipe.phimm.utils.open_asr_normalizer.data_utils import MultilingualNormalizer
from recipe.phimm.utils.open_asr_normalizer.eval_utils import measure_wer, normalize_for_wer
from recipe.phimm.utils.open_asr_normalizer.expansion_normalizer import (
    MultilingualNormalizer as ExpansionMultilingualNormalizer,
)
from recipe.phimm.utils.open_asr_normalizer.hf_english_normalizer import _HFEnglishTextNormalizer
from recipe.phimm.utils.open_asr_normalizer.normalizer import (
    BasicMultilingualTextNormalizer,
    BasicTextNormalizer,
    _BRACKET_PAIRS,
    _NON_SPEECH_TAGS,
    remove_non_speech_annotations,
)


@pytest.mark.parametrize("normalizer", [EnglishTextNormalizer(), _HFEnglishTextNormalizer()])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("they said (hello) today", "they said hello today"),
        ("they said [music] today", "they said today"),
        ("they study (music theory)", "they study music theory"),
        ("hello[laughter]world", "hello world"),
    ],
)
def test_english_normalizers_remove_only_non_speech_annotations(normalizer, text, expected):
    assert normalizer(text).strip() == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a ( Music  Playing ) b", "a   b"),
        ("a [<music>] b", "a [ ] b"),
        ("a (music [noise]) b", "a   b"),
        ("a （laughter） b ［noise］ c ＜music＞ d", "a   b   c   d"),
        ("a (music [notes]) b", "a (music [notes]) b"),
        ("a [music) b", "a [music) b"),
    ],
)
def test_annotation_helper_matches_delimiters_and_nested_tags(text, expected):
    assert remove_non_speech_annotations(text) == expected


@pytest.mark.parametrize(
    "normalizer",
    [
        BasicTextNormalizer(),
        BasicMultilingualTextNormalizer(remove_diacritics=False),
        MultilingualNormalizer(remove_diacritics=False),
        ExpansionMultilingualNormalizer(remove_diacritics=False),
    ],
)
def test_basic_and_multilingual_normalizers_preserve_ordinary_enclosures(normalizer):
    assert normalizer("bonjour (monde) [musique] fin").split() == ["bonjour", "monde", "fin"]


@pytest.mark.parametrize("lang", ["en", "fr", "pt"])
def test_reference_and_prediction_use_the_same_bracket_rules(lang):
    text = "they said (hello) [music] today"
    hyp_norm, ref_norm = normalize_for_wer(text, text, lang=lang)
    assert hyp_norm == ref_norm
    expected, _ = normalize_for_wer("they said hello today", "they said hello today", lang=lang)
    assert ref_norm == expected
    assert "music" not in ref_norm.split()
    assert measure_wer(text, text, lang=lang)["n_err"] == 0


@pytest.mark.parametrize("opening,closing", _BRACKET_PAIRS)
@pytest.mark.parametrize("tag", sorted(_NON_SPEECH_TAGS))
def test_complete_allowlist_with_all_delimiters(tag, opening, closing):
    body = "\t  ".join(tag.upper().split())
    assert remove_non_speech_annotations(f"left{opening} {body} {closing}right") == "left right"


@pytest.mark.parametrize("normalizer", [EnglishTextNormalizer(), _HFEnglishTextNormalizer()])
def test_enclosed_words_still_receive_number_and_spelling_normalization(normalizer):
    assert normalizer("［Twenty］ (colour) ＜music theory＞").strip() == "20 color music theory"


@pytest.mark.parametrize("lang", ["en", "de", "fr", "it", "es", "nl", "pt", "hy"])
def test_brackets_on_either_side_produce_matching_scores(lang):
    enclosed, plain = "bonjour (monde) [music]", "bonjour monde"
    assert measure_wer(enclosed, plain, lang=lang)["n_err"] == 0
    assert measure_wer(plain, enclosed, lang=lang)["n_err"] == 0


def test_supplementary_language_routing_is_preserved():
    normalizer = ExpansionMultilingualNormalizer(remove_diacritics=False)
    received = []
    normalizer._language_normalizers["hy"] = lambda text: received.append(text) or "routed"
    assert normalizer("Հայերեն (բառ)", lang="hy") == "routed"
    assert received == ["Հայերեն (բառ)"]


def test_supplementary_mandarin_brackets_preserve_spoken_words():
    from recipe.phimm.utils.open_asr_normalizer.chinese_normalizer import ChineseTextNormalizer

    normalizer = ChineseTextNormalizer()
    assert "".join(normalizer("他說［你好］＜music＞（世界）").split()) == normalizer("他说你好世界")


def test_hindi_lattice_scorer_keeps_its_input_and_routing(monkeypatch):
    from recipe.phimm.reward import asr_eval
    import voi_oiwer

    received = {}
    def fake_oiwer(**kwargs):
        received.update(kwargs)
        return (0.0, None, None, None, (0, 0, 0), 2, None, None)

    monkeypatch.setattr(voi_oiwer, "oiwer", fake_oiwer)
    lattice = [["नमस्ते"], ["दुनिया"]]
    result = asr_eval.openasr_hi_in_eval(
        "<src=Hindi><tgt=Hindi>\n<VERBATIM>\nनमस्ते (दुनिया) [music]",
        "unused", version=2609, extra_info={"language": "Hindi", "lattice": lattice},
    )
    assert result["n_err"] == 0
    assert received == {
        "hypothesis": "नमस्ते (दुनिया) [music]",
        "reference_lists": lattice,
        "input_language": "hindi",
    }

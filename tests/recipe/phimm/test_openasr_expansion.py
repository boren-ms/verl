import pytest

from recipe.phimm.reward.openasr_expansion import measure_expansion


def test_mandarin_counts_characters_without_compound_merging():
    result = measure_expansion("你好世", "你好世界", "zh")
    assert result["n_ref"] == 4
    assert result["n_del"] == 1
    assert result["cer"] == 0.25
    assert "wer" not in result


def test_mandarin_normalizes_traditional_numbers_and_han_spaces():
    result = measure_expansion("臺 灣有12個", "台湾有十二个", "zh")
    assert result["cer"] == 0


def test_vietnamese_numbers_and_tone_placement():
    result = measure_expansion("hoà bình 1.000 và 1,05%", "hòa bình một nghìn và một phẩy không năm phần trăm", "vi")
    assert result["wer"] == 0


def test_vietnamese_preserves_meaningful_diacritics():
    assert measure_expansion("ma", "má", "vi")["wer"] == 1


def test_french_compounds_and_corpus_counts():
    result = measure_expansion("bonjour porte monnaie", "bonjour portemonnaie", "fr")
    assert result["wer"] == 0
    assert result["n_ref"] == 2


def test_unconfigured_language_fails():
    with pytest.raises(ValueError, match="Unsupported"):
        measure_expansion("hello", "hello", "en")

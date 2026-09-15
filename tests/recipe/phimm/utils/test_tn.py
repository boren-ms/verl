from recipe.phimm.utils.tn import text_norm


def test_text_norm_supports_hf_english():
    assert text_norm("It's Wi Fi", name="hf_english") == "it is wifi"


def test_text_norm_supports_hf_english_lists():
    assert text_norm(["It's Wi Fi", "E Mail"], name="hf_english") == ["it is wifi", "email"]
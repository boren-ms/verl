import json

import pytest

from recipe.phimm.summarize_keyword_eval import summarize


def test_corpus_ratios_and_zero_keyword_denominator(tmp_path):
    rows = [
        dict(data_source="clean", id="a", step=0, n_err=0, n_ref=100, nb_err=0, nb_ref=1, nu_err=0, nu_ref=99),
        dict(data_source="clean", id="b", step=0, n_err=1, n_ref=1, nb_err=1, nb_ref=1, nu_err=0, nu_ref=0),
        dict(data_source="other", id="c", step=0, n_err=1, n_ref=2, nb_err=0, nb_ref=0, nu_err=1, nu_ref=2),
    ]
    path = tmp_path / "result.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    result = summarize([path])
    assert result["clean"]["WER"] == pytest.approx(1 / 101)
    assert result["clean"]["BWER"] == 0.5
    assert result["clean"]["UWER"] == result["clean"]["NWER"] == 0
    assert result["other"]["BWER"] is None
    assert result["other"]["keyword_word_accuracy"] is None
    with pytest.raises(ValueError, match="duplicate"):
        summarize([path, path])

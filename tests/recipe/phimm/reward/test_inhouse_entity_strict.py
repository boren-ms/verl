"""Official EER failures must not appear as perfect recognition."""

import pytest

from recipe.phimm.reward import asr_inhouse_measure as measure


@pytest.mark.parametrize(
    "record",
    [
        None,
        {},
        {"Metrics": []},
        {"Metrics": [{"EntityInfo": {}}]},
        {"Metrics": [{"EntityInfo": {"NumTransEnts": 0, "NumTransEntsMatched": 0}}]},
        {"Metrics": [{"EntityInfo": {"NumTransEnts": 2, "NumTransEntsMatched": 3}}]},
    ],
)
def test_strict_eer_rejects_failed_or_invalid_metrics(monkeypatch, record):
    monkeypatch.setattr(measure, "get_metrics", lambda *a, **k: record)
    with pytest.raises(RuntimeError):
        measure._compute_eer("hello <NE> World </NE>", "hello world", strict=True)


def test_official_counts_used(monkeypatch):
    monkeypatch.setattr(
        measure,
        "get_metrics",
        lambda *a, **k: {"Metrics": [{"EntityInfo": {"NumTransEnts": 3, "NumTransEntsMatched": 2}}]},
    )
    assert measure._compute_eer("<NE> a </NE>", "b", strict=True) == (1, 3, 1 / 3)


def test_dter_and_eer_keep_their_own_reference(monkeypatch):
    monkeypatch.setattr(measure, "ensure_pack_dir", lambda *a, **k: None)
    refs = {}

    def dter(ref, hyp, **kwargs):
        refs["display"] = ref
        return 0, 3, 0.0, {}

    def eer(ref, hyp, **kwargs):
        refs["entity"] = ref
        assert kwargs["strict"]
        return 1, 2, 0.5

    monkeypatch.setattr(measure, "_compute_dter", dter)
    monkeypatch.setattr(measure, "_compute_eer", eer)
    result = measure.eval_score(
        "<VERBAT>hello world</VERBAT>",
        "[0 1] display reference",
        extra_info={"entity_transcription": "<NE> original </NE> reference"},
        compute_eer=True,
        strict_eer=True,
    )
    assert refs == {"display": "display reference", "entity": "<NE> original </NE> reference"}
    assert result["eer_n_ref"] == 2
    assert result["eer_n_err"] == 1

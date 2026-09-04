import pytest

from recipe.phimm.reward import asr_inhouse_measure


def test_eval_score_reports_dter_p_err(monkeypatch):
    monkeypatch.setattr(asr_inhouse_measure, "ensure_pack_dir", lambda _pack_dir: None)
    monkeypatch.setattr(
        asr_inhouse_measure,
        "get_hyp_text",
        lambda _solution, version=None: "recognized text",
    )
    monkeypatch.setattr(
        asr_inhouse_measure,
        "_compute_dter",
        lambda _ref, _hyp, locale: (2, 10, 0.2, None),
    )

    result = asr_inhouse_measure.eval_score(
        "<TXT>recognized text</TXT>",
        "reference text",
        locale="en-us",
    )

    assert result["dter"] == 0.2
    assert result["dter_p_err"] == 0.2
    assert result["dter_n_err"] == 2
    assert result["dter_n_ref"] == 10


def test_eval_score_normalizes_hypothesis_whitespace(monkeypatch):
    monkeypatch.setattr(asr_inhouse_measure, "ensure_pack_dir", lambda _pack_dir: None)
    monkeypatch.setattr(
        asr_inhouse_measure,
        "get_hyp_text",
        lambda _solution, version=None: "first segment\nsecond\tsegment",
    )
    received = {}

    def compute_dter(ref, hyp, locale):
        received.update(ref=ref, hyp=hyp, locale=locale)
        return 0, 4, 0.0, None

    monkeypatch.setattr(asr_inhouse_measure, "_compute_dter", compute_dter)

    asr_inhouse_measure.eval_score("ignored", "reference text", locale="en-us")

    assert received == {
        "ref": "reference text",
        "hyp": "first segment second segment",
        "locale": "en-us",
    }


def test_compute_dter_propagates_backend_failure(monkeypatch):
    class FailingBackend:
        def compute_ter_from_strings(self, **_kwargs):
            raise ValueError("need to escape, but no escapechar set")

    monkeypatch.setattr(asr_inhouse_measure, "_get_ter_backend", lambda _locale: FailingBackend())

    with pytest.raises(ValueError, match="escapechar"):
        asr_inhouse_measure._compute_dter("reference", "hypothesis", locale="en-us")


def test_compute_dter_rejects_zero_reference_tokens(monkeypatch):
    class InvalidBackend:
        def compute_ter_from_strings(self, **_kwargs):
            return {
                "summary": {
                    "ter_info": {
                        "number_of_edits": 0,
                        "number_of_tokens": 0,
                        "display_ter": 0.0,
                    }
                }
            }

    monkeypatch.setattr(asr_inhouse_measure, "_get_ter_backend", lambda _locale: InvalidBackend())

    with pytest.raises(RuntimeError, match="zero reference tokens"):
        asr_inhouse_measure._compute_dter("reference", "hypothesis", locale="en-us")

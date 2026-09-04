import pytest

from recipe.phimm.reward import asr_eval
from recipe.phimm.reward.asr_edge import _parse_response
from recipe.phimm.utils.open_asr_normalizer import eval_utils


@pytest.mark.parametrize(
    ("solution_str", "version", "expected_fmt"),
    [
        (
            "Audio Language: English.\n<ASR><lang=English><TXT>hello world</TXT></ASR>",
            2607,
            1.0,
        ),
        ("<src=English><tgt=English>\nhello world", 2609, 1.0),
        ("hello world", 2609, 0.0),
    ],
)
def test_parse_response_extracts_hyp_text_by_version(solution_str, version, expected_fmt):
    result = _parse_response(
        solution_str,
        ground_truth="hello world",
        language="English",
        version=version,
    )

    assert result["hyp_text"] == "hello world"
    assert result["p_fmt"] == expected_fmt
    if expected_fmt:
        assert result["p_lang"] == 1.0


def test_openasr_eval_gets_versioned_hyp_text_directly(monkeypatch):
    received = {}

    def get_hyp_text(solution_str, version=None):
        received.update(solution_str=solution_str, version=version)
        return "hello world"

    monkeypatch.setattr(asr_eval, "get_hyp_text", get_hyp_text)
    monkeypatch.setattr(
        eval_utils,
        "measure_wer",
        lambda hyp, ref, lang: {"wer": 0.0, "n_err": 0, "n_ref": 2},
    )

    result = asr_eval.openasr_eval(
        "formatted response",
        "hello world",
        language="English",
        version=2607,
    )

    assert received == {"solution_str": "formatted response", "version": 2607}
    assert result == {"score": 1.0, "wer": 0.0, "n_err": 0, "n_ref": 2}
import pytest

from recipe.phimm.reward import asr_eval
from recipe.phimm.reward.asr_edge import _parse_response
from recipe.phimm.utils.open_asr_normalizer import eval_utils


@pytest.mark.parametrize("version", [None, 2607, 2609])
@pytest.mark.parametrize(
    ("solution_str", "expected_n_err"),
    [
        ("<nonspeech>", 0),
        ("  <nonspeech>  ", 0),
        ("", 0),
        ("  ", 0),
        (".", 0),
        ("<TXT></TXT>", 0),
        ("speech", 1),
        ("<nonspeech> speech", 1),
        ("speech <nonspeech>", 1),
        ("<nonspeech>.", 0),
    ],
)
def test_non_speech_eval_requires_empty_hyp_text(solution_str, expected_n_err, version):
    result = asr_eval.non_speech_eval(solution_str, ground_truth="", version=version)

    assert result == {
        "score": 1.0 - expected_n_err,
        "wer": float(expected_n_err),
        "n_err": expected_n_err,
        "n_ref": 1,
    }


@pytest.mark.parametrize("version", [2607, 2609])
@pytest.mark.parametrize(
    ("texts", "expected_n_err"),
    [
        (("<nonspeech>",), 0),
        (("  <nonspeech>  ",), 0),
        (("<nonspeech>.",), 0),
        ((".",), 0),
        (("<sep>",), 0),
        (("speech",), 1),
        (("<nonspeech>", "<nonspeech>"), 0),
        (("<nonspeech>", "speech"), 1),
        (("speech", "<nonspeech>"), 1),
    ],
)
def test_non_speech_eval_checks_cleaned_segments(version, texts, expected_n_err):
    if version == 2607:
        segments = "".join(f"<lang=Unknown><TXT>{text}</TXT>" for text in texts)
        response = f"Audio Language: Unknown.\n<ASR>{segments}</ASR>"
    else:
        response = "\n".join(f"<src=Unknown><tgt=Unknown>\n{text}" for text in texts)

    result = asr_eval.non_speech_eval(response, ground_truth="", version=version)

    assert result == {
        "score": 1.0 - expected_n_err,
        "wer": float(expected_n_err),
        "n_err": expected_n_err,
        "n_ref": 1,
    }


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
        lambda hyp, ref, lang, merge_compounds: {"wer": 0.0, "n_err": 0, "n_ref": 2},
    )

    result = asr_eval.openasr_eval(
        "formatted response",
        "hello world",
        language="English",
        version=2607,
    )

    assert received == {"solution_str": "formatted response", "version": 2607}
    assert result == {"score": 1.0, "wer": 0.0, "n_err": 0, "n_ref": 2}


def test_openasr_evals_use_versioned_compound_aware_wer():
    response = "Audio Language: English.\n<ASR><lang=English><TXT>icecream</TXT></ASR>"

    legacy_result = asr_eval.openasr_eval(
        response,
        "ice cream",
        language="English",
        version=2607,
        merge_compounds=False,
    )
    compound_result = asr_eval.openasr_eval(
        response,
        "ice cream",
        language="English",
        version=2607,
    )
    en_result = asr_eval.openasr_en_eval(
        response,
        "ice cream",
        version=2607,
    )

    assert legacy_result["wer"] > 0
    assert compound_result == {"score": 1.0, "wer": 0.0, "n_err": 0, "n_ref": 1}
    assert en_result == {
        "score": 1.0,
        "wer": 0.0,
        "kw_acc": 1.0,
        "n_err": 0,
        "n_ref": 2,
        "nb_err": 0,
        "nb_ref": 0,
    }


def test_openasr_en_eval_reports_keyword_error_counts():
    result = asr_eval.openasr_en_eval(
        "the quick blue fox",
        "the quick brown fox",
        extra_info={"keywords": ["brown"]},
    )

    assert result["nb_err"] == 1
    assert result["nb_ref"] == 1


def test_openasr_en_eval_uses_hf_english_for_keywords():
    result = asr_eval.openasr_en_eval(
        "connect wifi",
        "connect wi fi",
        extra_info={"keywords": ["wifi"]},
    )

    assert result["nb_err"] == 0
    assert result["nb_ref"] == 1


def test_measure_openasr_en_wer_exposes_normalized_error_breakdown():
    result = asr_eval.measure_openasr_en_wer("icecream today", "ice cream tomorrow")

    assert result == {
        "wer": 1 / 3,
        "n_err": 1,
        "n_ref": 3,
        "n_ins": 0,
        "n_del": 0,
        "n_sub": 1,
        "normalized_ref": "ice cream tomorrow",
        "normalized_hyp": "icecream today",
    }

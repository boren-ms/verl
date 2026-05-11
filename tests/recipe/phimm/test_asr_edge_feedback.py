from recipe.phimm.reward.asr_edge import collect_wrong_ref_words, compute_score


def _asr_response(text: str, lang: str = "English") -> str:
    return f"<ASR><lang={lang}><TXT>{text}</TXT></ASR>"


def test_collect_wrong_ref_words_uses_substituted_and_deleted_reference_words():
    feedback = collect_wrong_ref_words(
        hyp="hello brave new",
        ref="hello bright new world",
        tgt_lang="english",
        unit="word",
    )

    assert feedback == "bright, world"


def test_compute_score_returns_wrong_words_as_feedback():
    result = compute_score(
        solution_str=_asr_response("hello brave new"),
        ground_truth="hello bright new world",
        extra_info={"language": "English"},
        unit="word",
    )

    assert result["feedback"] == "bright, world"


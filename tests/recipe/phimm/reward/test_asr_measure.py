import pytest

from recipe.phimm.reward.asr_measure import _parse_task_output, check_fmt, check_lang, lang_score


def test_accepts_headerless_code_switch_output():
    output = (
        "<ASR><lang=Chinese><TXT>祖父叶与良。</TXT>\n"
        "<lang=Italian><TXT>E, inoltre, attore.</TXT></ASR>"
    )

    assert _parse_task_output(output) == (
        [],
        ["Chinese", "Italian"],
        ["祖父叶与良。", "E, inoltre, attore."],
    )
    assert check_fmt(output)
    assert check_lang(output, "Chinese Italian") == 1.0


def test_header_language_must_still_match_segments():
    output = "Audio Language: English.\n<ASR><lang=French><TXT>Bonjour</TXT></ASR>"

    assert not check_fmt(output)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("<ASR><lang=English><TXT>Hello</TXT></ASR>", 1.0),
        ("<ASR><lang=French><TXT>Bonjour</TXT></ASR>", 0.0),
        ("malformed", 0.0),
    ],
)
def test_lang_score_reports_only_p_lang(output, expected):
    assert lang_score(output, language="English") == {
        "score": expected,
        "p_lang": expected,
    }
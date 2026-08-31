import pytest

from recipe.phimm.reward.asr_measure import (
    _parse_response,
    _parse_task_output,
    check_fmt,
    check_lang,
    lang_score,
)


def test_accepts_code_switch_output():
    output = (
        "<src=Chinese><tgt=Chinese>\n祖父叶与良。\n"
        "<src=Italian><tgt=Italian>\nE, inoltre, attore."
    )
    task_output = _parse_task_output(output)

    assert task_output == (
        ["Chinese", "Italian"],
        ["Chinese", "Italian"],
        ["祖父叶与良。", "E, inoltre, attore."],
    )
    assert check_fmt(task_output)
    assert check_lang(task_output, "Chinese Italian") == 1.0


def test_source_and_target_languages_must_match():
    output = "<src=English><tgt=French>\nBonjour"

    assert not check_fmt(_parse_task_output(output))


def test_parse_response_uses_structured_task_output():
    result = _parse_response(
        "<src=English><tgt=English>\nhello world",
        ground_truth="hello world",
        language="English",
    )

    assert result["word"] == 1.0
    assert result["lang"] == 1.0
    assert result["fmt"] == 1.0


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("<src=English><tgt=English>\nHello", 1.0),
        ("<src=French><tgt=French>\nBonjour", 0.0),
        ("malformed", 0.0),
    ],
)
def test_lang_score_reports_only_p_lang(output, expected):
    assert lang_score(output, language="English") == {
        "score": expected,
        "p_lang": expected,
    }
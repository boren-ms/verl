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


@pytest.mark.parametrize("mode_tag", ["LEXICAL", "verbatim", "ASR_READABLE"])
def test_parse_task_output_removes_asr_mode_tags(mode_tag):
    output = f"<src=English><tgt=English> \n<{mode_tag}>\nOK OK i think if you have"

    task_output = _parse_task_output(output)

    assert task_output == (
        ["English"],
        ["English"],
        ["OK OK i think if you have"],
    )
    assert check_fmt(task_output)


def test_parse_task_output_accepts_text_without_language_header():
    task_output = _parse_task_output("<VERBATIM>\nShe's pregnant.")

    assert task_output == ([], [], ["She's pregnant."])
    assert check_fmt(task_output)
    assert check_lang(task_output, "English") == 1.0


def test_parse_response_uses_text_without_language_header():
    result = _parse_response(
        "<VERBATIM>\nShe's pregnant.",
        ground_truth="She's pregnant.",
        language="English",
    )

    assert result["word"] == 1.0
    assert result["fmt"] == 1.0
    assert result["lang"] == 1.0


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
        ("<src=English><tgt=English>Hello", 0.0),
    ],
)
def test_lang_score_reports_only_p_lang(output, expected):
    assert lang_score(output, language="English") == {
        "score": expected,
        "p_lang": expected,
    }
import pytest

from recipe.phimm.reward.asr_measure import (
    _parse_response,
    _parse_task_output,
    check_fmt,
    check_lang,
    compute_kw_acc,
    get_asr_text,
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


def test_format_and_language_ignore_source_language():
    output = "<src=English><tgt=French>\nBonjour"
    task_output = _parse_task_output(output)

    assert check_fmt(task_output)
    assert check_lang(task_output, "French") == 1.0


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


def test_get_asr_text_uses_task_output():
    task_output = _parse_task_output(
        "<src=English><tgt=English>\nhello\n<src=Chinese><tgt=Chinese>\n你好"
    )

    assert get_asr_text(task_output) == "hello 你好"


@pytest.mark.parametrize("tag", ["ASR", "ASR_LEXICAL", "ASR_VERBATIM", "ASR_READABLE"])
def test_parse_task_output_accepts_2607_response_format(tag):
    output = f"Audio Language: English.\n<{tag}><lang=English><TXT>hello world</TXT></{tag}>"

    task_output = _parse_task_output(output, version=2607)

    assert task_output == (["English"], ["English"], ["hello world"])
    assert check_fmt(task_output)
    assert check_lang(task_output, "English") == 1.0


def test_parse_task_output_accepts_2607_code_switch_response():
    output = (
        "Audio Language: English and Chinese.\n"
        "<ASR><lang=English><TXT>hello</TXT><lang=Chinese><TXT>你好</TXT></ASR>"
    )

    task_output = _parse_task_output(output, version=2607)

    assert task_output == (["English", "Chinese"], ["English", "Chinese"], ["hello", "你好"])
    assert check_fmt(task_output)
    assert check_lang(task_output, "English Chinese") == 1.0


def test_parse_response_uses_2607_response_text():
    result = _parse_response(
        "Audio Language: English.\n<ASR><lang=English><TXT>hello world</TXT></ASR>",
        ground_truth="hello world",
        language="English",
        version=2607,
    )

    assert result["word"] == 1.0
    assert result["lang"] == 1.0
    assert result["fmt"] == 1.0


def test_parse_response_accepts_2607_response_without_audio_language():
    result = _parse_response(
        "<ASR><lang=English><TXT>hello world</TXT></ASR>",
        ground_truth="hello world",
        language="English",
        version=2607,
    )

    assert result["word"] == 1.0
    assert result["lang"] == 1.0
    assert result["fmt"] == 1.0

    task_output = _parse_task_output(
        "<ASR><lang=English><TXT>hello world</TXT></ASR>",
        version=2607,
    )
    assert task_output == ([], ["English"], ["hello world"])


@pytest.mark.parametrize(
    ("reference", "hypothesis", "keywords", "expected"),
    [
        ("the quick brown fox", "the quick brown fox", ["brown"], 1.0),
        ("the quick brown fox", "the quick blue fox", ["brown"], 0.0),
        ("the quick brown fox", "the quick fox", ["brown"], 0.0),
        ("the quick brown fox", "the quick brown brown fox", ["brown"], 0.0),
        ("new york state", "new york state", ["new york"], 1.0),
        ("new york state", "new state", ["new york"], 0.5),
        ("the quick brown fox", "the quick blue fox", None, 1.0),
    ],
)
def test_compute_kw_acc(reference, hypothesis, keywords, expected):
    result = compute_kw_acc(reference, hypothesis, keywords)

    assert result == expected


def test_parse_response_reports_keyword_accuracy():
    result = _parse_response(
        "the quick blue fox",
        ground_truth="the quick brown fox",
        extra_info={"keywords": ["brown"]},
    )

    assert result["keyword"] == 0.0


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
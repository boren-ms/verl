import pytest

from recipe.phimm.data.prompts import (
    get_task_output,
    get_task_prefix,
    get_task_prompt,
    resolve_task_language,
)
from recipe.phimm.reward.asr_measure import check_fmt
from recipe.phimm.reward.asr_response import parse_task_output


def test_get_task_output_formats_mixed_components():
    output = get_task_output(
        task="lang_asr",
        lang="en_zh",
        text="hello 你好",
        components=[
            {"language": "en", "text": "hello"},
            {"language": "zh", "text": "你好"},
        ],
    )

    assert output == (
        "<src=English><tgt=English>\n<TXT>hello</TXT>\n"
        "<src=Chinese><tgt=Chinese>\n<TXT>你好</TXT>"
    )
    assert check_fmt(parse_task_output(output, version=2609))


def test_get_task_output_preserves_single_language_format():
    assert get_task_output(task="lang_asr", lang="en", text="hello") == (
        "<src=English><tgt=English>\n<TXT>hello</TXT>"
    )


def test_get_task_output_supports_new_2609_format():
    output = get_task_output(task="lang_asr", lang="en", text="hello", version=2609)

    assert output == "<src=English><tgt=English>\n<TXT>hello</TXT>"
    assert parse_task_output(output, version=2609) == [
        {"src": "English", "tgt": "English", "text": "hello"}
    ]


@pytest.mark.parametrize(
    ("task", "mode"),
    [
        ("lang_asr_lex", "LEXICAL"),
        ("lang_asr_verb", "VERBATIM"),
    ],
)
def test_get_task_output_supports_2609_mode_tags(task, mode):
    output = get_task_output(task=task, lang="en", text="hello", version="2609")

    assert output == f"<src=English><tgt=English>\n<{mode}>\n<TXT>hello</TXT>"
    assert parse_task_output(output, version=2609) == [
        {"src": "English", "tgt": "English", "text": "hello"}
    ]


def test_get_task_output_supports_2609_source_target_languages():
    output = get_task_output(
        task="lang_asr",
        lang="es",
        text="hello",
        components=[
            {
                "language": "Spanish",
                "target_language": "English",
                "text": "hello",
            }
        ],
        version=2609,
    )

    assert output == "<src=Spanish><tgt=English>\n<TXT>hello</TXT>"


def test_get_task_output_keeps_non_lid_2609_output_plain():
    assert get_task_output(task="asr", text="hello", version=2609) == "hello"


def test_get_task_output_supports_2607_format():
    output = get_task_output(task="lang_asr", lang="en", text="hello", version=2607)

    assert output == "Audio Language: English.\n<ASR><lang=English><TXT>hello</TXT></ASR>"
    assert parse_task_output(output, version=2607) == [
        {"src": "English", "tgt": "English", "text": "hello"}
    ]


def test_get_task_output_rejects_2607_language_list_without_components():
    with pytest.raises(ValueError, match="requires per-language components"):
        get_task_output(
            task="lang_asr",
            lang="Hebrew Hindi",
            text="מאין לך זאת? सुप्रीम कोर्ट",
            version=2607,
        )


def test_task_prefix_and_output_support_2607_completion_format():
    prefix = get_task_prefix(task="lang_asr", lang="en", version=2607)
    output = get_task_output(task="lang_asr", lang="en", text="hello", version=2607)

    assert prefix == "Audio Language: English\n"
    assert output == "Audio Language: English.\n<ASR><lang=English><TXT>hello</TXT></ASR>"
    assert parse_task_output(output, version=2607) == [
        {"src": "English", "tgt": "English", "text": "hello"}
    ]


def test_get_task_prefix_supports_2607_multiple_languages():
    prefix = get_task_prefix(task="lang_asr", lang="en_zh", version=2607)

    assert prefix == "Audio Language: English and Chinese\n"


def test_get_task_output_supports_2607_mixed_components():
    output = get_task_output(
        task="lang_asr",
        lang="en_zh",
        text="hello 你好",
        components=[
            {"language": "en", "text": "hello"},
            {"language": "zh", "text": "你好"},
        ],
        version="2607",
    )

    assert output == (
        "Audio Language: English and Chinese.\n"
        "<ASR><lang=English><TXT>hello</TXT>\n<lang=Chinese><TXT>你好</TXT></ASR>"
    )
    assert parse_task_output(output, version=2607) == [
        {"src": "English", "tgt": "English", "text": "hello"},
        {"src": "Chinese", "tgt": "Chinese", "text": "你好"},
    ]


def test_lang_asr_verb_uses_verbatim_prompt_and_output_format():
    assert get_task_prompt(task="lang_asr_verb", rand=False) == (
        "Detect the language and transcribe the audio clip into text. "
        "Transcribe verbatim, including all filler words and disfluencies."
    )
    assert get_task_output(task="lang_asr_verb_en", lang="en", text="um hello") == (
        "<src=English><tgt=English>\n<VERBATIM>\n<TXT>um hello</TXT>"
    )
    assert resolve_task_language(task="lang_asr_verb_en") == "English"


def test_get_task_prompt_supports_2609_known_language():
    assert get_task_prompt(
        task="lang_asr", rand=False, version=2609, lang="English"
    ) == "Transcribe the audio clip into text.<audio>\nThe language is English."


def test_get_task_prompt_supports_2609_known_languages_and_mode():
    assert get_task_prompt(
        task="lang_asr_verb",
        rand=False,
        version=2609,
        lang="English Spanish",
    ) == (
        "Transcribe the audio clip into text. "
        "Transcribe verbatim, including all filler words and disfluencies.<audio>\n"
        "The languages are English and Spanish."
    )


def test_resolve_task_language_samples_supported_language(monkeypatch):
    monkeypatch.setattr("recipe.phimm.data.prompts.random.choice", lambda languages: "french")

    assert resolve_task_language(task="lang_asr", lang="random") == "French"
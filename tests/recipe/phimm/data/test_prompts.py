import pytest

from recipe.phimm.data.prompts import (
    get_think,
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


def test_keyword_think_adds_prompt_instruction_and_precedes_2607_output():
    prompt = get_task_prompt(
        task="lang_asr_verb",
        rand=False,
        version=2607,
        think="keyword",
    )
    output = get_task_output(
        task="lang_asr_verb",
        lang="en",
        text="um hello",
        version=2607,
        think="keyword",
        keywords=["rare phrase", "surname"],
    )

    assert prompt == (
        "Detect the language and transcribe the audio clip into text. "
        "Transcribe verbatim, including all filler words and disfluencies. "
        "Think about the rare words in the audio first before transcribing."
    )
    assert output == (
        "<think>rare phrase,surname</think>\n"
        "Audio Language: English.\n"
        "<ASR><lang=English><TXT>um hello</TXT></ASR>"
    )


@pytest.mark.parametrize(
    ("think", "keywords", "expected"),
    [
        (None, ["rare phrase"], ""),
        ("keyword", [" rare phrase ", "", "surname"], "<think>rare phrase,surname</think>\n"),
        ("keyword", None, "<think></think>\n"),
    ],
)
def test_get_think_formats_mode_output(think, keywords, expected):
    assert get_think(think, keywords) == expected


def test_null_think_keeps_standard_prompt_and_output():
    assert get_task_prompt(
        task="lang_asr_verb",
        rand=False,
        version=2607,
        think=None,
    ) == (
        "Detect the language and transcribe the audio clip into text. "
        "Transcribe verbatim, including all filler words and disfluencies."
    )
    assert get_task_output(
        task="lang_asr_verb",
        lang="en",
        text="um hello",
        version=2607,
        think=None,
        keywords=["rare phrase"],
    ) == "Audio Language: English.\n<ASR><lang=English><TXT>um hello</TXT></ASR>"


def test_unknown_think_mode_is_rejected():
    with pytest.raises(ValueError, match="Unsupported think mode"):
        get_task_prompt(task="lang_asr_verb", think="reasoning")


@pytest.mark.parametrize("version", [2607, 2609])
def test_get_task_prompt_supports_known_language_for_all_versions(version):
    assert get_task_prompt(
        task="lang_asr", rand=False, version=version, lang="English"
    ) == (
        "Detect the language and transcribe the audio clip into text.<audio>\n"
        "The language is English."
    )


def test_get_task_prompt_supports_2609_known_languages_and_mode():
    assert get_task_prompt(
        task="lang_asr_verb",
        rand=False,
        version=2609,
        lang="English Spanish",
    ) == (
        "Detect the language and transcribe the audio clip into text. "
        "Transcribe verbatim, including all filler words and disfluencies.<audio>\n"
        "The languages are English and Spanish."
    )


def test_resolve_task_language_samples_supported_language(monkeypatch):
    monkeypatch.setattr("recipe.phimm.data.prompts.random.choice", lambda languages: "french")

    assert resolve_task_language(task="lang_asr", lang="random") == "French"
import pytest

from recipe.phimm.data.prompts import (
    get_task_output,
    get_task_prefix,
    get_task_prompt,
    resolve_task_language,
)
from recipe.phimm.reward.asr_measure import check_fmt, check_lang
from recipe.phimm.reward.asr_response import parse_task_output
from recipe.phimm.utils.languages import get_language_code, get_language_name


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("Assamese", "as"),
        ("Bengali", "bn"),
        ("Gujarati", "gu"),
        ("Hindi", "hi"),
        ("Kannada", "kn"),
        ("Malayalam", "ml"),
        ("Marathi", "mr"),
        ("Odia", "or"),
        ("Punjabi", "pa"),
        ("Tamil", "ta"),
        ("Telugu", "te"),
        ("Urdu", "ur"),
    ],
)
def test_indian_languages_normalize_names_codes_prompts_and_rewards(name, code):
    for identifier in (name, name.lower(), name.upper(), code, code.upper()):
        assert get_language_name(identifier) == name
        assert get_language_code(identifier) == code
        assert resolve_task_language("lang_asr", lang=identifier) == name
        assert resolve_task_language(f"lang_asr_{identifier}") == name
        assert get_task_prompt(task="lang_asr", lang=identifier, version=2609) == (
            f"Transcribe the audio clip into text.<audio>\nThe language is {name}."
        )
        output = get_task_output(task="lang_asr", lang=identifier, text="sample", version=2609)
        assert output == f"<src={name}><tgt={name}>\nsample"
        assert check_lang(parse_task_output(output, version=2609), code) == {
            "acc": 1.0, "n_err": 0, "n_ref": 1,
        }


def test_indian_languages_normalize_code_switch_identifiers():
    assert get_language_name("as_or_pa") == "Assamese Odia Punjabi"
    assert get_language_code("Assamese Odia Punjabi") == "as_or_pa"


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
        "<src=English><tgt=English>\nhello\n"
        "<src=Chinese><tgt=Chinese>\n你好"
    )
    assert check_fmt(parse_task_output(output, version=2609))


def test_get_task_output_preserves_single_language_format():
    assert get_task_output(task="lang_asr", lang="en", text="hello") == (
        "<src=English><tgt=English>\nhello"
    )


def test_get_task_output_supports_new_2609_format():
    output = get_task_output(task="lang_asr", lang="en", text="hello", version=2609)

    assert output == "<src=English><tgt=English>\nhello"
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

    assert output == f"<src=English><tgt=English>\n<{mode}>\nhello"
    assert parse_task_output(output, version=2609) == [
        {"src": "English", "tgt": "English", "text": f"<{mode}>\nhello"}
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

    assert output == "<src=Spanish><tgt=English>\nhello"


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
        "<src=English><tgt=English>\n<VERBATIM>\num hello"
    )
    assert resolve_task_language(task="lang_asr_verb_en") == "English"


@pytest.mark.parametrize("version", [None, 2607])
def test_get_task_prompt_uses_detect_language_style_without_lang(version):
    assert get_task_prompt(
        task="lang_asr", rand=False, version=version
    ) == "Detect the language and transcribe the audio clip into text."


def test_get_task_prompt_uses_moe_style_for_2609_known_language():
    assert get_task_prompt(
        task="lang_asr", rand=False, version=2609, lang="English"
    ) == "Transcribe the audio clip into text.<audio>\nThe language is English."


def test_get_task_prompt_uses_moe_style_when_language_hint_enabled_without_version():
    assert get_task_prompt(
        task="lang_asr", rand=False, lang="English"
    ) == "Transcribe the audio clip into text.<audio>\nThe language is English."


def test_get_task_prompt_does_not_rewrite_2609_prompt_without_language_hint():
    assert get_task_prompt(
        task="lang_asr", rand=False, version=2609, lang=None
    ) == "Detect the language and transcribe the audio clip into text."


def test_get_task_prompt_uses_moe_style_for_known_mixed_languages():
    assert get_task_prompt(
        task="lang_asr",
        rand=False,
        version=2609,
        lang="Hungarian Turkish",
    ) == (
        "Transcribe the audio clip into text.<audio>\n"
        "The languages are Hungarian and Turkish."
    )


def test_get_task_prompt_supports_2609_known_languages_and_mode():
    assert get_task_prompt(
        task="lang_asr_verb",
        rand=False,
        version=2609,
        lang="English Spanish",
    ) == (
        "Transcribe the audio clip into text. Transcribe verbatim, including all filler words "
        "and disfluencies.<audio>\n"
        "The languages are English and Spanish."
    )


def test_resolve_task_language_samples_supported_language(monkeypatch):
    monkeypatch.setattr("recipe.phimm.data.prompts.random.choice", lambda languages: "french")

    assert resolve_task_language(task="lang_asr", lang="random") == "French"
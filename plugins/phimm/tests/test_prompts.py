"""Style, language, and language-selection behavior for ASR prompts and outputs."""

import pytest
from recipe.phimm.data.prompts import get_task_output, get_task_prompt
from recipe.phimm.reward.asr_response import get_hyp_text


@pytest.mark.parametrize(
    "style,tag,instruction",
    [
        ("verb", "VERBATIM", "Transcribe verbatim, including all filler words and disfluencies."),
        ("lex", "LEXICAL", "Output must be in lexical format."),
        ("display", "READABLE", ""),
    ],
)
def test_style_and_language(style, tag, instruction):
    prompt = get_task_prompt(style, "en")
    assert prompt.count("<audio>") == 1
    expected = "Transcribe the audio clip into text."
    if instruction:
        expected += f" {instruction}"
    assert prompt == f"{expected}<audio>\nThe language is English."
    assert "Only include" not in prompt
    assert "Omit speech" not in prompt
    assert prompt.endswith("The language is English.")
    output = get_task_output(style, "en", "Hello")
    assert output == f"<src=English><tgt=English>\n<{tag}>\nHello"
    assert get_hyp_text(output) == "Hello"


@pytest.mark.parametrize("style", ["asr", "rare_asr", "biasing", "invalid"])
@pytest.mark.parametrize("api", [get_task_prompt, get_task_output])
def test_unknown_style_is_rejected(api, style):
    with pytest.raises(ValueError, match="Unknown style"):
        api(style=style)


@pytest.mark.parametrize("style", ["verb", "lex", "display"])
def test_prompt_without_language_requests_detection(style):
    prompt = get_task_prompt(style=style)
    assert prompt.startswith("Detect the language and transcribe")
    assert prompt.endswith("<audio>")
    assert prompt.count("<audio>") == 1


def test_language_hint_can_describe_multiple_languages():
    prompt = get_task_prompt(language="en_fr")
    assert prompt.endswith("<audio>\nThe languages are English and French.")
    assert prompt.count("<audio>") == 1


@pytest.mark.parametrize("style", ["verb", "lex", "display"])
def test_language_hint_can_be_disabled(style):
    prompt = get_task_prompt(style=style, language="English", lang_hint=False)
    assert prompt.startswith("Detect the language and transcribe")
    assert prompt.endswith("<audio>")
    assert "The language is" not in prompt
    assert "Only include" not in prompt


def test_components_preserve_prepared_references():
    components = [{"language": "en", "text": "Hello"}, {"language": "fr", "text": "Bonjour"}]
    output = get_task_output(language="English", components=components)
    assert "<src=English><tgt=English>\n<VERBATIM>\nHello" in output
    assert "<src=French><tgt=French>\n<VERBATIM>\nBonjour" in output


def test_wholly_foreign_components_are_not_silently_discarded():
    assert get_task_output(language="English", components=[{"language": "fr", "text": "Bonjour"}]) == (
        "<src=French><tgt=French>\n<VERBATIM>\nBonjour"
    )


def test_components_keep_explicit_translation_targets():
    assert get_task_output("lex", components=[{"language": "fr", "target_language": "en", "text": "Hello"}]) == (
        "<src=French><tgt=English>\n<LEXICAL>\nHello"
    )

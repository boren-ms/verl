"""Style, language, and language-selection behavior for ASR prompts and outputs."""

import pytest
from recipe.phimm.data.prompts import get_task_output, get_task_prompt
from recipe.phimm.reward.asr_response import get_hyp_text


@pytest.mark.parametrize(
    "style,tag,instruction",
    [
        ("verb", "VERBATIM", "Transcribe verbatim, including all filler words and disfluencies."),
        ("lex", "LEXICAL", "Output must be in lexical format."),
        ("display", "READABLE", "Output must be in display format."),
    ],
)
@pytest.mark.parametrize("target_only", [False, True])
def test_style_language_and_selection(style, tag, instruction, target_only):
    prompt = get_task_prompt(style, "en", target_only)
    assert prompt.count("<audio>") == 1
    assert prompt.startswith(f"Transcribe the audio clip into text. {instruction}")
    if target_only:
        assert "Only include speech in English." in prompt
        assert "Omit speech in other languages; do not translate it." in prompt
        assert prompt.endswith("The language is English.")
    else:
        assert "Only include" not in prompt
        assert prompt.endswith("The language is English.")
    output = get_task_output(style, "en", "Hello", target_only=target_only)
    assert output == f"<src=English><tgt=English>\n<{tag}>\nHello"
    assert get_hyp_text(output) == "Hello"


@pytest.mark.parametrize("style", ["asr", "rare_asr", "biasing", "invalid"])
@pytest.mark.parametrize("api", [get_task_prompt, get_task_output])
def test_unknown_style_is_rejected(api, style):
    with pytest.raises(ValueError, match="Unknown style"):
        api(style=style)


@pytest.mark.parametrize("language", [None, "Unknown", "English French"])
@pytest.mark.parametrize("api", [get_task_prompt, get_task_output])
def test_target_only_requires_one_known_language(api, language):
    with pytest.raises(ValueError, match="requires one known language"):
        api(language=language, target_only=True)


@pytest.mark.parametrize("api", [get_task_prompt, get_task_output])
def test_target_only_requires_an_explicit_language(api):
    with pytest.raises(ValueError, match="requires one known language"):
        api(target_only=True)


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
@pytest.mark.parametrize("target_only", [False, True])
def test_language_hint_can_be_disabled_independently(style, target_only):
    prompt = get_task_prompt(style=style, language="English", target_only=target_only, lang_hint=False)
    assert prompt.startswith("Detect the language and transcribe")
    assert prompt.endswith("<audio>")
    assert "The language is" not in prompt
    assert ("Only include speech in English." in prompt) == target_only


@pytest.mark.parametrize("target_only", [False, True])
def test_components_preserve_prepared_references(target_only):
    components = [{"language": "en", "text": "Hello"}, {"language": "fr", "text": "Bonjour"}]
    output = get_task_output(language="English", components=components, target_only=target_only)
    assert "<src=English><tgt=English>\n<VERBATIM>\nHello" in output
    assert "<src=French><tgt=French>\n<VERBATIM>\nBonjour" in output


def test_wholly_foreign_components_are_not_silently_discarded():
    assert get_task_output(
        language="English", components=[{"language": "fr", "text": "Bonjour"}], target_only=True
    ) == ("<src=French><tgt=French>\n<VERBATIM>\nBonjour")


def test_components_keep_explicit_translation_targets():
    assert get_task_output("lex", components=[{"language": "fr", "target_language": "en", "text": "Hello"}]) == (
        "<src=French><tgt=English>\n<LEXICAL>\nHello"
    )

"""Style, language, and language-selection behavior for ASR prompts and outputs."""

import json

import pytest
from recipe.phimm.data.prompts import get_task_output, get_task_prompt
from recipe.phimm.reward.asr_response import get_hyp_text


@pytest.mark.parametrize(
    "style,instruction",
    [
        ("verb", "Transcribe verbatim, including all filler words and disfluencies."),
        ("lex", "Output must be in lexical format."),
        ("display", ""),
    ],
)
def test_style_and_language(style, instruction):
    prompt = get_task_prompt(style, "en")
    assert prompt.count("<audio>") == 1
    expected = "Transcribe the audio clip into text."
    if instruction:
        expected += f" {instruction}"
    assert prompt.startswith(f"{expected}<audio>\nThe language is English.\nReturn JSONL only:")
    assert "Only include" not in prompt
    assert "Omit speech" not in prompt
    assert "The language is English." in prompt
    assert '{"language":"target","text":"transcription"}' in prompt
    output = get_task_output(style, "en", "Hello")
    assert json.loads(output) == {"language": "English", "text": "Hello"}
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
    assert "<audio>\nReturn JSONL only" in prompt
    assert prompt.count("<audio>") == 1


def test_language_hint_can_describe_multiple_languages():
    prompt = get_task_prompt(language="en_fr")
    assert "<audio>\nThe languages are English and French." in prompt
    assert prompt.count("<audio>") == 1


@pytest.mark.parametrize("style", ["verb", "lex", "display"])
def test_language_hint_can_be_disabled(style):
    prompt = get_task_prompt(style=style, language="English", lang_hint=False)
    assert prompt.startswith("Detect the language and transcribe")
    assert "<audio>\nReturn JSONL only" in prompt
    assert "The language is" not in prompt
    assert "Only include" not in prompt


def test_components_preserve_prepared_references():
    components = [{"language": "en", "text": "Hello"}, {"language": "fr", "text": "Bonjour"}]
    output = get_task_output(language="English", components=components)
    assert list(map(json.loads, output.splitlines())) == [
        {"language": "English", "text": "Hello"},
        {"language": "French", "text": "Bonjour"},
    ]


def test_wholly_foreign_components_are_not_silently_discarded():
    output = get_task_output(language="English", components=[{"language": "fr", "text": "Bonjour"}])
    assert json.loads(output) == {"language": "French", "text": "Bonjour"}


def test_components_keep_explicit_translation_targets():
    output = get_task_output(
        "lex", text="Bonjour", components=[{"language": "fr", "target_language": "en", "text": "Hello"}]
    )
    assert json.loads(output) == {"language": "English", "text": "Hello"}


def test_jsonl_target_keeps_segment_text_and_escapes_embedded_newlines():
    text = '你好，"hello"\n世界'
    components = [{"language": "zh", "text": text}, {"language": "en", "text": "Next"}]
    lines = get_task_output(components=components).splitlines()
    assert list(map(json.loads, lines)) == [
        {"language": "Chinese", "text": text},
        {"language": "English", "text": "Next"},
    ]

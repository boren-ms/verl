# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Task prompts and response formatting for model 2609."""

import random

from recipe.phimm.utils.languages import LANGUAGES, get_language_name

STYLES = {
    "verb": ("Transcribe verbatim, including all filler words and disfluencies.", "VERBATIM"),
    "lex": ("Output must be in lexical format.", "LEXICAL"),
    "display": ("", "READABLE"),
}

BIASING_POSITIONS = ("before_audio", "after_audio")
BIAS_INSTRUCTION = "Pay extra attention to the following phrases:"


def resolve_language(language):
    """Normalize language names/codes and resolve a random language if requested."""
    if language and language.lower() == "random":
        language = random.choice(tuple(LANGUAGES))
    return get_language_name(language)


def add_language_hint(prompt, languages):
    """Request detection or append a hint using already-resolved language names."""
    if not languages:
        return "Detect the language and " + prompt[0].lower() + prompt[1:]
    if len(languages) == 1:
        prompt += f"\nThe language is {languages[0]}."
    elif languages:
        prompt += f"\nThe languages are {' and '.join(languages)}."
    return prompt


def add_biasing_context(prompt, biasing_words, *, position):
    """Insert validated biasing candidates at the configured position in a base prompt."""
    if position not in BIASING_POSITIONS:
        raise ValueError(f"Unknown biasing position: {position!r}; expected {BIASING_POSITIONS}")
    if not isinstance(biasing_words, list | tuple) or any(
        not isinstance(word, str) or not word.strip() or "<audio>" in word for word in biasing_words
    ):
        raise ValueError("biasing_words must be a list of nonempty strings without <audio> markers")
    context = ", ".join(f"*{word}*" for word in dict.fromkeys(biasing_words))
    bias_prompt = f"{BIAS_INSTRUCTION} {context}."
    if position == "before_audio":
        return prompt.replace("<audio>", f" {bias_prompt}<audio>", 1)
    # Preserve the MoE format-2 ordering: instruction, audio, language hint, bias.
    separator = " " if not prompt.endswith("<audio>") else "\n"
    return f"{prompt}{separator}{bias_prompt}"


def get_task_prompt(style="verb", language=None, *, lang_hint=True, biasing=None, biasing_words=None):
    """Build a complete ASR prompt with its audio marker and optional language hint."""
    if style not in STYLES:
        raise ValueError(f"Unknown style: {style!r}; expected verb, lex, or display")
    prompt = "Transcribe the audio clip into text."
    if instruction := STYLES[style][0]:
        prompt += f" {instruction}"
    languages = resolve_language(language).split() if language else []
    prompt = add_language_hint(f"{prompt}<audio>", languages if lang_hint else [])
    if biasing is not None and biasing_words:
        prompt = add_biasing_context(prompt, biasing_words, position=biasing)
    return prompt


def get_task_output(style="verb", language=None, text="", components=None):
    """Format prepared references; language selection belongs to preprocessing."""
    if style not in STYLES:
        raise ValueError(f"Unknown style: {style!r}; expected verb, lex, or display")
    language = resolve_language(language)
    segments = components or [{"language": language, "text": text}]
    outputs = []
    for segment in segments:
        source = segment.get("language", language)
        target = segment.get("target_language", source)
        source = resolve_language(source)
        target = resolve_language(target)
        outputs.append(f"<src={source}><tgt={target}>\n<{STYLES[style][1]}>\n{segment.get('text', '')}")
    return "\n".join(outputs)

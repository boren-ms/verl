# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Task prompts and response formatting for model 2609."""

import random

from recipe.phimm.utils.languages import LANGUAGES, get_language_name

STYLES = {
    "verb": ("Transcribe verbatim, including all filler words and disfluencies.", "VERBATIM"),
    "lex": ("Output must be in lexical format.", "LEXICAL"),
    "display": ("Output must be in display format.", "READABLE"),
}


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


def get_task_prompt(style="verb", language=None, *, lang_hint=True):
    """Build a complete ASR prompt with its audio marker and optional language hint."""
    if style not in STYLES:
        raise ValueError(f"Unknown style: {style!r}; expected verb, lex, or display")
    prompt = f"Transcribe the audio clip into text. {STYLES[style][0]}"
    languages = resolve_language(language).split() if language else []
    return add_language_hint(f"{prompt}<audio>", languages if lang_hint else [])


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

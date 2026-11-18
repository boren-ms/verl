# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Task prompts and JSONL targets for ASR."""

import json
import random

from recipe.phimm.utils.languages import LANGUAGES, get_language_name

STYLES = {
    "verb": "Transcribe verbatim, including all filler words and disfluencies.",
    "lex": "Output must be in lexical format.",
    "display": "Output must be in display format.",
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
    prompt = f"Transcribe the audio clip into text. {STYLES[style]}"
    languages = resolve_language(language).split() if language else []
    prompt = add_language_hint(f"{prompt}<audio>", languages if lang_hint else [])
    return (
        prompt + '\nReturn JSONL only: one {"language":"target","text":"transcription"} object per segment, '
        "one object per line. Use the target language name and the segment transcript."
    )


def get_task_output(style="verb", language=None, text="", components=None):
    """Write one JSON line per prepared segment."""
    if style not in STYLES:
        raise ValueError(f"Unknown style: {style!r}; expected verb, lex, or display")
    segments = components or [{"text": text}]
    return "\n".join(
        json.dumps(
            {
                "language": resolve_language(segment.get("target_language") or segment.get("language") or language),
                "text": segment.get("text") or "",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for segment in segments
    )

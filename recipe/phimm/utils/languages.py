import re

LANGUAGES = {
    "english": "en",
    "chinese": "zh",
    "cantonese": "yue",
    "japanese": "ja",
    "korean": "ko",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "portuguese": "pt",
    "italian": "it",
    "dutch": "nl",
    "russian": "ru",
    "polish": "pl",
    "czech": "cs",
    "turkish": "tr",
    "arabic": "ar",
    "hindi": "hi",
    "thai": "th",
    "vietnamese": "vi",
    "indonesian": "id",
    "malay": "ms",
    "tagalog": "tl",
    "swedish": "sv",
    "danish": "da",
    "norwegian": "no",
    "finnish": "fi",
    "hungarian": "hu",
    "romanian": "ro",
    "greek": "el",
    "hebrew": "he",
    "ukrainian": "uk",
    "catalan": "ca",
    "galician": "gl",
    "welsh": "cy",
    "persian": "fa",
    "urdu": "ur",
    "bengali": "bn",
    "tamil": "ta",
    "telugu": "te",
    "marathi": "mr",
    "gujarati": "gu",
    "kannada": "kn",
    "malayalam": "ml",
    "swahili": "sw",
    "afrikaans": "af",
    "slovenian": "sl",
    "slovak": "sk",
    "croatian": "hr",
    "serbian": "sr",
    "bulgarian": "bg",
    "lithuanian": "lt",
    "latvian": "lv",
    "estonian": "et",
    "icelandic": "is",
}

LANG_CODE_TO_NAME = {code: name.capitalize() for name, code in LANGUAGES.items()}
LANG_NAME_MAPPING = {"mandarin": "chinese"}


def _single_language_name(lang):
    lang = LANG_NAME_MAPPING.get(lang, lang)
    if lang in LANGUAGES:
        return lang.capitalize()
    return LANG_CODE_TO_NAME.get(lang, lang)


def get_language_name(lang):
    """Convert any language identifier to its full name. e.g. "de" -> "German", "German" -> "German".

    A null/empty or "unknown" identifier resolves to "Unknown" (e.g. for
    language-detection tasks where the spoken language is not specified).

    Mixed / code-switch identifiers separated by ``_`` or whitespace are resolved
    per component, e.g. "en_zh" -> "English Chinese", "english chinese" -> "English Chinese".
    """
    if not lang:
        return "Unknown"
    lang = lang.lower()
    if lang == "unknown":
        return "Unknown"
    parts = re.split(r"[_\s]+", lang.strip())
    if len(parts) > 1:
        return " ".join(_single_language_name(part) for part in parts if part)
    return _single_language_name(lang)


def get_language_code(lang):
    """Convert any language identifier to its ISO code. e.g. "German" -> "de", "de" -> "de".

    Mixed / code-switch identifiers separated by ``_`` or whitespace are resolved
    per component and rejoined with ``_``, e.g. "English Chinese" -> "en_zh".
    """
    lang = lang.lower()
    parts = re.split(r"[_\s]+", lang.strip())
    if len(parts) > 1:
        return "_".join(_single_language_code(part) for part in parts if part)
    return _single_language_code(lang)


def _single_language_code(lang):
    if lang in LANG_CODE_TO_NAME:
        return lang
    lang = LANG_NAME_MAPPING.get(lang, lang)
    return LANGUAGES.get(lang, lang)

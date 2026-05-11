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


def get_language_name(lang):
    """Convert any language identifier to its full name. e.g. "de" -> "German", "German" -> "German"."""
    lang = lang.lower()
    lang = LANG_NAME_MAPPING.get(lang, lang)
    if lang in LANGUAGES:
        return lang.capitalize()
    return LANG_CODE_TO_NAME.get(lang, lang)


def get_language_code(lang):
    """Convert any language identifier to its ISO code. e.g. "German" -> "de", "de" -> "de"."""
    lang = lang.lower()
    if lang in LANG_CODE_TO_NAME:
        return lang
    lang = LANG_NAME_MAPPING.get(lang, lang)
    return LANGUAGES.get(lang, lang)

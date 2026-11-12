"""New multilingual rows: normalizer from MOE 18143a6e; legacy scorers stay stable."""
import re
import unicodedata
import regex
import num2words
from .normalizer import remove_non_speech_annotations, remove_symbols_and_diacritics

FILLER_WORDS = {}

def remove_symbols_keep_marks(s: str):
    """Replace symbols and punctuation with spaces while preserving combining marks."""
    return "".join(" " if unicodedata.category(c)[0] in "SP" else c for c in unicodedata.normalize("NFKC", s))


class BasicMultilingualTextNormalizer:
    def __init__(self, remove_diacritics: bool = True):
        self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols_keep_marks

    def __call__(self, s: str):
        s = s.lower()
        s = remove_non_speech_annotations(s)
        s = self.clean(s).lower()

        # Remove punctuation and extra spaces.
        s = regex.sub(r"[^\w\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()

        return s


class MultilingualNormalizer(BasicMultilingualTextNormalizer):
    """BasicMultilingualTextNormalizer with optional number normalization.

    Call with just text for standard normalization (backward-compatible).
    Pass lang= to also convert digits to words via num2words and remove
    language-specific filler words from FILLER_WORDS.
    """

    def __init__(self, remove_diacritics: bool = True):
        super().__init__(remove_diacritics)
        self._language_normalizers = {
        }
        self._filler_patterns = {}
        base_normalize = super().__call__
        for lang, words in FILLER_WORDS.items():
            normalized_words = {base_normalize(word) for word in words}
            normalized_words.discard("")
            self._filler_patterns[lang] = re.compile(
                r"(?<!\S)(?:"
                + "|".join(re.escape(word) for word in sorted(normalized_words, key=len, reverse=True))
                + r")(?!\S)"
            )

    def _remove_fillers(self, text, lang):
        pattern = self._filler_patterns.get(lang)
        if pattern is None:
            return text
        return re.sub(r"\s+", " ", pattern.sub("", text)).strip()

    def _normalize_numbers(self, text, lang):
        # Join space-separated thousand groups (e.g. "10 000" -> "10000")
        text = re.sub(r"(\d)\s+(\d{3})\b", r"\1\2", text)
        # Convert remaining digit sequences to words
        def _replace(m):
            try:
                return num2words.num2words(int(m.group()), lang=lang)
            except Exception:
                return m.group()
        return re.sub(r"\d+", _replace, text)

    @staticmethod
    def _normalize_vi_numbers(text):
        text = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", text)

        def _decimal(match):
            fraction = match.group(2)
            zeros = len(fraction) - len(fraction.lstrip("0"))
            parts = [
                match.group(1),
                "phẩy",
                *(["không"] * zeros),
            ]
            if fraction[zeros:]:
                parts.append(fraction[zeros:])
            return " ".join(parts)

        text = re.sub(r"(\d+),(\d+)", _decimal, text)
        return text.replace("%", " phần trăm")

    _VI_TONE_MOVES = [
        ("o", "àáảãạ", "òóỏõọ", "a"),
        ("o", "èéẻẽẹ", "òóỏõọ", "e"),
        ("u", "ỳýỷỹỵ", "ùúủũụ", "y"),
    ]
    _VI_TONE_PATTERNS = [
        (
            re.compile(
                r"(?<!q)"
                + first
                + "(["
                + marked
                + r"])(?!\w)"
            ),
            dict(zip(marked, moved)),
            second,
        )
        for first, marked, moved, second in _VI_TONE_MOVES
    ]

    @classmethod
    def _normalize_vi_tone_placement(cls, text):
        for pattern, moved, second in cls._VI_TONE_PATTERNS:
            text = pattern.sub(
                lambda match: moved[match.group(1)] + second,
                text,
            )
        return text

    def __call__(self, s, lang=None):
        language_normalizer = self._language_normalizers.get(lang)
        if language_normalizer is not None:
            return language_normalizer(s)

        if lang == "vi":
            s = self._normalize_vi_numbers(s)
        s = super().__call__(s)
        if lang == "vi":
            s = self._normalize_vi_tone_placement(s)
        if lang is not None:
            s = self._remove_fillers(s, lang)
            s = self._normalize_numbers(s, lang)
        return s

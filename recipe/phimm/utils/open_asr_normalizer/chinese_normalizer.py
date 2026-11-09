import re
import unicodedata

import regex

from .expansion_normalizer import remove_symbols_keep_marks


VULGAR_FRACTION_RE = re.compile(r"(\d?)([¼-¾⅐-⅞])")
BRACKETS_RE = re.compile(r"[<\[][^>\]]*[>\]]")
SPACE_NEXT_TO_HAN_RE = regex.compile(
    r"(?<=\p{Han})\s+|\s+(?=\p{Han})"
)


def _split_vulgar_fraction(match):
    fraction = unicodedata.normalize(
        "NFKC",
        match.group(2),
    ).replace("⁄", "/")
    if match.group(1):
        return f"{match.group(1)} {fraction}"
    return fraction


class ChineseTextNormalizer:
    """Normalize Mandarin references and hypotheses for character scoring."""

    def __init__(self):
        self._t2s = None
        self._number_normalizer = None

    def _load(self):
        import opencc
        from wetext import Normalizer

        self._t2s = opencc.OpenCC("t2s")
        self._number_normalizer = Normalizer(
            lang="zh",
            operator="tn",
        )

    def __call__(self, text):
        if self._number_normalizer is None:
            self._load()
        text = VULGAR_FRACTION_RE.sub(
            _split_vulgar_fraction,
            text,
        )
        text = unicodedata.normalize("NFKC", text)
        text = self._t2s.convert(text)
        text = text.lower()
        text = BRACKETS_RE.sub("", text)
        text = re.sub(r"[()]", " ", text)
        text = SPACE_NEXT_TO_HAN_RE.sub("", text)
        text = self._number_normalizer.normalize(text)
        text = text.replace("〇", "零")
        text = remove_symbols_keep_marks(text)
        text = regex.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

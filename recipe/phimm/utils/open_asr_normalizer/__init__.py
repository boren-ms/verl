from .normalizer import EnglishTextNormalizer, BasicMultilingualTextNormalizer


def __getattr__(name):
    if name == "MultilingualNormalizer":
        from .data_utils import MultilingualNormalizer
        return MultilingualNormalizer
    if name == "TextNorm":
        from .cn_tn import TextNorm
        return TextNorm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
# %%
from whisper_normalizer.english import EnglishTextNormalizer
from whisper_normalizer.basic import BasicTextNormalizer
import jiwer.transforms as tr


class RemovePunctuationExclude(tr.RemovePunctuation):
    """RemovePunctuation excluding certain characters."""

    def __init__(self, exclude=None):
        super().__init__()
        self.exclude = exclude or []
        self.tokens_to_remove = [x for x in self.tokens_to_remove if x not in self.exclude]
        # print(f"tokens_to_remove: {self.tokens_to_remove}")


def identity(text):
    """Identity normalization function."""
    return text


def lower(text):
    """Lowercase normalization function."""
    return text.lower()


def simple_with_tag(text):
    """Simple normalization function."""
    norm = tr.Compose(
        [
            tr.ToLowerCase(),
            # tr.ExpandCommonEnglishContractions(),
            # tr.RemovePunctuation(),
            RemovePunctuationExclude(exclude=["*"]),
            tr.RemoveWhiteSpace(replace_by_space=True),
            tr.RemoveMultipleSpaces(),
            tr.Strip(),
            tr.ReduceToSingleSentence(),
            # tr.ReduceToListOfListOfWords(),
        ]
    )
    return norm(text)


def simple(text):
    """Simple normalization function."""
    norm = tr.Compose(
        [
            tr.ToLowerCase(),
            tr.RemovePunctuation(),
            tr.RemoveWhiteSpace(replace_by_space=True),
            tr.RemoveMultipleSpaces(),
            tr.Strip(),
            tr.ReduceToSingleSentence(),
        ]
    )
    return norm(text)


TN_DICT = {
    "english": EnglishTextNormalizer(),
    "identity": identity,
    "lower": lower,
    "basic": BasicTextNormalizer(),
    "simple": simple,
    "simple_with_tag": simple_with_tag,
}


def text_norm(txt, name=None):
    """Normalize tokens by removing leading and trailing whitespace."""
    name = name or "english"  # Default to EnglishTextNormalizer
    norm = TN_DICT[name]
    if isinstance(txt, str):
        return norm(txt.strip())
    elif isinstance(txt, (list, tuple)):
        return [norm(x) for x in txt]
    else:
        raise ValueError(f"Unsupported type for text normalization: {type(txt)}. Expected str or list of str.")

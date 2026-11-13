# %%
from whisper_normalizer.english import EnglishTextNormalizer
from whisper_normalizer.basic import BasicTextNormalizer
import jiwer.transforms as tr
from recipe.phimm.utils.shared import is_list
from recipe.phimm.utils.open_asr_normalizer import EnglishTextNormalizer as OpenASREnglishNormalizer
from recipe.phimm.utils.open_asr_normalizer import BasicMultilingualTextNormalizer as OpenASRMultilingualNormalizer
from recipe.phimm.utils.open_asr_normalizer import PortugueseNormalizer as OpenASRPortugueseNormalizer
from recipe.phimm.utils.open_asr_normalizer.hf_english_normalizer import _HFEnglishTextNormalizer


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


def voi_hi(text):
    """Use the exact voi-oiwer Hindi evaluation preprocessing, preserving vowel marks.

    This normalizes a single text, not a reference lattice. Selecting
    ``text_norm: openasr_ml`` explicitly retains the legacy Hindi behavior.
    """
    try:
        from voi_oiwer import standardize_hypothesis
    except ImportError as exc:
        raise ImportError("Install voi-oiwer==0.1.4 to normalize Hindi training rewards.") from exc

    return " ".join(standardize_hypothesis(text, "hi"))


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
    "openasr_en": OpenASREnglishNormalizer(),
    "hf_english": _HFEnglishTextNormalizer(),
    "openasr_ml": OpenASRMultilingualNormalizer(),
    "voi_hi": voi_hi,
    "openasr_pt": OpenASRPortugueseNormalizer(remove_diacritics=False),
}


def default_tn_name(lang="english"):
    """Use evaluation-equivalent Hindi normalization; retain other language defaults."""
    lang = (lang or "english").lower().strip()
    if lang in ("hindi", "hi", "hi_in", "hi-in"):
        return "voi_hi"
    return "openasr_en" if lang == "english" else "openasr_ml"


def text_norm(txt, name=None):
    """Normalize tokens by removing leading and trailing whitespace."""
    name = name or "english"  # Default to EnglishTextNormalizer
    norm = TN_DICT[name]
    if isinstance(txt, str):
        return norm(txt.strip())
    elif is_list(txt):
        return [norm(x) for x in txt]
    else:
        raise ValueError(f"Unsupported type for text normalization: {type(txt)}. Expected str or list of str.")

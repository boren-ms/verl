import re

from recipe.phimm.reward.asr_response import get_hyp_text
from recipe.phimm.utils.languages import get_language_code
from recipe.phimm.utils.open_asr_normalizer.normalizer import (
    EnglishTextNormalizer,
    remove_symbols_and_diacritics,
)


_HF_ENGLISH_COMPOUNDS = {
    r"\bet\s+cetera\b": "etc",
    r"\bal\s+right\b": "alright",
    r"\ball\s+right\b": "alright",
    r"\bhow\s+ever\b": "however",
    r"\bwi\s+fi\b": "wifi",
    r"\bhi\s+fi\b": "hifi",
    r"\blo\s+fi\b": "lofi",
    r"\bsci\s+fi\b": "scifi",
    r"\be\s+mail\b": "email",
    r"\be\s+book\b": "ebook",
    r"\be\s+commerce\b": "ecommerce",
    r"\bx\s+ray\b": "xray",
    r"\bt\s+shirt\b": "tshirt",
    r"\ba\s+m\b": "am",
    r"\bp\s+m\b": "pm",
    r"\bo\s+k\b": "okay",
}


class _HFEnglishTextNormalizer(EnglishTextNormalizer):
    def __init__(self):
        super().__init__()
        self.ignore_patterns = (
            r"\b(hmm|mm|mhm|mmm|uh|um|ah|aha|ahh|ahm|eh|ehehe|em|hm|huh|hum|mhum|uhm|umm|uhuh)\b"
        )
        self.replacers.pop(r"'s\b")
        self.replacers[r"\b(it|he|she|what|that|who|here|there|how|when|where|why|this)'s\b"] = r"\1 is"

    @staticmethod
    def _normalize_acronyms(text):
        words = text.split()
        result = []
        i = 0
        while i < len(words):
            if len(words[i]) == 1 and words[i].isalnum():
                run = [words[i]]
                j = i + 1
                while j < len(words) and len(words[j]) == 1 and words[j].isalnum():
                    run.append(words[j])
                    j += 1
                has_common_word = any(char in ("a", "i") for char in run)
                if len(run) >= (3 if has_common_word else 2):
                    result.append("".join(run))
                else:
                    result.extend(run)
                i = j
            else:
                result.append(words[i])
                i += 1
        return " ".join(result)

    def __call__(self, text):
        text = text.lower()
        text = re.sub(r"[<\[][^>\]]*[>\]]", "", text)
        text = re.sub(r"\(([^)]+?)\)", "", text)
        text = re.sub(self.ignore_patterns, "", text)
        text = re.sub(r"\s+'", "'", text)
        for pattern, replacement in self.replacers.items():
            text = re.sub(pattern, replacement, text)
        text = re.sub(r"(\d),(\d)", r"\1\2", text)
        text = re.sub(r"\.([^0-9]|$)", r" \1", text)
        text = remove_symbols_and_diacritics(text, keep=".%$¢€£")
        for pattern, replacement in _HF_ENGLISH_COMPOUNDS.items():
            text = re.sub(pattern, replacement, text)
        text = self.standardize_numbers(text)
        text = self.standardize_spellings(text)
        text = self._normalize_acronyms(text)
        text = re.sub(r"[.$¢€£]([^0-9])", r" \1", text)
        text = re.sub(r"([^0-9])%", r"\1 ", text)
        return re.sub(r"\s+", " ", text)


_hf_english_normalizer = _HFEnglishTextNormalizer()


def openasr_eval(solution_str, ground_truth, **kwargs):
    """Evaluate a response using OpenASR normalization and compound-aware WER."""
    from recipe.phimm.utils.open_asr_normalizer.eval_utils import measure_wer

    extra_info = kwargs.get("extra_info") or {}
    tgt_lang = extra_info.get("language", kwargs.get("language", "English")).lower().strip()
    hyp_text = get_hyp_text(solution_str, version=kwargs.get("version"))
    result = measure_wer(
        hyp_text,
        ground_truth,
        lang=get_language_code(tgt_lang),
        merge_compounds=kwargs.get("merge_compounds", True),
    )

    return {
        "score": 1.0 - result["wer"],
        "wer": result["wer"],
        "n_err": result["n_err"],
        "n_ref": result["n_ref"],
    }


def openasr_en_eval(solution_str, ground_truth, **kwargs):
    """Evaluate English OpenASR responses with HF compound-aware WER."""
    from kaldialign import batch_error_rate

    hyp_text = get_hyp_text(solution_str, version=kwargs.get("version"))
    ref_words = tuple(_hf_english_normalizer(ground_truth.strip()).split())
    hyp_words = tuple(_hf_english_normalizer(hyp_text.strip()).split())
    result = batch_error_rate([ref_words], [hyp_words], merge_compounds=True)
    n_ref = max(result["ref_len"], 1)
    wer = result["total"] / n_ref
    return {
        "score": 1.0 - wer,
        "wer": wer,
        "n_err": result["total"],
        "n_ref": n_ref,
    }
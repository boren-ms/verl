# %%
"""OpenASR-style WER reward — calls their normalizers + jiwer.wer directly."""
import re
from difflib import SequenceMatcher

from jiwer import wer as jiwer_wer

from recipe.phimm.utils.shared import parse_asr_response
from recipe.phimm.reward.openasr_normalizer import (
    EnglishTextNormalizer,
    BasicMultilingualTextNormalizer,
)

_en_normalizer = EnglishTextNormalizer()


class MultilingualNormalizer(BasicMultilingualTextNormalizer):
    def _normalize_numbers(self, text, lang):
        import num2words
        text = re.sub(r"(\d)\s+(\d{3})\b", r"\1\2", text)
        def _replace(m):
            try:
                return num2words.num2words(int(m.group()), lang=lang)
            except Exception:
                return m.group()
        return re.sub(r"\d+", _replace, text)

    def __call__(self, s, lang=None):
        s = super().__call__(s)
        if lang is not None:
            s = self._normalize_numbers(s, lang)
        return s


_ml_normalizer = MultilingualNormalizer(remove_diacritics=False)


def _normalize_compound_pairs(ref_norm, hyp_norm):
    ref_words, hyp_words = ref_norm.split(), hyp_norm.split()
    sm = SequenceMatcher(None, ref_words, hyp_words)
    new_rw, new_hw = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            new_rw.extend(ref_words[i1:i2])
            new_hw.extend(hyp_words[j1:j2])
        else:
            rc = "".join(ref_words[i1:i2])
            hc = "".join(hyp_words[j1:j2])
            if rc == hc:
                new_rw.append(rc)
                new_hw.append(hc)
            else:
                new_rw.extend(ref_words[i1:i2])
                new_hw.extend(hyp_words[j1:j2])
    return " ".join(new_rw), " ".join(new_hw)


def measure(hyp, ref, **kwargs):
    """Return dict with wer, n_err, n_ref using OpenASR normalizers + jiwer."""
    multilingual = kwargs.get("multilingual", False)
    lang = kwargs.get("lang", None)

    ref_norm = (_ml_normalizer(ref.strip(), lang=lang) if multilingual
                else _en_normalizer(ref.strip()))
    hyp_norm = (_ml_normalizer(hyp.strip(), lang=lang) if multilingual
                else _en_normalizer(hyp.strip()))

    if multilingual:
        ref_norm, hyp_norm = _normalize_compound_pairs(ref_norm, hyp_norm)

    n_ref = len(ref_norm.split())
    if not ref_norm.strip():
        wer_val = 1.0 if hyp_norm.strip() else 0.0
        return {"wer": wer_val, "n_err": n_ref, "n_ref": max(n_ref, 1)}

    wer_val = jiwer_wer(ref_norm, hyp_norm)
    n_err = round(wer_val * n_ref)
    return {"wer": wer_val, "n_err": n_err, "n_ref": n_ref}


def compute_score(solution_str, ground_truth, **kwargs):
    solution_str = parse_asr_response(solution_str)["text"]
    result = measure(solution_str, ground_truth, **kwargs)
    return {"score": result["wer"], **result}


def eval_score(solution_str, ground_truth, **kwargs):
    solution_str = parse_asr_response(solution_str)["text"]
    result = measure(solution_str, ground_truth, **kwargs)
    return {**result}


# %%
if __name__ == "__main__":
    pairs = [
        ("turn on the living room lights", "turn on the living room lights"),
        ("play jazz music now", "play some jazz music now"),
        ("I won't go there", "I will not go there"),
    ]
    for ref, hyp in pairs:
        r = measure(hyp, ref)
        print(f"ref={ref!r}  hyp={hyp!r}  wer={r['wer']:.2%}  n_err={r['n_err']}  n_ref={r['n_ref']}")

# %%

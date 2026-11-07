"""MOE 18143a6e scoring for supplementary Mandarin/Vietnamese/French rows."""

import regex

from recipe.phimm.reward.asr_response import get_hyp_text
from recipe.phimm.utils.languages import get_language_code
from recipe.phimm.utils.open_asr_normalizer.chinese_normalizer import ChineseTextNormalizer
from recipe.phimm.utils.open_asr_normalizer.expansion_normalizer import MultilingualNormalizer
from recipe.phimm.utils.open_asr_normalizer.eval_utils import normalize_compound_pairs

_zh = ChineseTextNormalizer()
_ml = MultilingualNormalizer(remove_diacritics=False)


def measure_expansion(hyp, ref, lang):
    from kaldialign import batch_error_rate

    if lang == "zh":
        ref_norm, hyp_norm = _zh(ref), _zh(hyp)
        refs = tuple(regex.findall(r"\X", "".join(ref_norm.split())))
        hyps = tuple(regex.findall(r"\X", "".join(hyp_norm.split())))
        metric = "cer"
    elif lang in {"vi", "fr"}:
        ref_norm, hyp_norm = _ml(ref, lang=lang), _ml(hyp, lang=lang)
        [ref_norm], [hyp_norm] = normalize_compound_pairs([ref_norm], [hyp_norm])
        refs, hyps = tuple(ref_norm.split()), tuple(hyp_norm.split())
        metric = "wer"
    else:
        raise ValueError(f"Unsupported expansion language: {lang}")
    counts = batch_error_rate([refs], [hyps], merge_compounds=False)
    n_ref = counts["ref_len"]
    rate = counts["total"] / n_ref if n_ref else float(bool(hyps))
    return {
        "score": 1.0 - rate,
        metric: rate,
        "n_err": counts["total"],
        "n_ref": n_ref,
        "n_ins": counts["ins"],
        "n_del": counts["del"],
        "n_sub": counts["sub"],
    }


def openasr_expansion_eval(solution_str, ground_truth, **kwargs):
    extra = kwargs.get("extra_info") or {}
    language = extra.get("language", kwargs.get("language", ""))
    lang = get_language_code(language.lower().strip())
    hyp = get_hyp_text(solution_str, version=kwargs.get("version"))
    return measure_expansion(hyp, ground_truth, lang)

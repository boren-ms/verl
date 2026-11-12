"""Mining-only scorer for the five OSS increments frozen on 2026-11-12.

Use the existing OpenASR normalization and compound alignment for WER;
Mandarin/Cantonese use the existing Chinese normalizer and grapheme CER.
Language identity comes from the source ISO code, not English display names.
This module does not alter any training reward or the older OSS mining run.
"""
import regex

from recipe.phimm.reward.asr_response import get_hyp_text
from recipe.phimm.utils.open_asr_normalizer.chinese_normalizer import ChineseTextNormalizer
from recipe.phimm.utils.open_asr_normalizer.eval_utils import normalize_for_wer

SCORER_ID = "oss_gap_error_mining_v1"
CER_LANGUAGES = {"zh", "yue"}
_zh = ChineseTextNormalizer()


def compute_score(solution_str, ground_truth, **kwargs):
    from kaldialign import batch_error_rate

    extra = kwargs.get("extra_info") or {}
    lang = extra.get("source_language")
    if not isinstance(lang, str) or not regex.fullmatch(r"[a-z]{2,3}", lang):
        raise ValueError(f"Missing frozen ISO source_language: {lang!r}")
    hyp = get_hyp_text(solution_str, version=kwargs.get("version"))
    if lang in CER_LANGUAGES:
        ref_norm, hyp_norm = _zh(ground_truth), _zh(hyp)
        refs = tuple(regex.findall(r"\X", "".join(ref_norm.split())))
        hyps = tuple(regex.findall(r"\X", "".join(hyp_norm.split())))
        metric, unit = "cer", "character"
    else:
        hyp_norm, ref_norm = normalize_for_wer(hyp, ground_truth, lang=lang, merge_compounds=True)
        refs, hyps = tuple(ref_norm.split()), tuple(hyp_norm.split())
        metric, unit = "wer", "word"
    counts = batch_error_rate([refs], [hyps], merge_compounds=False)
    n_ref = counts["ref_len"]
    rate = counts["total"] / n_ref if n_ref else float(bool(hyps))
    return {"score": 1.0 - rate, metric: rate, "n_err": counts["total"], "n_ref": n_ref,
            "n_ins": counts["ins"], "n_del": counts["del"], "n_sub": counts["sub"],
            "selection_metric": metric, "selection_unit": unit, "selection_scorer": SCORER_ID}

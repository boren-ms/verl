"""Pinned mining scorer only. Does not change the Base2c training rewards.

Mandarin uses the existing expansion CER. Other languages use the established
OpenASR WER path. Global ranking is explicitly absolute n_err, so Chinese
character counts and other-language word counts are different units.
"""
from recipe.phimm.reward.asr_eval import openasr_eval
from recipe.phimm.reward.openasr_expansion import openasr_expansion_eval
from recipe.phimm.utils.languages import get_language_code

SUPPORTED = set("ar cs da de en es fi fr he hi hu it ja ko nl pl pt ru sv th tr zh".split())


def compute_score(solution_str, ground_truth, **kwargs):
    extra = kwargs.get("extra_info") or {}
    language = get_language_code(extra.get("language", "").lower().strip())
    if language not in SUPPORTED:
        raise ValueError(f"Unfrozen mining language: {language!r}")
    scorer = openasr_expansion_eval if language == "zh" else openasr_eval
    result = scorer(solution_str, ground_truth, **kwargs)
    return {**result, "selection_metric": "cer" if language == "zh" else "wer",
            "selection_unit": "character" if language == "zh" else "word",
            "selection_scorer": "oss_error_mining_v1"}

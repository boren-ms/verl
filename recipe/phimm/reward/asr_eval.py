from recipe.phimm.reward.asr_response import get_hyp_text
from recipe.phimm.utils.languages import get_language_code
from recipe.phimm.utils.open_asr_normalizer.hf_english_normalizer import (
    _HFEnglishTextNormalizer,
)


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
from recipe.phimm.reward.asr_response import get_hyp_text
from recipe.phimm.utils.languages import get_language_code


def openasr_eval(solution_str, ground_truth, **kwargs):
    """Evaluate a response using OpenASR normalization and edit-distance WER."""
    from recipe.phimm.utils.open_asr_normalizer.eval_utils import measure_wer

    extra_info = kwargs.get("extra_info") or {}
    tgt_lang = extra_info.get("language", kwargs.get("language", "English")).lower().strip()
    hyp_text = get_hyp_text(solution_str, version=kwargs.get("version"))
    result = measure_wer(hyp_text, ground_truth, lang=get_language_code(tgt_lang))

    return {
        "score": 1.0 - result["wer"],
        "wer": result["wer"],
        "n_err": result["n_err"],
        "n_ref": result["n_ref"],
    }
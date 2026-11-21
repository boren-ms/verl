"""Bounded transcription/diarization reward for native VibeVoice responses."""

import math

from vibevoice_asr.transcription import NON_SPEECH_TAGS, parse_transcription, transcription_text

# Unintelligible speech remains speech for diarization, as in the paper audit.
NON_SPEECH_EVENTS = NON_SPEECH_TAGS - {"[unintelligible speech]"}


def one_response(response):
    if isinstance(response, list):
        if len(response) != 1:
            raise ValueError("VibeVoice RL requires one native response per audio clip")
        response = response[0]
    if not isinstance(response, str):
        raise ValueError("Response must be a string")
    return response


def native_prediction(response):
    segments = parse_transcription(one_response(response), allow_invalid_event_times=True)
    activity = [
        {"speaker": item["speaker"], "start": item["start_s"], "end": item["end_s"]}
        for item in segments
        if item["text"].strip().lower() not in NON_SPEECH_EVENTS
    ]
    words = [
        {"speaker": item["speaker"], "text": item["text"]}
        for item in segments
        if item["text"].strip().lower() not in NON_SPEECH_TAGS
    ]
    return activity, words, transcription_text(segments)


def transcript_text(response):
    try:
        return native_prediction(response)[2]
    except (ValueError, TypeError, KeyError):
        return ""


def diarization_reward(solution_str, ground_truth, *, extra_info, der_weight=0.5):
    """Use zero-collar, overlap-inclusive DER; retain unclipped metric counts.

    Reward = 1 - w*min(DER,1) - (1-w)*min(WER,1). Invalid output gets zero.
    Only the optimization reward is bounded, never the reported DER or WER.
    No delimiter repairs, missing-speaker invention or reference-derived output.
    """
    from recipe.phimm.reward.asr_eval import measure_openasr_en_wer
    from recipe.phimm.reward.diarization_eval import measure_der, measure_wder

    if not isinstance(der_weight, (int, float)) or not math.isfinite(der_weight) or not 0 <= der_weight <= 1:
        raise ValueError("der_weight must be between zero and one")
    duration = extra_info["evaluation_duration_s"]
    reference = extra_info["reference_segments"]
    # Dataset defects must raise independently of prediction validity.
    missed = measure_der(reference, [], evaluation_duration_s=duration)
    reference_words = extra_info.get("reference_words")
    result = {"score": 0.0, "n_json_valid": 0, "n_json_total": 1, "der_unavailable": 0}
    try:
        activity, words, hypothesis = native_prediction(solution_str)
        der = measure_der(reference, activity, evaluation_duration_s=duration)
    except (ValueError, TypeError, KeyError, OverflowError):
        # A failed prediction counts as missed reference speech in pooled DER;
        # count the failure separately so it cannot shrink evaluation coverage.
        der = missed
        hypothesis, words = "", []
        result["invalid_output"] = 1
    else:
        result.update(n_json_valid=1, invalid_output=0)
    wer = measure_openasr_en_wer(hypothesis, ground_truth)
    result.update(der)
    result.update({key: wer[key] for key in ("wer", "n_err", "n_ref")})
    if reference_words:
        result.update(measure_wder(reference_words, words))
        result["wder_unavailable"] = 0
    else:
        result.update(wder=None, wder_n_err=0, wder_n_ref=0, wder_unavailable=1)
    if result["n_json_valid"] and der["der"] is not None:
        result["score"] = 1 - der_weight * min(der["der"], 1) - (1 - der_weight) * min(wer["wer"], 1)
    return result

"""Recording-level DER/WDER rewards using the same interface as ``asr_eval``.

References live in extra_info on the JSONL row. Responses may be speaker JSONL
(text, optional spk and start/end in 10 ms ticks), or legacy JSON objects with
``segments`` (speaker/start_s/end_s) and/or ``words`` (speaker/text). With SVAD
audio lists, ``svad_bounds`` maps each response's local clock back to the original
recording. Speaker IDs must stay consistent across responses: no independent
per-segment label remapping is performed. ``prediction_format=text`` instead
scores a single-speaker baseline: every nonempty decoded segment receives the
same speaker label and the whole decoding interval. No acoustic diarizer or
word-timing model is used, and references never generate hypothesis labels.
"""

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping

from recipe.phimm.reward.asr_response import clean_text, get_hyp_text, parse_jsonl_response, parse_response_outputs


def _assignment(matrix):
    from scipy.optimize import linear_sum_assignment

    if not matrix or not matrix[0]:
        return []
    rows, columns = linear_sum_assignment(matrix, maximize=True)
    return list(zip(rows, columns, strict=True))


def _speaker(item):
    if not isinstance(item, Mapping) or not isinstance(item.get("speaker"), str) or not item["speaker"].strip():
        raise ValueError("Every annotation needs a nonempty string speaker label")
    return item["speaker"]


def _timeline(segments, duration):
    if not isinstance(segments, list):
        raise ValueError("Segments must be a list")
    spans = defaultdict(list)
    for segment in segments:
        speaker = _speaker(segment)
        start = segment.get("start_s", segment.get("start"))
        end = segment.get("end_s", segment.get("end"))
        if any(isinstance(t, bool) or not isinstance(t, int | float) or not math.isfinite(t) for t in (start, end)):
            raise ValueError("Segment timestamps must be finite numbers")
        if start < 0 or end < start:
            raise ValueError("Segment timestamps must be nonnegative and ordered")
        end = min(end, duration) if duration is not None else end
        if end > start:
            spans[speaker].append((start, end))
    timeline = []
    for speaker, intervals in sorted(spans.items()):
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        timeline.extend((speaker, start, end) for start, end in merged)
    return timeline


def measure_der(reference, hypothesis, *, evaluation_duration_s=None):
    """Exact NIST DER: zero collar, overlap included, optimal speaker mapping.

    Same-speaker overlapping/touching regions are unioned. Prepared DAI reference
    segments already include the upstream 200 ms word-gap merge; native predicted
    segments are used as supplied. Counts are speaker-seconds, not clip seconds.
    """
    duration = evaluation_duration_s
    if duration is not None and (
        isinstance(duration, bool)
        or not isinstance(duration, int | float)
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError("evaluation_duration_s must be finite and positive")
    timelines = [_timeline(segments, duration) for segments in (reference, hypothesis)]
    speakers = [sorted({speaker for speaker, _, _ in timeline}) for timeline in timelines]
    indices = [{speaker: i for i, speaker in enumerate(labels)} for labels in speakers]
    overlap = [[0.0] * len(speakers[1]) for _ in speakers[0]]
    events = defaultdict(list)
    for side, timeline in enumerate(timelines):
        for speaker, start, end in timeline:
            events[start].append((side, speaker, 1))
            events[end].append((side, speaker, -1))
    active = [set(), set()]
    previous = None
    missed = false_alarm = common = ref_time = 0.0
    for boundary in sorted(events):
        if previous is not None:
            seconds = boundary - previous
            nr, nh = map(len, active)
            ref_time += seconds * nr
            missed += seconds * max(0, nr - nh)
            false_alarm += seconds * max(0, nh - nr)
            common += seconds * min(nr, nh)
            for ref in active[0]:
                for hyp in active[1]:
                    overlap[indices[0][ref]][indices[1][hyp]] += seconds
        for side, speaker, delta in events[boundary]:
            if delta == 1:
                active[side].add(speaker)
            else:
                active[side].remove(speaker)
        previous = boundary
    correct = sum(overlap[i][j] for i, j in _assignment(overlap))
    confusion = max(0.0, common - correct)
    errors = missed + false_alarm + confusion
    return {
        "der": errors / ref_time if ref_time else None,
        "der_n_err": errors,
        "der_n_ref": ref_time,
        "der_n_miss": missed,
        "der_n_fa": false_alarm,
        "der_n_confusion": confusion,
    }


def _word_tokens(words, normalization, language):
    if not isinstance(words, list):
        raise ValueError("Words must be a list")
    if normalization == "hf_english":
        from recipe.phimm.reward.asr_eval import _hf_english_normalizer

        if language.lower() not in {"english", "en"}:
            raise ValueError("hf_english WDER requires English; select the appropriate normalization")
        normalize = _hf_english_normalizer
    elif normalization == "dai_asr_i18n":
        from dai_asr_i18n import normalize as dai_normalize

        normalize = lambda text: dai_normalize(text, language)  # noqa: E731
    else:
        raise ValueError(f"Unknown WDER normalization: {normalization!r}")
    tokens = []
    for word in words:
        speaker = _speaker(word)
        if not isinstance(word.get("text"), str):
            raise ValueError("Every word needs string text")
        tokens.extend((token, speaker) for token in normalize(word["text"]).split())
    return tokens


def measure_wder(reference, hypothesis, *, normalization="hf_english", language="English"):
    """Speaker errors / aligned word pairs, including substitutions, excluding I/D.

    Levenshtein alignment precedes optimal one-to-one speaker mapping per recording.
    This supplemental WDER is distinct from WER and David AI's official metrics.
    Normalization is applied per labelled word to preserve speaker attribution.
    """
    from rapidfuzz.distance import Levenshtein

    ref, hyp = [_word_tokens(words, normalization, language) for words in (reference, hypothesis)]
    pairs = Counter()
    for tag, i, end_i, j, end_j in Levenshtein.opcodes([w for w, _ in ref], [w for w, _ in hyp]):
        if tag in {"equal", "replace"}:
            for r, h in zip(ref[i:end_i], hyp[j:end_j], strict=True):
                pairs[r[1], h[1]] += 1
    ref_speakers = sorted({r for r, _ in pairs})
    hyp_speakers = sorted({h for _, h in pairs})
    matrix = [[pairs[r, h] for h in hyp_speakers] for r in ref_speakers]
    aligned = sum(pairs.values())
    errors = aligned - sum(matrix[i][j] for i, j in _assignment(matrix))
    return {"wder": errors / aligned if aligned else None, "wder_n_err": errors, "wder_n_ref": aligned}


def _prediction(solution_str):
    if isinstance(solution_str, list):
        if len(solution_str) != 1:
            raise ValueError("Diarization requires one response per original recording")
        solution_str = solution_str[0]
    if not isinstance(solution_str, str):
        raise ValueError("Diarization response must be a JSON string")
    text = solution_str.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Diarization response must be a JSON object")
    return payload


def _segment_bounds(info):
    bounds = info.get("svad_bounds")
    if bounds is None:
        return None
    if not isinstance(bounds, list) or not bounds:
        raise ValueError("svad_bounds must be a nonempty list of [start, end] pairs")
    previous_end = 0.0
    for pair in bounds:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each SVAD bound must be [start, end]")
        start, end = pair
        if any(isinstance(t, bool) or not isinstance(t, int | float) or not math.isfinite(t) for t in pair):
            raise ValueError("SVAD bounds must be finite numbers")
        if start < previous_end or end <= start:
            raise ValueError("SVAD bounds must be positive, ordered and nonoverlapping")
        duration = info.get("evaluation_duration_s")
        if duration is not None and end > duration + 0.001:
            raise ValueError("SVAD bounds exceed the original recording duration")
        previous_end = end
    return bounds


def _recording_prediction(solution_str, bounds, decode=_prediction):
    if bounds is None:
        if isinstance(solution_str, list):
            if len(solution_str) != 1:
                raise ValueError("Diarization requires one response per original recording without SVAD bounds")
            solution_str = solution_str[0]
        return decode(solution_str)
    responses = solution_str if isinstance(solution_str, list) else [solution_str]
    if len(responses) != len(bounds):
        raise ValueError("SVAD requires exactly one response for each audio_path segment")
    payloads = []
    for response in responses:
        try:
            payloads.append(decode(response))
        except (ValueError, TypeError):
            payloads.append({})
    prediction = {}
    # Each metric requires support on every segment. A words-only response can
    # still yield WDER when another metric is unavailable for this recording.
    try:
        segments = []
        for payload, (start, end) in zip(payloads, bounds, strict=True):
            for speaker, left, right in _timeline(payload.get("segments"), end - start):
                segments.append({"speaker": speaker, "start_s": start + left, "end_s": start + right})
        prediction["segments"] = segments
    except (ValueError, TypeError):
        pass
    if all(isinstance(payload.get("words"), list) for payload in payloads):
        prediction["words"] = [word for payload in payloads for word in payload["words"]]
    return prediction


def _speaker_jsonl_prediction(response):
    """Translate a clip's start/end ticks to local seconds, then score per recording."""
    if not isinstance(response, str):
        raise ValueError("Diarization requires a JSONL response string")
    if not response.strip():
        return {"segments": [], "words": []}
    lines = parse_jsonl_response(response)
    if any(line.get("format") != "speaker_jsonl" for line in lines):
        raise ValueError("Expected speaker JSONL")
    segments, words = [], []
    der_available = wder_available = True
    for line in lines:
        speaker = str(line["spk"]) if "spk" in line else None
        text = clean_text(line["text"])
        if text:
            if speaker is None:
                wder_available = False
            else:
                words.extend({"speaker": speaker, "text": word} for word in text.split())
        if speaker is not None and "start" in line:
            segments.append({"speaker": speaker, "start_s": line["start"] / 100, "end_s": line["end"] / 100})
        elif text or "start" in line or speaker is not None:
            der_available = False
    result = {}
    if der_available:
        result["segments"] = segments
    if wder_available:
        result["words"] = words
    return result


def _response_format(solution_str, requested):
    if requested not in {"auto", "structured"}:
        return requested
    try:
        outputs = parse_response_outputs(solution_str)
    except (ValueError, TypeError):
        return requested if requested == "structured" else "text"
    if any(line.get("format") in {"speaker_jsonl", "invalid_speaker_jsonl"} for output in outputs for line in output):
        return "speaker_jsonl"
    if requested == "auto":
        responses = solution_str if isinstance(solution_str, list) else [solution_str]
        for response in responses:
            try:
                if {"segments", "words"} & _prediction(response).keys():
                    return "structured"
            except ValueError:
                continue
        return "text"
    return requested


def _text_recording_prediction(solution_str, bounds, info, *, speaker):
    _speaker({"speaker": speaker})
    responses = solution_str if isinstance(solution_str, list) else [solution_str]
    if not responses or any(not isinstance(response, str) for response in responses):
        raise ValueError("Text diarization requires a nonempty ordered list of response strings")
    texts = [get_hyp_text(response) for response in responses]
    if bounds is None:
        if "seg_start" in info and "seg_end" in info:
            bounds = _segment_bounds({**info, "svad_bounds": [[info["seg_start"], info["seg_end"]]]})
        elif len(texts) == 1 and info.get("evaluation_duration_s") is not None:
            bounds = [[0.0, info["evaluation_duration_s"]]]
    if bounds is not None and len(texts) != len(bounds):
        raise ValueError("Each decoded text segment must have exactly one decoding interval")
    prediction = {"words": [{"speaker": speaker, "text": word} for text in texts for word in text.split()]}
    if bounds is not None:
        prediction["segments"] = [
            {"speaker": speaker, "start_s": start, "end_s": end}
            for text, (start, end) in zip(texts, bounds, strict=True)
            if text.strip()
        ]
    return prediction


def diarization_eval(solution_str: str | list[str], ground_truth=None, **kwargs):
    """Score both metrics; reward is 1-DER or 1-WDER via ``reward_metric``.

    Missing/invalid model fields yield an unavailable metric and zero reward if
    selected. Malformed references raise, so dataset errors cannot look like model
    errors. Explicit empty lists are valid predictions. Undefined zero-denominator
    rates retain primitive counts for pooled corpus scoring. Rates are fractions.
    Text mode is an explicit single-speaker baseline using decoding intervals,
    not speaker recognition. SVAD gaps and within-segment silence are scored
    against the unchanged full reference timeline.
    """
    metric = kwargs.get("reward_metric", "der")
    if metric not in {"der", "wder"}:
        raise ValueError("reward_metric must be der or wder")
    prediction_format = kwargs.get("prediction_format", "structured")
    if prediction_format not in {"auto", "structured", "text", "speaker_jsonl"}:
        raise ValueError("prediction_format must be auto, structured, text or speaker_jsonl")
    prediction_format = _response_format(solution_str, prediction_format)
    speaker = kwargs.get("text_speaker", "speaker_0")
    if prediction_format == "text":
        _speaker({"speaker": speaker})
    info = kwargs.get("extra_info") or {}
    duration = info.get("evaluation_duration_s")
    normalization = kwargs.get("normalization", "hf_english")
    language = info.get("language", "English")
    # Validate references independently of the model's output.
    measure_der(info.get("reference_segments"), [], evaluation_duration_s=duration)
    _word_tokens(info.get("reference_words"), normalization, language)
    bounds = _segment_bounds(info)
    try:
        if prediction_format == "text":
            prediction = _text_recording_prediction(solution_str, bounds, info, speaker=speaker)
        elif prediction_format == "speaker_jsonl":
            prediction = _recording_prediction(solution_str, bounds, _speaker_jsonl_prediction)
        else:
            prediction = _recording_prediction(solution_str, bounds)
    except (ValueError, TypeError, OverflowError):
        prediction = {}
    result = {"der_n_miss": None, "der_n_fa": None, "der_n_confusion": None}
    for name, field, scorer in (
        ("der", "segments", lambda pred: measure_der(info["reference_segments"], pred, evaluation_duration_s=duration)),
        (
            "wder",
            "words",
            lambda pred: measure_wder(info["reference_words"], pred, normalization=normalization, language=language),
        ),
    ):
        try:
            measured = scorer(prediction.get(field))
        except (ValueError, TypeError):
            result.update({name: None, name + "_n_err": None, name + "_n_ref": None, name + "_unavailable": 1})
        else:
            result.update(measured)
            result[name + "_unavailable"] = 0
    rate = result[metric]
    result["score"] = 1.0 - rate if rate is not None else 0.0
    result["single_speaker_baseline"] = int(prediction_format == "text")
    if kwargs.get("compute_wer", False):
        from recipe.phimm.reward.asr_eval import measure_openasr_en_wer

        if prediction_format == "structured":
            hyp_text = " ".join(
                item["text"]
                for item in prediction.get("words", [])
                if isinstance(item, Mapping) and isinstance(item.get("text"), str)
            )
        else:
            hyp_text = get_hyp_text(solution_str)
        wer = measure_openasr_en_wer(hyp_text, ground_truth)
        result.update({key: wer[key] for key in ("wer", "n_err", "n_ref")})
    return result

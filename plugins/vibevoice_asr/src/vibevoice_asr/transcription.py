"""Strictly parse generated rich transcriptions; never score JSON as speech."""

import json
import math


def parse_transcription(text, *, allow_invalid_event_times=False):
    """Return canonical segments from the two official field-name variants.

    Invalid/truncated output raises instead of silently reporting a WER on a
    partial transcript. The caller retains the raw response and failure record.
    """
    text = text.strip()
    # The official offline prompt lets the model emit its assistant role line;
    # decoding with skip_special_tokens=True removes im_start but keeps this.
    if text.startswith("assistant\n"):
        text = text.removeprefix("assistant\n").lstrip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("Unclosed transcription code fence")
        text = "\n".join(lines[1:-1])
    items = json.loads(text)
    if not isinstance(items, list):
        raise ValueError("VibeVoice offline output must be a JSON array")
    result = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Transcription segments must be objects")
        content = item.get("Content")
        speaker = item.get("Speaker", item.get("Speaker ID"))
        start = item.get("Start", item.get("Start time"))
        end = item.get("End", item.get("End time"))
        if not isinstance(content, str):
            raise ValueError("Missing or invalid Content")
        non_speech = content.strip().lower() in NON_SPEECH_TAGS
        if (speaker is None and not non_speech) or isinstance(speaker, (dict, list, bool)):
            raise ValueError("Missing or invalid Speaker")
        finite_times = not any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (start, end)
        )
        timing_valid = finite_times and 0 <= start <= end
        if not timing_valid and not (finite_times and non_speech and allow_invalid_event_times):
            raise ValueError("Invalid segment time boundaries")
        result.append(
            {
                "start_s": float(start),
                "end_s": float(end),
                "speaker": None if speaker is None else str(speaker),
                "text": content,
            }
        )
        if not timing_valid:
            # WER excludes these events. Preserve the model's actual times and
            # expose its timing defect instead of repairing or hiding it.
            result[-1]["timing_valid"] = False
    return result


NON_SPEECH_TAGS = {
    "[unintelligible speech]",
    "[music]",
    "[human sounds]",
    "[environmental sounds]",
    "[noise]",
    "[silence]",
    # Observed in real offline generations as a speakerless acoustic event.
    "[sigh]",
}


def close_truncated_json(text):
    """Close JSON delimiters only; preserve every generated speech character.

    This is only for scoring outputs known to have hit the generation limit.
    Strict parsing remains the default. No words are supplied or removed.
    """
    body = text.strip().removeprefix("assistant\n").lstrip()
    if not body.startswith("["):
        raise ValueError("Cannot recover a truncated non-array output")
    stack = []
    in_string = False
    escaped = False
    for char in body:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack or stack.pop() != ("[" if char == "]" else "{"):
                raise ValueError("Mismatched JSON delimiters")
    if escaped:
        raise ValueError("Cannot invent an unfinished JSON escape")
    suffix = ('"' if in_string else "") + "".join("]" if opening == "[" else "}" for opening in reversed(stack))
    return body + suffix


def transcription_text(segments):
    """Join spoken content in emission order, excluding whole non-speech tags."""
    return " ".join(s["text"].strip() for s in segments if s["text"].strip().lower() not in NON_SPEECH_TAGS).strip()


def collapse_audio_tokens(input_ids, *, start_id, end_id, audio_id, newline_id=198):
    """Inverse of the official vLLM offline placeholder expansion.

    Keep the HF prompt verbatim, including its assistant prefix behavior. vLLM
    replaces AUDIO with speech-start, audio embeddings, speech-end and newline.
    """
    if input_ids.count(start_id) != 1 or input_ids.count(end_id) != 1:
        raise ValueError("Expected exactly one offline audio span")
    start = input_ids.index(start_id)
    end = input_ids.index(end_id)
    if end <= start or input_ids[end + 1 : end + 2] != [newline_id]:
        raise ValueError("Unexpected official offline audio boundary layout")
    return input_ids[:start] + [audio_id] + input_ids[end + 2 :]

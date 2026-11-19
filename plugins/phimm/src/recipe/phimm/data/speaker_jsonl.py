"""Format one supervised JSONL line per speech segment."""

import json
import math

SPEAKER_JSONL_PROMPT = (
    "Transcribe the English audio and identify the speakers.<audio>\n"
    "Return JSONL only, one object per speech segment in order of its start time. "
    "Always include string text. Include integer spk when the speaker is identifiable, "
    "numbering speakers from 1 by first appearance and reusing each speaker's number. "
    "When the boundaries are known, include integer start and end in units of 10 milliseconds "
    "from the start of this audio clip (for example, 170 ms is 17). "
    "Omit unknown fields. List overlapping speech as separate segments."
)


def _valid_time(value):
    return (type(value) is int or (type(value) is float and math.isfinite(value))) and value >= 0


def _span(item):
    start, end = item.get("start"), item.get("end")
    return (start, end) if _valid_time(start) and _valid_time(end) and start < end else None


def _speaker(item):
    value = item.get("speaker")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _text(item):
    value = item.get("text")
    return value.strip() if isinstance(value, str) else ""


def _time(item):
    value = item.get("start")
    return value if _valid_time(value) else math.inf


def _word_segments(words, intervals):
    """Populate annotated intervals using only words that can be assigned uniquely."""
    timed = [(item, span, _speaker(item)) for item in intervals if (span := _span(item)) is not None]
    buckets = [[] for _ in timed]
    unassigned = []
    for word in sorted(words, key=_time):
        if not _text(word):
            continue
        span, speaker = _span(word), _speaker(word)
        matches = []
        if span is not None:
            for index, (_, (start, end), label) in enumerate(timed):
                if (speaker is None or label is None or label == speaker) and start <= span[0] and span[1] <= end:
                    matches.append(index)
        if len(matches) == 1:
            buckets[matches[0]].append(word)
        else:
            unassigned.append(word)
    for (interval, _, label), group in zip(timed, buckets, strict=True):
        if not group:
            continue
        word_speakers = {_speaker(word) for word in group} - {None}
        if label is None and len(word_speakers) > 1:
            unassigned.extend(group)
            continue
        speaker = label or next(iter(word_speakers), None)
        yield {**interval, "speaker": speaker, "text": " ".join(_text(word) for word in group)}
    yield from unassigned


def format_speaker_jsonl(row):
    """Build turns and join adjacent same-speaker entries without filling timed gaps."""
    intervals = row.get("reference_segments") or []
    turns = [item for item in (row.get("reference_turns") or []) if _text(item)]
    if not turns:
        words = row.get("reference_words") or []
        if words:
            turns = list(_word_segments(words, intervals))
        else:
            turns = [item for item in intervals if _text(item)]
    if not turns:
        text = _text(row)
        if not text:
            raise ValueError("Speaker JSONL supervision requires nonempty reference text")
        turns = [{**(intervals[0] if len(intervals) == 1 else {}), "text": text}]

    speakers = {}
    segments = []
    duration = row.get("duration_s")
    clip_end = round(duration * 100) if _valid_time(duration) else None
    for turn in sorted(turns, key=_time):
        item = {}
        if (speaker := _speaker(turn)) is not None:
            item["spk"] = speakers.setdefault(speaker, len(speakers) + 1)
        item["text"] = _text(turn)
        if (span := _span(turn)) is not None:
            start, end = (round(value * 100) for value in span)
            # Keep short segments representable even when both ends round to
            # the same tick, without exceeding the quantized input duration.
            if clip_end is not None:
                end = min(end, clip_end)
            if start == end and end == clip_end and end > 0:
                start -= 1
            elif start == end and (clip_end is None or end < clip_end):
                end += 1
            if 0 <= start < end:
                item.update(start=start, end=end)
        previous = segments[-1] if segments else {}
        same_speaker = "spk" in item and previous.get("spk") == item["spk"]
        same_timing = ("start" in previous) == ("start" in item)
        if same_speaker and same_timing and ("start" not in item or item["start"] <= previous["end"]):
            previous["text"] += " " + item["text"]
            if "start" in item:
                previous["start"] = min(previous["start"], item["start"])
                previous["end"] = max(previous["end"], item["end"])
        else:
            segments.append(item)
    return "\n".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in segments)


def format_single_speaker_jsonl(text, audio_length_s):
    """Supervise pure ASR from zero to the input audio length in 10 ms units."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Single-speaker ASR requires nonempty reference text")
    if not _valid_time(audio_length_s) or audio_length_s <= 0:
        raise ValueError("Single-speaker ASR requires a finite positive audio length in seconds")
    end = round(audio_length_s * 100)
    if end <= 0:
        raise ValueError("Input audio is too short to represent in 10 ms units")
    return json.dumps({"spk": 1, "text": text, "start": 0, "end": end}, ensure_ascii=False, separators=(",", ":"))


def is_speaker_segment(fields, *, allow_empty_text=False):
    """Validate an object in the speaker JSONL schema."""
    if fields.keys() - {"spk", "text", "start", "end"} or not isinstance(fields.get("text"), str):
        return False
    if not allow_empty_text and not fields["text"].strip():
        return False
    if "spk" in fields and (type(fields["spk"]) is not int or fields["spk"] < 1):
        return False
    if "start" not in fields and "end" not in fields:
        return True
    start, end = fields.get("start"), fields.get("end")
    return type(start) is int and type(end) is int and 0 <= start < end


def is_speaker_jsonl(target):
    """Validate mandatory text and optional speaker and paired 10 ms boundaries."""
    if not isinstance(target, str) or not target.strip():
        return False
    try:
        for line in target.removesuffix("\n").split("\n"):
            if not line.lstrip().startswith("{"):
                return False
            pairs = json.loads(line, object_pairs_hook=list)
            fields = dict(pairs)
            if len(fields) != len(pairs) or not is_speaker_segment(fields):
                return False
    except (ValueError, TypeError, RecursionError, OverflowError):
        return False
    return True

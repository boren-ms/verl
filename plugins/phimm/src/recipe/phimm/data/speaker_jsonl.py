"""Format one supervised JSONL line per recording-local speaker."""

import json
import math

SPEAKER_JSONL_PROMPT = (
    "Transcribe the English audio and identify the speakers.<audio>\n"
    "Return JSONL only, one object per speaker, with integer spk and string text. "
    "Number speakers from 1 by first appearance. In each text, include everything "
    "that speaker says in chronological order. Do not include timestamps or speaker tags in text."
)


def format_speaker_jsonl(words):
    """Group timed reference words by speaker, keeping their spoken order."""
    if not isinstance(words, list) or not words:
        raise ValueError("Speaker JSONL supervision requires nonempty reference_words")
    ordered = []
    for word in words:
        start, speaker, text = word.get("start"), word.get("speaker"), word.get("text")
        if (
            not isinstance(start, (int, float))
            or not math.isfinite(start)
            or start < 0
            or not isinstance(speaker, str)
            or not speaker.strip()
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError("Speaker JSONL requires a start time, speaker label, and text on each reference word")
        ordered.append((start, speaker, text.strip()))
    by_speaker = {}
    for _, speaker, text in sorted(ordered, key=lambda word: word[0]):
        by_speaker.setdefault(speaker, []).append(text)
    return "\n".join(
        json.dumps({"spk": index, "text": " ".join(text)}, ensure_ascii=False, separators=(",", ":"))
        for index, text in enumerate(by_speaker.values(), 1)
    )


def is_speaker_jsonl(target):
    """Require one object per speaker with consecutive one-based IDs."""
    try:
        for index, line in enumerate(target.removesuffix("\n").split("\n"), 1):
            if not line.lstrip().startswith("{"):
                return False
            pairs = json.loads(line, object_pairs_hook=list)
            if not isinstance(pairs, list) or len(pairs) != 2:
                return False
            fields = dict(pairs)
            if (
                fields.keys() != {"spk", "text"}
                or type(fields["spk"]) is not int
                or fields["spk"] != index
                or not isinstance(fields["text"], str)
                or not fields["text"].strip()
            ):
                return False
    except (ValueError, TypeError, RecursionError):
        return False
    return bool(target.strip())

import json
import re
from typing import NotRequired, TypedDict

from recipe.phimm.data.speaker_jsonl import is_speaker_segment


class TaskOutputSegment(TypedDict):
    language: str | None
    text: str
    format: NotRequired[str]
    spk: NotRequired[int]
    start: NotRequired[int]
    end: NotRequired[int]


def clean_text(text: str) -> str:
    """Remove model markup, unmatched word-edge brackets, and empty punctuation."""
    cleanup_patterns = (
        r"(?m)^[ \t]*<src=[^>\n]+><tgt=[^>\n]+>[ \t]*(?:\n|$)",
        r"</?foreign_language>",
        r"<nonspeech>",
        r"</?lexical>",
        r"</?verbatim>",
        r"</?readable>",
        r"<sep>",
        r"</?TXT>",
    )
    cleaned = text

    for pattern in cleanup_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    cleaned = re.sub(r"(?<!\S)<([^\s<>]+)(?=\s|$)", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\S)([^\s<>]+)>(?=\s|$)", r"\1", cleaned).strip()

    if cleaned in (".", "。"):
        return ""

    return cleaned


def _parse_line(line: str) -> TaskOutputSegment:
    fallback: TaskOutputSegment = {"language": None, "text": line}
    if not line.lstrip(" \t").startswith("{"):
        return fallback
    try:
        pairs = json.loads(line, object_pairs_hook=list)
    except (ValueError, RecursionError):
        return fallback
    fields = dict(pairs)
    if len(fields) != len(pairs):
        return fallback
    language, text = fields.get("language"), fields.get("text")
    if fields.keys() == {"language", "text"} and isinstance(language, str) and language.strip():
        if isinstance(text, str):
            return {"language": language.strip(), "text": text}
    elif is_speaker_segment(fields, allow_empty_text=True):
        return {"language": None, "format": "speaker_jsonl", **fields}
    elif isinstance(text, str) and not fields.keys() - {"spk", "text", "start", "end"}:
        # A bad timestamp must not turn a readable transcript into JSON syntax
        # for ASR scoring, but the response is not a valid diarization record.
        return {"language": None, "format": "invalid_speaker_jsonl", "text": text}
    return fallback


def is_jsonl_segment(segment: TaskOutputSegment) -> bool:
    """Check response syntax independently of language prediction."""
    return segment["language"] is not None or segment.get("format") == "speaker_jsonl"


def parse_jsonl_response(segment_text: str) -> list[TaskOutputSegment]:
    """Parse JSONL in order, preserving invalid lines as raw text."""
    return [_parse_line(line) for line in segment_text.removesuffix("\n").split("\n")]


def parse_response_outputs(solution_str: str | list[str]) -> list[list[TaskOutputSegment]]:
    """Parse one response or an ordered list, preserving independent boundaries."""
    responses = solution_str if isinstance(solution_str, list) else [solution_str]
    if not responses:
        raise ValueError("ASR reward requires at least one segment response")
    if any(not isinstance(response, str) for response in responses):
        raise TypeError("ASR reward expects a string or a list of strings")
    return [parse_jsonl_response(segment_text) for segment_text in responses]


def get_hyp_text(solution_str: str | list[str]):
    """Join cleaned text from one response or ordered independent responses."""
    task_outputs = parse_response_outputs(solution_str)
    cleaned = (clean_text(segment["text"]) for output in task_outputs for segment in output)
    return " ".join(filter(None, cleaned))

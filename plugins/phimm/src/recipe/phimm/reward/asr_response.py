import re
from typing import TypedDict

_SEGMENT_RE = re.compile(
    r"(?:\A|\n)<src=(?P<src>[^>\n]+)><tgt=(?P<tgt>[^>\n]+)>[^\S\n]*\n"
    r"(?P<text>.*?)(?=\n<src=|\Z)",
    re.DOTALL,
)


class TaskOutputSegment(TypedDict):
    src: str | None
    tgt: str | None
    text: str | None


def _segment(src=None, tgt=None, text=None) -> TaskOutputSegment:
    return {"src": src, "tgt": tgt, "text": text}


def clean_text(text: str) -> str:
    """Remove model markup, unmatched word-edge brackets, and empty punctuation."""
    cleanup_patterns = (
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


def get_asr_text(task_output):
    """Join parsed ASR text segments without modifying their content."""
    return " ".join(segment["text"] for segment in task_output if segment["text"] is not None)


def parse_task_output(solution_str: str, version=None):
    """Parse a 2609 response, preserving malformed text for reward checks.

    Complete language headers and optional ``<TXT>`` tags are supported, as
    is standalone ``<nonspeech>``. An omitted version selects 2609.
    """
    if version is not None and str(version) != "2609":
        raise ValueError(f"ASR reward supports only model version 2609; got {version!r}")
    output = solution_str.strip()
    if output.lower() == "<nonspeech>":
        return [_segment(text="<nonspeech>")]
    segments = []
    pos = 0
    while pos < len(output):
        match = _SEGMENT_RE.match(output, pos)
        if not match:
            segments.append(_segment(text=output[pos:].strip()))
            break
        segments.append(
            _segment(
                src=match.group("src").strip(),
                tgt=match.group("tgt").strip(),
                text=match.group("text").strip(),
            )
        )
        pos = match.end()
    return segments or [_segment()]


def parse_response_outputs(solution_str: str | list[str], version=None) -> list[list[TaskOutputSegment]]:
    """Parse one response or an ordered list, preserving independent boundaries."""
    responses = solution_str if isinstance(solution_str, list) else [solution_str]
    if not responses:
        raise ValueError("ASR reward requires at least one segment response")
    if any(not isinstance(response, str) for response in responses):
        raise TypeError("ASR reward expects a string or a list of strings")
    return [parse_task_output(response, version=version) for response in responses]


def get_hyp_text(solution_str: str | list[str], version=None):
    """Join cleaned text from one response or ordered independent responses."""
    task_outputs = parse_response_outputs(solution_str, version=version)
    cleaned = (clean_text(segment["text"] or "") for output in task_outputs for segment in output)
    return " ".join(filter(None, cleaned))

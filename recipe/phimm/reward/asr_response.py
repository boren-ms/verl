import re
from typing import TypedDict


_SEGMENT_RE = re.compile(
    r"(?:\A|\n)<src=(?P<src>[^>\n]+)><tgt=(?P<tgt>[^>\n]+)>[^\S\n]*\n"
    r"(?P<text>.*?)(?=\n<src=|\Z)",
    re.DOTALL,
)
_2607_HEADER_RE = re.compile(r"^Audio Language:\s*(?P<langs>[^\n]+?)\.?\n(?P<body>.*)$", re.DOTALL)
_2607_TAG_RE = re.compile(r"^<(?P<tag>ASR(?:_[^>]+)?)>(?P<inner>.*)</(?P=tag)>$", re.DOTALL)
_2607_SEGMENT_RE = re.compile(r"\s*<lang=(?P<lang>[^>]+)><TXT>(?P<text>.*?)</TXT>\s*", re.DOTALL)
_2609_MODE_PREFIX_RE = re.compile(
    r"^\s*<(?:asr_)?(?:lexical|verbatim|readable)>\s*",
    re.IGNORECASE,
)
_TXT_WRAPPER_RE = re.compile(r"^\s*<TXT>(?P<text>.*?)</TXT>\s*$", re.DOTALL | re.IGNORECASE)


class TaskOutputSegment(TypedDict):
    src: str | None
    tgt: str | None
    text: str | None


def _segment(src=None, tgt=None, text=None) -> TaskOutputSegment:
    return {"src": src, "tgt": tgt, "text": text}


def clean_text(text: str) -> str:
    """Remove model markup and empty punctuation from generated text."""
    cleanup_patterns = (
        r"<nonspeech>",
        r"</?lexical>",
        r"</?verbatim>",
        r"</?readable>",
        r"<sep>",
        r"</?TXT>",
        r"</?AS[RT][^>]*>",
        r"</?lang(?:=[^>]*)?>",
    )
    cleaned = text

    for pattern in cleanup_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    if cleaned in (".", "。"):
        return ""

    return cleaned


def parse_task_output(solution_str, version=None):
    """Parse an ASR response into a list of ``src``/``tgt``/``text`` segments."""
    if str(version) == "2609":
        return _parse_task_output_2609(solution_str)
    if str(version) == "2607":
        return _parse_task_output_2607(solution_str)
    return _parse_task_output_2607(solution_str)


def get_asr_text(task_output):
    """Join the text segments from a parsed ASR task output."""
    return " ".join(segment["text"] for segment in task_output if segment["text"] is not None)


def _unwrap_txt(text: str) -> str | None:
    text = _2609_MODE_PREFIX_RE.sub("", text, count=1)
    match = _TXT_WRAPPER_RE.match(text)
    return match.group("text") if match else None


def _parse_task_output_2607(solution_str):
    """Parse the legacy 2607 ``Audio Language`` / ``<ASR>`` envelope."""
    if not isinstance(solution_str, str):
        return [_segment()]
    output = solution_str.strip()
    header_match = _2607_HEADER_RE.match(output)
    body = header_match.group("body").strip() if header_match else output
    tag_match = _2607_TAG_RE.match(body)
    if tag_match is None:
        return [_segment()]

    segments = []
    inner = tag_match.group("inner")
    pos = 0
    while pos < len(inner):
        segment_match = _2607_SEGMENT_RE.match(inner, pos)
        if segment_match is None:
            return [_segment()]
        segments.append((segment_match.group("lang").strip(), segment_match.group("text")))
        pos = segment_match.end()
    if not segments:
        return [_segment()]

    header_langs = _split_2607_langs(header_match.group("langs")) if header_match else []
    return [
        _segment(
            src=header_langs[index] if index < len(header_langs) else None,
            tgt=lang,
            text=text,
        )
        for index, (lang, text) in enumerate(segments)
    ]


def _split_2607_langs(header: str) -> list[str]:
    """Split a 2607 header into its individual language names."""
    header = header.strip().rstrip(".")
    header = re.sub(r"\band\b", " ", header, flags=re.IGNORECASE)
    return [part for part in re.split(r"[,\s、&/]+", header) if part]


def _parse_task_output_2609(solution_str):
    """Parse a 2609 single- or multi-segment task output."""
    if not isinstance(solution_str, str):
        return [_segment()]
    output = solution_str.strip()
    if output.lower() == "<nonspeech>":
        return [_segment(text="<nonspeech>")]
    if not output:
        return [_segment()]
    first_header = output.find("\n<src=")
    if not output.startswith("<src=") and first_header < 0:
        return [_segment(text=_unwrap_txt(output))]

    segments = []
    if first_header >= 0 and not output.startswith("<src="):
        text = _unwrap_txt(output[:first_header])
        segments.append(_segment(text=text.strip() if text is not None else None))
        pos = first_header
    else:
        pos = 0
    while pos < len(output):
        match = _SEGMENT_RE.match(output, pos)
        if not match:
            segments.append(_segment())
            break
        text = _unwrap_txt(match.group("text"))
        segments.append(
            _segment(
                src=match.group("src").strip(),
                tgt=match.group("tgt").strip(),
                text=text.strip() if text is not None else None,
            )
        )
        pos = match.end()
    return segments or [_segment()]


def get_hyp_text(solution_str, version=None):
    task_output = parse_task_output(solution_str, version=version)
    has_parsed_text = any(segment["text"] is not None for segment in task_output)
    hyp_text = get_asr_text(task_output) if has_parsed_text else str(solution_str or "")
    return clean_text(hyp_text)
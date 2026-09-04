import re


_SEGMENT_RE = re.compile(
    r"(?:\A|\n)<src=(?P<src>[^>\n]+)><tgt=(?P<tgt>[^>\n]+)>[^\S\n]*\n"
    r"(?P<text>.*?)(?=\n<src=|\Z)",
    re.DOTALL,
)
_2607_HEADER_RE = re.compile(r"^Audio Language:\s*(?P<langs>[^\n]+?)\.?\n(?P<body>.*)$", re.DOTALL)
_2607_TAG_RE = re.compile(r"^<(?P<tag>ASR(?:_[^>]+)?)>(?P<inner>.*)</(?P=tag)>$", re.DOTALL)
_2607_SEGMENT_RE = re.compile(r"\s*<lang=(?P<lang>[^>]+)><TXT>(?P<text>.*?)</TXT>\s*", re.DOTALL)
_ASR_MODE_TAG_RE = re.compile(
    r"</?(?:asr_)?(?:lexical|verbatim|readable)>",
    re.IGNORECASE,
)


def clean_asr_mode_tags(text: str) -> str:
    """Remove lexical, verbatim, and readable ASR mode tags."""
    return _ASR_MODE_TAG_RE.sub("", text)


def parse_task_output(solution_str, version=None):
    """Parse an ASR task output into ``(src_langs, tgt_langs, seg_texts)``."""
    if str(version) == "2607":
        return _parse_task_output_2607(solution_str)
    return _parse_task_output_2609(solution_str)


def get_asr_text(task_output):
    """Join the text segments from a parsed ASR task output."""
    return " ".join(task_output[2])


def _parse_task_output_2607(solution_str):
    """Parse the legacy 2607 ``Audio Language`` / ``<ASR>`` envelope."""
    if not isinstance(solution_str, str):
        return None
    output = solution_str.strip()
    header_match = _2607_HEADER_RE.match(output)
    body = header_match.group("body").strip() if header_match else output
    tag_match = _2607_TAG_RE.match(body)
    if tag_match is None:
        return None

    segments = []
    inner = tag_match.group("inner")
    pos = 0
    while pos < len(inner):
        segment_match = _2607_SEGMENT_RE.match(inner, pos)
        if segment_match is None:
            return None
        segments.append((segment_match.group("lang").strip(), segment_match.group("text")))
        pos = segment_match.end()
    if not segments:
        return None

    header_langs = _split_2607_langs(header_match.group("langs")) if header_match else []
    segment_langs = [lang for lang, _ in segments]
    segment_texts = [text for _, text in segments]
    return header_langs, segment_langs, segment_texts


def _split_2607_langs(header: str) -> list[str]:
    """Split a 2607 header into its individual language names."""
    header = header.strip().rstrip(".")
    header = re.sub(r"\band\b", " ", header, flags=re.IGNORECASE)
    return [part for part in re.split(r"[,\s、&/]+", header) if part]


def _parse_task_output_2609(solution_str):
    """Parse a 2609 single- or multi-segment task output."""
    if not isinstance(solution_str, str):
        return None
    output = clean_asr_mode_tags(solution_str).strip()
    if not output:
        return None
    first_header = output.find("\n<src=")
    if not output.startswith("<src=") and first_header < 0:
        return [], [], [output]

    segments = []
    if first_header >= 0 and not output.startswith("<src="):
        segments.append((None, None, output[:first_header].strip()))
        pos = first_header
    else:
        pos = 0
    while pos < len(output):
        match = _SEGMENT_RE.match(output, pos)
        if not match:
            return None
        segments.append(
            (
                match.group("src").strip(),
                match.group("tgt").strip(),
                match.group("text").strip(),
            )
        )
        pos = match.end()
    if not segments:
        return None
    src_langs = [src for src, _, _ in segments if src is not None]
    tgt_langs = [tgt for _, tgt, _ in segments if tgt is not None]
    seg_texts = [text for _, _, text in segments]
    return src_langs, tgt_langs, seg_texts


def get_hyp_text(solution_str, version=None):
    task_output = parse_task_output(solution_str, version=version)
    return get_asr_text(task_output) if task_output is not None else str(solution_str or "")
"""Deterministic LibriSpeech keyword labels and the opt-in KEYWORDS envelope."""

from dataclasses import dataclass, field
import re

PROTOCOL = "asr-keywords-v1"
ANNOTATION_POLICY = "ls-common5k-v1"
KEYWORD_TAG = "<KEYWORDS>"
KEYWORD_INSTRUCTION = (
    "\nAlso output the rare words that occur in this audio. After the complete "
    "transcription, write <KEYWORDS> on a new line, then a comma-separated list "
    "on the next line. List each word once, in its order of first occurrence. "
    "If there are no keywords, leave that line empty. Do not add explanations."
)
WORD_RE = re.compile(r"[\w'\-]+")


def words(text):
    return WORD_RE.findall(text.lower())


def target_keywords(text, common_words):
    return list(dict.fromkeys(w for w in words(text) if w not in common_words))


def validate_keywords(text, keywords):
    """Missing labels are errors; [] is a real, supervised empty target."""
    if not isinstance(keywords, list) or any(not isinstance(w, str) for w in keywords):
        raise ValueError("keywords must be an explicit list of words, including [] for empty targets")
    if any(not WORD_RE.fullmatch(w) or w != w.lower() for w in keywords):
        raise ValueError("v1 keywords must be lowercase single words (no CSV quoting or phrases)")
    if len(set(keywords)) != len(keywords):
        raise ValueError("duplicate target keywords")
    tokens = words(text)
    if not set(keywords) <= set(tokens):
        raise ValueError("target keyword is absent from the reference transcription")
    if keywords != list(dict.fromkeys(w for w in tokens if w in set(keywords))):
        raise ValueError("target keywords must follow first occurrence in the reference")
    if re.search(r"<\s*/?keywords\b", text, re.IGNORECASE):
        raise ValueError("reserved keyword marker in reference transcription")
    return keywords


def append_keyword_block(body, keywords):
    line = ", ".join(keywords)
    return body + "\n" + KEYWORD_TAG + "\n" + line + "\n"


@dataclass
class KeywordOutput:
    raw_response: str
    body_raw: str
    predicted_keywords: list[str] = field(default_factory=list)
    keyword_block_status: str = "missing"
    protocol_errors: list[str] = field(default_factory=list)

    @property
    def valid(self):
        return self.keyword_block_status in {"valid_empty", "valid_nonempty", "valid_empty_normalized"}


def parse_keyword_output(response, finish_reason=None):
    """Strip only a valid uppercase block; preserve invalid suffix words for WER.

    Terminal metadata is essential when a decoder strips an empty target's
    trailing whitespace. A length-limited response is never a valid empty list.
    """
    raw = response if isinstance(response, str) else ""
    text = raw.replace("\r\n", "\n")
    matches = list(re.finditer(r"(?m)^<KEYWORDS>$", text))
    markers = list(re.finditer(r"<\s*/?keywords\b[^>]*>?", text, re.IGNORECASE))
    result = KeywordOutput(raw, text)
    if len(matches) != 1 or len(markers) != 1:
        if markers:
            result.keyword_block_status = "malformed"
            result.protocol_errors.append("invalid_or_duplicate_header")
        if finish_reason in {"length", "timeout"}:
            result.keyword_block_status = "truncated"
        # Prevent a later tag-cleaning normalizer from hiding bad suffix words.
        if markers:
            pos = markers[0].start()
            result.body_raw = text[:pos] + text[pos:].replace("<", " ").replace(">", " ")
        return result

    match = matches[0]
    body = text[: match.start()].rstrip("\n")
    tail = text[match.end() :]
    if tail.startswith("\n"):
        tail = tail[1:]
    lines = tail.split("\n")
    line = lines[0].strip()
    errors = []
    if any(part.strip() for part in lines[1:]):
        errors.append("extra_lines")
    if line:
        items = [part.strip() for part in line.split(",")]
        result.predicted_keywords = [part.lower() for part in items if part]
        if any(not WORD_RE.fullmatch(part) for part in items):
            errors.append("invalid_or_empty_item")
        if len(set(result.predicted_keywords)) != len(result.predicted_keywords):
            errors.append("duplicate_items")
    elif not tail and finish_reason not in {"stop", "eos"}:
        errors.append("incomplete_empty")

    if finish_reason in {"length", "timeout"}:
        result.keyword_block_status = "truncated"
        errors.append("generation_truncated")
    elif errors:
        result.keyword_block_status = "incomplete" if errors == ["incomplete_empty"] else "malformed"
    else:
        result.keyword_block_status = "valid_nonempty" if line else "valid_empty" if tail else "valid_empty_normalized"
    result.protocol_errors = errors
    if result.valid:
        result.body_raw = body
    else:
        suffix = text[match.start() :].replace("<", " ").replace(">", " ")
        result.body_raw = body + "\n" + suffix
    return result

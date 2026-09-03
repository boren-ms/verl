"""Lightweight punctuation & capitalization error measurement.

Uses ``jiwer`` for word-level alignment and simple Unicode-category-based
classification to count punctuation and capitalization errors between a
reference and hypothesis string.  No dependency on DTER / dfmetrics / dotnet.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from jiwer import process_words

from recipe.phimm.utils.languages import get_language_code


# One ``<src=X><tgt=Y>\n...`` segment. Several newline-separated
# segments may appear for code-switch / mixed audio.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Regular punctuation marks that matter for ASR punc accuracy.
# Excludes apostrophes, hyphens, quotes, brackets, etc.
REGULAR_PUNC = set(".,?!;:;。？！、，：；…")


def _is_punc_char(c: str) -> bool:
    """Return True if *c* is a Unicode punctuation character."""
    return unicodedata.category(c).startswith("P")


def _is_regular_punc(c: str) -> bool:
    """Return True if *c* is a regular (sentence-level) punctuation mark."""
    return c in REGULAR_PUNC


def _is_edge_punc(word: str, idx: int) -> bool:
    """Return True if char at *idx* is a regular punctuation at a word edge.

    Only punctuation at the very start or end of *word* counts.  Mid-word
    punctuation (e.g. the ``.`` in ``4.5`` or ``U.S.A``) is ignored.
    """
    if not _is_regular_punc(word[idx]):
        return False
    # Walk outward — all chars between idx and the nearest edge must also be punc.
    # e.g. "Hello," → comma is edge; "4.5" → period is NOT edge.
    # Left edge: every char from 0..idx is punc.
    left_ok = all(_is_punc_char(word[j]) for j in range(0, idx))
    # Right edge: every char from idx..end is punc.
    right_ok = all(_is_punc_char(word[j]) for j in range(idx + 1, len(word)))
    return left_ok or right_ok


def _strip_punc(word: str) -> str:
    """Remove all Unicode punctuation characters from *word*."""
    return "".join(c for c in word if not _is_punc_char(c))


def _is_pure_punc(word: str) -> bool:
    """Return True if *word* consists entirely of punctuation characters."""
    return len(word) > 0 and all(_is_punc_char(c) for c in word)


def _has_regular_punc(word: str) -> bool:
    """Return True if *word* has regular punctuation at a word edge."""
    return any(_is_edge_punc(word, i) for i in range(len(word)))


def _is_pure_regular_punc(word: str) -> bool:
    """Return True if *word* consists entirely of regular punctuation marks."""
    return len(word) > 0 and all(_is_regular_punc(c) for c in word)


def _punc_signature(word: str) -> str:
    """Return a string encoding edge regular-punctuation positions & characters.

    E.g. ``"Hello,"`` → ``",5"`` (comma at index 5).
    Used to detect whether two tokens differ only in punctuation attachment.
    Only considers regular punctuation at word edges.
    """
    return "".join(c + str(i) for i, c in enumerate(word) if _is_edge_punc(word, i))


# ---------------------------------------------------------------------------
# Edit classification
# ---------------------------------------------------------------------------


@dataclass
class EditDetail:
    """One aligned edit between ref and hyp."""

    op: str  # "sub", "ins", "del"
    ref_word: str | None
    hyp_word: str | None
    category: str | None = None  # one of {"punc", "cap", "lex"} or None


def classify_edit(ref_word: str, hyp_word: str) -> str:
    """Classify a substitution pair into one of ``{"punc", "cap", "lex"}``.

    Priority: ``lex`` > ``punc`` > ``cap`` — each edit counts as exactly one
    category, the most severe one.
    """
    ref_base = _strip_punc(ref_word)
    hyp_base = _strip_punc(hyp_word)

    if ref_base.lower() != hyp_base.lower():
        return "lex"
    if _punc_signature(ref_word) != _punc_signature(hyp_word):
        return "punc"
    if ref_base != hyp_base:
        return "cap"
    # Identical after stripping but original strings still differ — treat as punc.
    return "punc"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_fmt_acc(ref: str, hyp: str) -> dict:
    """Compute punctuation, capitalisation and lexical accuracies.

    Parameters
    ----------
    ref : str
        Reference (ground-truth) text — should contain punctuation & casing.
    hyp : str
        Hypothesis text from the ASR model.

    Returns
    -------
    dict with keys ``punc``, ``cap``, ``lex`` mapping to ``1 - err / (hit + err)``
    (defaults to 1.0 when ``hit + err == 0``).
    """
    ref = (ref or "").strip()
    hyp = (hyp or "").strip()

    if not ref and not hyp:
        return _empty_result()

    output = process_words(ref, hyp)

    ref_words: list[str] = output.references[0]
    hyp_words: list[str] = output.hypotheses[0]

    counts: dict[str, dict[str, int]] = {
        "punc": {"err": 0, "hit": 0},
        "cap": {"err": 0, "hit": 0},
        "lex": {"err": 0, "hit": 0},
    }
    details: list[EditDetail] = []

    for chunk in output.alignments[0]:
        if chunk.type == "equal":
            # Equal-aligned words count as hits per category independently:
            # punc hit if the word has edge regular punctuation; cap hit if
            # it contains any uppercase; lex hit always.
            for ri in range(chunk.ref_start_idx, chunk.ref_end_idx):
                rw = ref_words[ri]
                counts["lex"]["hit"] += 1
                if _has_regular_punc(rw):
                    counts["punc"]["hit"] += 1
                if any(c.isupper() for c in rw):
                    counts["cap"]["hit"] += 1
            continue

        if chunk.type == "substitute":
            for ri, hi in zip(
                range(chunk.ref_start_idx, chunk.ref_end_idx),
                range(chunk.hyp_start_idx, chunk.hyp_end_idx),
                strict=False,
            ):
                rw = ref_words[ri]
                hw = hyp_words[hi]
                cat = classify_edit(rw, hw)
                details.append(EditDetail(op="sub", ref_word=rw, hyp_word=hw, category=cat))
                counts[cat]["err"] += 1

        elif chunk.type == "delete":
            for ri in range(chunk.ref_start_idx, chunk.ref_end_idx):
                rw = ref_words[ri]
                cat = "punc" if _is_pure_regular_punc(rw) else "lex"
                details.append(EditDetail(op="del", ref_word=rw, hyp_word=None, category=cat))
                counts[cat]["err"] += 1

        elif chunk.type == "insert":
            for hi in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                hw = hyp_words[hi]
                cat = "punc" if _is_pure_regular_punc(hw) else "lex"
                details.append(EditDetail(op="ins", ref_word=None, hyp_word=hw, category=cat))
                counts[cat]["err"] += 1

    return {cat: _accuracy(c["err"], c["hit"]) for cat, c in counts.items()}


def _accuracy(err: int, hit: int) -> float:
    n = err + hit
    return 1.0 - (err / n if n else 0.0)


def _empty_result() -> dict:
    return {"punc": 1.0, "cap": 1.0, "lex": 1.0}


# ---------------------------------------------------------------------------
# Reward / scoring
# ---------------------------------------------------------------------------


def clean_asr_mode_tags(text: str) -> str:
    """Remove lexical, verbatim, and readable ASR mode tags."""
    return _ASR_MODE_TAG_RE.sub("", text)


def compute_openml_acc(hyp_text: str, ground_truth: str, tgt_lang: str, **kwargs) -> dict:
    """Return bracket, repeat, and tail-hallucination accuracies."""
    from recipe.phimm.utils.shared import has_brackets, has_repeat_error, has_tail_hallucination

    repeat_opts = kwargs.get("repeat") or {}
    tail_opts = kwargs.get("tail_hallucination") or {}
    return {
        "bracket": 1.0 - float(has_brackets(hyp_text)),
        "repeat": 1.0
        - float(
            has_repeat_error(
                hyp_text,
                ground_truth,
                min_reps=repeat_opts.get("min_reps", 4),
                max_ngram=repeat_opts.get("max_ngram", 5),
                tn_name=kwargs.get("text_norm"),
                lang=tgt_lang,
            )
        ),
        "tail_hallu": 1.0
        - float(
            has_tail_hallucination(
                hyp_text,
                ground_truth,
                min_words=tail_opts.get("min_words", 3),
                tn_name=tail_opts.get("text_norm"),
                lang=tgt_lang,
            )
        ),
    }


def _keyword_indices(words: list[str], keywords: list[str]) -> set[int]:
    indices = set()
    for keyword in keywords:
        keyword_words = keyword.split()
        if not keyword_words:
            continue
        width = len(keyword_words)
        for start in range(len(words) - width + 1):
            if words[start : start + width] == keyword_words:
                indices.update(range(start, start + width))
    return indices


def compute_kw_acc(
    ref: str,
    hyp: str,
    keywords: list[str] | None = None,
    text_norm: str | None = None,
    tgt_lang: str = "english",
) -> float:
    """Return word accuracy over normalized keyword spans."""
    from recipe.phimm.reward.asr_edge import _norm_text

    if not keywords:
        return 1.0

    ref_words = _norm_text(ref or "", name=text_norm, lang=tgt_lang).split()
    hyp_words = _norm_text(hyp or "", name=text_norm, lang=tgt_lang).split()
    kw_words = [_norm_text(keyword, name=text_norm, lang=tgt_lang) for keyword in keywords]
    ref_keyword_indices = _keyword_indices(ref_words, kw_words)
    hyp_keyword_indices = _keyword_indices(hyp_words, kw_words)

    output = process_words(" ".join(ref_words), " ".join(hyp_words))
    errors = 0
    references = 0
    for chunk in output.alignments[0]:
        if chunk.type == "equal":
            references += sum(idx in ref_keyword_indices for idx in range(chunk.ref_start_idx, chunk.ref_end_idx))
        elif chunk.type in {"substitute", "delete"}:
            keyword_errors = sum(idx in ref_keyword_indices for idx in range(chunk.ref_start_idx, chunk.ref_end_idx))
            references += keyword_errors
            errors += keyword_errors
        elif chunk.type == "insert":
            errors += sum(idx in hyp_keyword_indices for idx in range(chunk.hyp_start_idx, chunk.hyp_end_idx))

    return 1.0 - errors / references if references else 1.0


def _parse_response(solution_str, ground_truth=None, **kwargs):
    """Extract text, format/language, lexical, and edge-check accuracies."""
    from recipe.phimm.reward.asr_edge import measure

    extra_info = kwargs.get("extra_info") or {}
    tgt_lang = extra_info.get("language", kwargs.get("language", "English")).lower().strip()
    version = kwargs.get("version")
    task_output = _parse_task_output(solution_str, version=version)
    hyp_text = get_asr_text(task_output) if task_output is not None else str(solution_str or "")

    char_error = measure(hyp_text, ground_truth, tgt_lang=tgt_lang, unit="char", **kwargs)
    word_error = measure(hyp_text, ground_truth, tgt_lang=tgt_lang, unit="word", **kwargs)
    keyword_acc = compute_kw_acc(
        ground_truth or "",
        hyp_text,
        keywords=extra_info.get("keywords"),
        text_norm=kwargs.get("text_norm"),
        tgt_lang=tgt_lang,
    )
    fmts = compute_fmt_acc(ground_truth or "", hyp_text or "")
    openml_acc = compute_openml_acc(hyp_text, ground_truth, tgt_lang, **kwargs)

    return {
        "char": char_error.accuracy(),
        "word": word_error.accuracy(),
        "keyword": keyword_acc,
        **fmts,
        "lang": check_lang(task_output, tgt_lang),
        "fmt": float(check_fmt(task_output)),
        **openml_acc,
    }


def _parse_task_output(solution_str, version=None):
    """Parse an ASR task output into ``(src_langs, tgt_langs, seg_texts)``.

    Model version 2607 uses the legacy ``Audio Language`` / ``<ASR>``
    envelope. Newer versions use the ``<src=X><tgt=Y>`` format.

    """
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
    """Parse the 2609 single- or multi-segment task output.

    The first ``<src=X><tgt=Y>`` header is optional because it may already be
    supplied as the generation prefix. Later code-switch / mixed-audio
    segments must have newline-separated headers. Returns ``None`` when a
    structured output is malformed.
    """
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
        m = _SEGMENT_RE.match(output, pos)
        if not m:
            return None
        segments.append(
            (
                m.group("src").strip(),
                m.group("tgt").strip(),
                m.group("text").strip(),
            )
        )
        pos = m.end()
    if not segments:
        return None
    src_langs = [src for src, _, _ in segments if src is not None]
    tgt_langs = [tgt for _, tgt, _ in segments if tgt is not None]
    seg_texts = [text for _, _, text in segments]
    return src_langs, tgt_langs, seg_texts


def _lang_code_set(lang) -> set[str]:
    """Return the set of ISO language codes contained in *lang* (or empty set)."""
    if not lang:
        return set()
    return {c for c in get_language_code(lang).split("_") if c}


def check_lang(task_output, tgt_lang) -> float:
    """Language-identification score in ``[0, 1]`` with partial credit.

    Predicted language(s) come from the parsed target/segment language values.
    A missing or malformed parsed output scores ``0.0``.

    The score is the Jaccard overlap between the predicted and target language
    sets, so a code-switch output that identifies only some of the spoken
    languages still earns proportional credit (``1.0`` = exact set match).

    A ``<nonspeech>`` hypothesis always scores ``1.0`` (no language to judge).
    """
    if task_output is None:
        return 0.0

    _, tgt_langs, seg_texts = task_output
    if " ".join(seg_texts).strip().lower() == "<nonspeech>":
        return 1.0

    tgt_codes = _lang_code_set(tgt_lang)
    pred_codes: set[str] = set()
    for name in tgt_langs:
        pred_codes |= _lang_code_set(name)

    if not pred_codes:
        return 1.0
    if not tgt_codes:
        return 0.0
    return len(pred_codes & tgt_codes) / len(tgt_codes)


def check_fmt(task_output) -> bool:
    """Return whether the output was parsed as a supported ASR format."""
    return task_output is not None


def clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def signed_pow(x, gamma):
    sign = 1 if x >= 0 else -1
    return (abs(x) ** gamma) * sign


def reduce_scores(scores, mode="sum"):
    """Reduce a list of scores into a single value.

    Modes: "sum", "mean", "multiply", "geometric", "harmonic".
    """
    if not scores:
        return 0.0
    if mode == "multiply":
        score = 1.0
        for s in scores:
            score *= s
        return score
    elif mode == "mean":
        return sum(scores) / len(scores)
    elif mode == "geometric":
        product = 1.0
        for s in scores:
            product *= abs(s)
        return product ** (1.0 / len(scores))
    elif mode == "harmonic":
        try:
            return len(scores) / sum(1.0 / s for s in scores)
        except ZeroDivisionError:
            return 0.0
    else:
        return sum(scores)


def scale_score(acc, cfg):
    """Compute a single weighted score from an accuracy value and config."""
    cfg = cfg or {}
    beta = float(cfg.get("beta", 1.0))
    gamma = float(cfg.get("gamma", 1.0))
    lo = float(cfg.get("low", 0.0))
    hi = float(cfg.get("high", 1.0))
    acc = clip(acc, lo, hi)
    return beta * signed_pow(acc, gamma)


def compute_score(solution_str, ground_truth, **kwargs):
    """Combined lexical + formatting reward.

    Uses :func:`recipe.phimm.reward.asr_edge.measure` for the lexical accuracy
    and :func:`compute_fmt_acc` for punctuation/capitalisation accuracy.

    Only the components listed in ``scores`` contribute to the reward. The
    contribution of each component ``k`` is::

        beta * signed_pow(clip(acc_k, lo, hi), gamma)

    Both ``beta`` and ``gamma`` default to ``1.0`` when omitted.
    A component with ``cut`` forces the combined reward to zero when its raw
    accuracy is less than or equal to that threshold.

    Configuration example (YAML)::

        reward_kwargs:
          reduce: sum            # "sum" (default), "mean", or "multiply"
          scores:
            char: {beta: 1.0, gamma: 0.5, low: 0.0, high: 1.0}
            fmt: {beta: 0.5, cut: 0.5}
    """
    parsed = _parse_response(solution_str, ground_truth=ground_truth, **kwargs)

    measures = kwargs.get("measures") or {}
    reduce = kwargs.get("reduce", "sum").lower()
    gamma = float(kwargs.get("gamma", 1.0))
    scores = [scale_score(parsed.get(k, 1.0), cfg) for k, cfg in measures.items()]
    score = reduce_scores(scores, reduce)
    score = signed_pow(score, gamma)

    if any("cut" in cfg and parsed.get(name, 1.0) <= float(cfg["cut"]) for name, cfg in measures.items()):
        score = 0.0

    return {
        "score": score,
        **parsed,
    }


def lang_score(solution_str, ground_truth=None, **kwargs):
    """Return only the language-identification reward and metric."""
    extra_info = kwargs.get("extra_info") or {}
    tgt_lang = extra_info.get("language", kwargs.get("language", "English")).lower().strip()
    version = kwargs.get("version")
    task_output = _parse_task_output(solution_str, version=version)
    p_lang = check_lang(task_output, tgt_lang)
    return {
        "score": p_lang,
        "p_lang": p_lang,
    }

"""Lightweight punctuation & capitalization error measurement.

Uses ``jiwer`` for word-level alignment and simple Unicode-category-based
classification to count punctuation and capitalization errors between a
reference and hypothesis string.  No dependency on DTER / dfmetrics / dotnet.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from jiwer import process_words


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
    return "".join(
        c + str(i)
        for i, c in enumerate(word)
        if _is_edge_punc(word, i)
    )


# ---------------------------------------------------------------------------
# Edit classification
# ---------------------------------------------------------------------------


@dataclass
class EditDetail:
    """One aligned edit between ref and hyp."""

    op: str  # "sub", "ins", "del"
    ref_word: str | None
    hyp_word: str | None
    categories: set[str] = field(default_factory=set)  # subset of {"punc", "cap", "lex"}


def classify_edit(ref_word: str, hyp_word: str) -> set[str]:
    """Classify a substitution pair into ``{"punc", "cap", "lex"}``.

    A single substitution may belong to more than one category.

    * **punc** – the words differ in punctuation attachment (e.g. ``"Hello,"``
      vs ``"Hello"``).
    * **cap** – the words differ in capitalisation (e.g. ``"Hello"`` vs
      ``"hello"``).
        * **lex** – the base words (stripped of punctuation, compared
      case-insensitively) are different.
    """
    cats: set[str] = set()

    ref_base = _strip_punc(ref_word)
    hyp_base = _strip_punc(hyp_word)

    # Check if the underlying word is the same (ignoring case + punc).
    if ref_base.lower() == hyp_base.lower():
        # Pure formatting difference.
        if _punc_signature(ref_word) != _punc_signature(hyp_word):
            cats.add("punc")
        if ref_base != hyp_base:  # case differs
            cats.add("cap")
        # Edge-case: identical after stripping but original strings still
        # differ (e.g. different whitespace normalisation) — treat as punc.
        if not cats:
            cats.add("punc")
    else:
        cats.add("lex")
        # There may still be a punc/cap component on top of the lexical error.
        if _punc_signature(ref_word) != _punc_signature(hyp_word):
            cats.add("punc")
        if ref_base.lower() == hyp_base.lower():
            # Should not reach here (handled above) — defensive.
            cats.add("cap")
        elif ref_base != hyp_base and ref_base.lower() != hyp_base.lower():
            pass  # genuine lexical difference

    return cats


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
    dict with keys:

    * ``punc`` – ``1 - punc_err / punc_ref`` (1.0 when punc_ref == 0).
    * ``cap``  – ``1 - cap_err / cap_ref`` (1.0 when cap_ref == 0).
    * ``lex``  – ``1 - lex_err / n_ref`` (1.0 when n_ref == 0).
    """
    ref = (ref or "").strip()
    hyp = (hyp or "").strip()

    if not ref and not hyp:
        return _empty_result()

    output = process_words(ref, hyp)

    ref_words: list[str] = output.references[0]
    hyp_words: list[str] = output.hypotheses[0]

    punc_err = 0
    cap_err = 0
    lex_err = 0
    details: list[EditDetail] = []

    for chunk in output.alignments[0]:
        if chunk.type == "equal":
            continue

        if chunk.type == "substitute":
            for ri, hi in zip(
                range(chunk.ref_start_idx, chunk.ref_end_idx),
                range(chunk.hyp_start_idx, chunk.hyp_end_idx),
            ):
                rw = ref_words[ri]
                hw = hyp_words[hi]
                cats = classify_edit(rw, hw)
                d = EditDetail(op="sub", ref_word=rw, hyp_word=hw, categories=cats)
                details.append(d)
                if "punc" in cats:
                    punc_err += 1
                if "cap" in cats:
                    cap_err += 1
                if "lex" in cats:
                    lex_err += 1

        elif chunk.type == "delete":
            for ri in range(chunk.ref_start_idx, chunk.ref_end_idx):
                rw = ref_words[ri]
                cats: set[str] = set()
                if _is_pure_regular_punc(rw):
                    cats.add("punc")
                else:
                    cats.add("lex")
                    if _has_regular_punc(rw):
                        cats.add("punc")
                details.append(EditDetail(op="del", ref_word=rw, hyp_word=None, categories=cats))
                if "punc" in cats:
                    punc_err += 1
                if "lex" in cats:
                    lex_err += 1

        elif chunk.type == "insert":
            for hi in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                hw = hyp_words[hi]
                cats: set[str] = set()
                if _is_pure_regular_punc(hw):
                    cats.add("punc")
                else:
                    cats.add("lex")
                    if _has_regular_punc(hw):
                        cats.add("punc")
                details.append(EditDetail(op="ins", ref_word=None, hyp_word=hw, categories=cats))
                if "punc" in cats:
                    punc_err += 1
                if "lex" in cats:
                    lex_err += 1

    n_ref = len(ref_words)
    punc_ref = sum(1 for w in ref_words if _has_regular_punc(w))
    cap_ref = sum(1 for w in ref_words if any(c.isupper() for c in w))

    return {
        "punc": 1.0 - (punc_err / punc_ref if punc_ref else 0.0),
        "cap": 1.0 - (cap_err / cap_ref if cap_ref else 0.0),
        "lex": 1.0 - (lex_err / n_ref if n_ref else 0.0),
    }


def _empty_result() -> dict:
    return {
        "punc": 1.0,
        "cap": 1.0,
        "lex": 1.0,
    }


# ---------------------------------------------------------------------------
# Reward / scoring
# ---------------------------------------------------------------------------


def _parse_response(solution_str, ground_truth=None, **kwargs):
    """Extract hyp text, target language, fmt/lang accuracy, and char/punc/cap/lex accuracies."""
    from recipe.phimm.reward.asr_edge import measure
    from recipe.phimm.utils.shared import parse_asr_response

    extra_info = kwargs.get("extra_info") or {}
    tgt_lang = extra_info.get("language", kwargs.get("language", "English")).lower().strip()
    trans_dict = parse_asr_response(solution_str)
    hyp_text = trans_dict["text"]
    pred_lang = (trans_dict["lang"] or "").lower().strip()
    is_nonspeech = (hyp_text or "").strip().lower() == "<nonspeech>"

    unit = kwargs.pop("unit", "char").lower()
    error = measure(hyp_text, ground_truth, tgt_lang=tgt_lang, unit=unit, **kwargs)
    fmts = compute_fmt_acc(ground_truth or "", hyp_text or "")

    return {
        "char": error.accuracy(),
        **fmts,
        "lang": 1.0 if is_nonspeech else float(pred_lang == tgt_lang),
        "fmt": float(bool(trans_dict["formatted"])),
    }


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
    acc = clip(acc, 0.0, 1.0)
    return beta * signed_pow(acc, gamma)


def compute_score(solution_str, ground_truth, **kwargs):
    """Combined lexical + formatting reward.

    Uses :func:`recipe.phimm.reward.asr_edge.measure` for the lexical accuracy
    and :func:`compute_fmt_acc` for punctuation/capitalisation accuracy.

    Only the components listed in ``scores`` contribute to the reward. The
    contribution of each component ``k`` is::

        beta * signed_pow(clip(acc_k, 0, 1), gamma)

    Both ``beta`` and ``gamma`` default to ``1.0`` when omitted.

    Configuration example (YAML)::

        reward_kwargs:
          reduce: sum            # "sum" (default), "mean", or "multiply"
          scores:
            char: {beta: 1.0, gamma: 0.5}
            punc: {beta: 0.5, gamma: 0.2}
    """
    parsed = _parse_response(solution_str, ground_truth=ground_truth, **kwargs)
    is_good = parsed["fmt"] > 0
    
    measures = kwargs.get("measures") or {}
    reduce = kwargs.get("reduce", "sum").lower()
    gamma = float(kwargs.get("gamma", 1.0))
    scores = [scale_score(parsed.get(k, 1.0), cfg) for k, cfg in measures.items()]
    score = reduce_scores(scores, reduce)
    score = signed_pow(score, gamma)

    return {
        "score": score if is_good else -1.0,
        **parsed,
    }

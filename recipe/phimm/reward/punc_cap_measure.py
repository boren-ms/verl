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

def _is_punc_char(c: str) -> bool:
    """Return True if *c* is a Unicode punctuation character."""
    return unicodedata.category(c).startswith("P")


def _strip_punc(word: str) -> str:
    """Remove all Unicode punctuation characters from *word*."""
    return "".join(c for c in word if not _is_punc_char(c))


def _is_pure_punc(word: str) -> bool:
    """Return True if *word* consists entirely of punctuation characters."""
    return len(word) > 0 and all(_is_punc_char(c) for c in word)


def _punc_signature(word: str) -> str:
    """Return a string encoding punctuation positions & characters.

    E.g. ``"Hello,"`` → ``",5"`` (comma at index 5).
    Used to detect whether two tokens differ only in punctuation attachment.
    """
    return "".join(c + str(i) for i, c in enumerate(word) if _is_punc_char(c))


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

def compute_punc_cap_errors(ref: str, hyp: str) -> dict:
    """Compute punctuation and capitalisation error counts.

    Parameters
    ----------
    ref : str
        Reference (ground-truth) text — should contain punctuation & casing.
    hyp : str
        Hypothesis text from the ASR model.

    Returns
    -------
    dict with keys:

    * ``punc_errors``  – number of edits involving punctuation.
    * ``cap_errors``   – number of edits involving capitalisation.
    * ``lex_errors`` – number of edits involving a real word change.
    * ``total_errors`` – total edits (S + D + I), same as WER numerator.
    * ``n_ref``        – reference word count.
    * ``punc_error_rate`` – ``punc_errors / n_ref`` (0 when n_ref == 0).
    * ``cap_error_rate``  – ``cap_errors / n_ref``.
    * ``details``      – list of :class:`EditDetail` (one per edit).
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
                if _is_pure_punc(rw):
                    cats.add("punc")
                else:
                    cats.add("lex")
                    if any(_is_punc_char(c) for c in rw):
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
                if _is_pure_punc(hw):
                    cats.add("punc")
                else:
                    cats.add("lex")
                    if any(_is_punc_char(c) for c in hw):
                        cats.add("punc")
                details.append(EditDetail(op="ins", ref_word=None, hyp_word=hw, categories=cats))
                if "punc" in cats:
                    punc_err += 1
                if "lex" in cats:
                    lex_err += 1

    n_ref = len(ref_words)
    total_err = output.substitutions + output.deletions + output.insertions

    return {
        "punc_errors": punc_err,
        "cap_errors": cap_err,
        "lex_errors": lex_err,
        "total_errors": total_err,
        "n_ref": n_ref,
        "punc_error_rate": punc_err / n_ref if n_ref else 0.0,
        "cap_error_rate": cap_err / n_ref if n_ref else 0.0,
        "details": details,
    }


def _empty_result() -> dict:
    return {
        "punc_errors": 0,
        "cap_errors": 0,
        "lex_errors": 0,
        "total_errors": 0,
        "n_ref": 0,
        "punc_error_rate": 0.0,
        "cap_error_rate": 0.0,
        "details": [],
    }

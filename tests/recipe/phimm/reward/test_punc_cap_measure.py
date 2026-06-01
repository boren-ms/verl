"""Tests for recipe.phimm.reward.punc_cap_measure."""

import pytest

from recipe.phimm.reward.punc_cap_measure import (
    EditDetail,
    _is_pure_punc,
    _strip_punc,
    classify_edit,
    compute_punc_cap_errors,
)


# ── _strip_punc ──────────────────────────────────────────────────────────

class TestStripPunc:
    def test_no_punc(self):
        assert _strip_punc("hello") == "hello"

    def test_trailing_comma(self):
        assert _strip_punc("hello,") == "hello"

    def test_leading_quote(self):
        assert _strip_punc('"hello') == "hello"

    def test_surrounded(self):
        assert _strip_punc('"hello,"') == "hello"

    def test_apostrophe(self):
        assert _strip_punc("don't") == "dont"

    def test_pure_punc(self):
        assert _strip_punc("...") == ""

    def test_unicode_punc(self):
        # U+2014 EM DASH, U+201C LEFT DOUBLE QUOTATION MARK
        assert _strip_punc("\u201CHello\u2014") == "Hello"

    def test_empty(self):
        assert _strip_punc("") == ""


# ── _is_pure_punc ────────────────────────────────────────────────────────

class TestIsPurePunc:
    def test_period(self):
        assert _is_pure_punc(".")

    def test_ellipsis(self):
        assert _is_pure_punc("...")

    def test_mixed(self):
        assert not _is_pure_punc("a.")

    def test_word(self):
        assert not _is_pure_punc("hello")

    def test_empty(self):
        assert not _is_pure_punc("")


# ── classify_edit ─────────────────────────────────────────────────────────

class TestClassifyEdit:
    def test_punc_only_trailing(self):
        cats = classify_edit("Hello,", "Hello")
        assert cats == {"punc"}

    def test_punc_only_period(self):
        cats = classify_edit("world.", "world")
        assert cats == {"punc"}

    def test_cap_only(self):
        cats = classify_edit("Hello", "hello")
        assert cats == {"cap"}

    def test_cap_only_uppercase(self):
        cats = classify_edit("the", "The")
        assert cats == {"cap"}

    def test_punc_and_cap(self):
        cats = classify_edit("Hello,", "hello")
        assert cats == {"punc", "cap"}

    def test_lexical(self):
        cats = classify_edit("cat", "dog")
        assert cats == {"lex"}

    def test_lexical_with_punc_diff(self):
        cats = classify_edit("cat,", "dog")
        assert "lex" in cats
        assert "punc" in cats

    def test_same_word(self):
        # Identical words should never reach classify_edit in practice
        # (they would be "equal"), but if they do, it should return punc
        # as a fallback.
        cats = classify_edit("hello", "hello")
        assert "lex" not in cats


# ── compute_punc_cap_errors ──────────────────────────────────────────────

class TestComputePuncCapErrors:
    def test_identical(self):
        r = compute_punc_cap_errors("Hello world", "Hello world")
        assert r["punc_errors"] == 0
        assert r["cap_errors"] == 0
        assert r["lex_errors"] == 0
        assert r["total_errors"] == 0
        assert r["n_ref"] == 2

    def test_pure_punc_diffs(self):
        r = compute_punc_cap_errors("Hello, world.", "Hello world")
        assert r["punc_errors"] == 2
        assert r["cap_errors"] == 0
        assert r["lex_errors"] == 0

    def test_pure_cap_diffs(self):
        r = compute_punc_cap_errors("The Quick Brown Fox", "the quick brown fox")
        assert r["cap_errors"] == 4  # The→the, Quick→quick, Brown→brown, Fox→fox
        assert r["punc_errors"] == 0
        assert r["lex_errors"] == 0

    def test_mixed_punc_cap_lexical(self):
        # jiwer alignment may not pair Hello,→hello (it can choose any
        # minimum-cost alignment), so only assert on totals.
        r = compute_punc_cap_errors("Hello, World.", "hello world cat")
        assert r["punc_errors"] >= 1  # at least one punc-bearing word mismatched
        assert r["total_errors"] == 3

    def test_empty_ref_and_hyp(self):
        r = compute_punc_cap_errors("", "")
        assert r["punc_errors"] == 0
        assert r["cap_errors"] == 0
        assert r["n_ref"] == 0
        assert r["punc_error_rate"] == 0.0
        assert r["cap_error_rate"] == 0.0

    def test_empty_hyp(self):
        r = compute_punc_cap_errors("Hello world", "")
        assert r["total_errors"] == 2  # two deletions
        assert r["n_ref"] == 2

    def test_empty_ref(self):
        r = compute_punc_cap_errors("", "Hello world")
        assert r["total_errors"] == 2  # two insertions
        assert r["n_ref"] == 0

    def test_none_inputs(self):
        r = compute_punc_cap_errors(None, None)
        assert r["punc_errors"] == 0
        assert r["n_ref"] == 0

    def test_inserted_punc_token(self):
        r = compute_punc_cap_errors("Hello world", "Hello , world")
        # The comma is an inserted pure-punctuation token.
        assert r["punc_errors"] >= 1

    def test_deleted_punc_token(self):
        # Standalone period as a separate token in ref.
        r = compute_punc_cap_errors("Hello world .", "Hello world")
        assert r["punc_errors"] >= 1

    def test_error_rates(self):
        r = compute_punc_cap_errors("Hello, World.", "hello world")
        # 2 ref words, punc_errors=2 (comma+period), cap_errors=2 (Hello→hello, World→world)
        assert r["punc_error_rate"] == pytest.approx(1.0)   # 2/2
        assert r["cap_error_rate"] == pytest.approx(1.0)     # 2/2

    def test_details_populated(self):
        r = compute_punc_cap_errors("Hello, world.", "hello world")
        assert len(r["details"]) > 0
        for d in r["details"]:
            assert isinstance(d, EditDetail)
            assert d.op in ("sub", "ins", "del")
            assert len(d.categories) > 0

    def test_apostrophe_contraction(self):
        r = compute_punc_cap_errors("don't", "dont")
        assert r["punc_errors"] == 1
        assert r["lex_errors"] == 0

    def test_multi_sentence(self):
        ref = "The cat sat on the mat. It was happy."
        hyp = "the cat sat on the mat it was happy"
        r = compute_punc_cap_errors(ref, hyp)
        # "The" → "the" (cap), "mat." → "mat" (punc), "happy." → "happy" (punc)
        assert r["cap_errors"] >= 1
        assert r["punc_errors"] >= 2


# ── _compute_fmt_errors_lite (integration) ───────────────────────────────

class TestFmtErrorsLiteIntegration:
    def test_basic(self):
        from recipe.phimm.reward.asr_edge import _compute_fmt_errors_lite

        punc, cap = _compute_fmt_errors_lite("Hello, World.", "hello world")
        assert punc == 2
        assert cap == 2

    def test_no_errors(self):
        from recipe.phimm.reward.asr_edge import _compute_fmt_errors_lite

        punc, cap = _compute_fmt_errors_lite("hello world", "hello world")
        assert punc == 0
        assert cap == 0

    def test_none_inputs(self):
        from recipe.phimm.reward.asr_edge import _compute_fmt_errors_lite

        punc, cap = _compute_fmt_errors_lite(None, None)
        assert punc == 0
        assert cap == 0

"""Tests for recipe.phimm.reward.punc_cap_measure."""

import pytest

from recipe.phimm.reward.punc_cap_measure import (
    check_fmt,
    _is_pure_punc,
    _strip_punc,
    classify_edit,
    compute_fmt_acc,
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
        assert classify_edit("Hello,", "Hello") == "punc"

    def test_punc_only_period(self):
        assert classify_edit("world.", "world") == "punc"

    def test_cap_only(self):
        assert classify_edit("Hello", "hello") == "cap"

    def test_cap_only_uppercase(self):
        assert classify_edit("the", "The") == "cap"

    def test_punc_and_cap_prefers_punc(self):
        # Priority: lex > punc > cap → punc wins over cap.
        assert classify_edit("Hello,", "hello") == "punc"

    def test_lexical(self):
        assert classify_edit("cat", "dog") == "lex"

    def test_lexical_with_punc_diff_prefers_lex(self):
        # Priority: lex > punc > cap → lex wins over punc.
        assert classify_edit("cat,", "dog") == "lex"

    def test_same_word(self):
        # Identical words should never reach classify_edit in practice
        # (they would be "equal"), but if they do, fallback is "punc".
        assert classify_edit("hello", "hello") == "punc"


# ── compuate_fmt_acc ──────────────────────────────────────────────

class TestComputeFmtAcc:
    def test_identical(self):
        r = compute_fmt_acc("Hello world", "Hello world")
        assert r["punc"] == 1.0
        assert r["cap"] == 1.0
        assert r["lex"] == 1.0

    def test_pure_punc_diffs(self):
        # Both ref words carry punctuation and both are wrong → punc acc 0.
        r = compute_fmt_acc("Hello, world.", "Hello world")
        assert r["punc"] == pytest.approx(0.0)
        assert r["cap"] == 1.0
        assert r["lex"] == 1.0

    def test_pure_cap_diffs(self):
        r = compute_fmt_acc("The Quick Brown Fox", "the quick brown fox")
        assert r["cap"] == pytest.approx(0.0)
        assert r["punc"] == 1.0
        assert r["lex"] == 1.0

    def test_lexical_error(self):
        # One lexical substitution out of three reference words.
        r = compute_fmt_acc("the cat sat", "the dog sat")
        assert r["lex"] == pytest.approx(2.0 / 3.0)

    def test_empty_ref_and_hyp(self):
        r = compute_fmt_acc("", "")
        assert r["punc"] == 1.0
        assert r["cap"] == 1.0
        assert r["lex"] == 1.0

    def test_empty_hyp(self):
        # Both reference words deleted → lexical accuracy 0.
        r = compute_fmt_acc("Hello world", "")
        assert r["lex"] == pytest.approx(0.0)

    def test_empty_ref(self):
        # No reference words → punc/cap have no events (default 1.0); lex
        # gets 2 insertion errors out of 2 (denominator = hits+errs = 0+2).
        r = compute_fmt_acc("", "Hello world")
        assert r["punc"] == 1.0
        assert r["cap"] == 1.0
        assert r["lex"] == pytest.approx(0.0)

    def test_none_inputs(self):
        r = compute_fmt_acc(None, None)
        assert r["punc"] == 1.0
        assert r["cap"] == 1.0
        assert r["lex"] == 1.0

    def test_accuracies_upper_bounded(self):
        # Raw accuracies are not clipped here (clipping happens in
        # compute_score); they are always <= 1.0 but may go negative
        # when errors exceed the reference count.
        r = compute_fmt_acc("Hello, World.", "hello world cat dog")
        for k in ("punc", "cap", "lex"):
            assert r[k] <= 1.0


# ── check_fmt ─────────────────────────────────────────────────────────────


class TestCheckFmt:
    def test_accepts_asr(self):
        s = "Audio Language: English.\n<ASR><lang=English><TXT>Hello world</TXT></ASR>"
        assert check_fmt(s)

    def test_accepts_asr_star(self):
        s = (
            "Audio Language: English.\n"
            "<ASR_LEXICAL><lang=English><TXT>Hello world</TXT></ASR_LEXICAL>"
        )
        assert check_fmt(s)

    def test_rejects_lang_mismatch(self):
        s = "Audio Language: English.\n<ASR><lang=French><TXT>Hello world</TXT></ASR>"
        assert not check_fmt(s)

    def test_rejects_unknown_tag(self):
        s = "Audio Language: English.\n<XYZ><lang=English><TXT>Hello world</TXT></XYZ>"
        assert not check_fmt(s)

    def test_rejects_mismatched_closing_tag(self):
        s = "Audio Language: English.\n<ASR><lang=English><TXT>Hello world</TXT></ASR_LEXICAL>"
        assert not check_fmt(s)

    def test_accepts_missing_period_in_header(self):
        s = "Audio Language: English\n<ASR><lang=English><TXT>Hello world</TXT></ASR>"
        assert check_fmt(s)

    def test_rejects_malformed_txt_tags(self):
        s = "Audio Language: English.\n<ASR><lang=English><TXT>Hello world</TXT><TXT></ASR>"
        assert not check_fmt(s)

    def test_rejects_trailing_text_after_closing_tag(self):
        s = "Audio Language: English.\n<ASR><lang=English><TXT>Hello world</TXT></ASR>...adafda "
        assert not check_fmt(s)

    def test_rejects_non_string(self):
        assert not check_fmt(None)

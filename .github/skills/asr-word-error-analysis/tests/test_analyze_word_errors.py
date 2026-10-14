import importlib.util
import sys
from pathlib import Path

import pytest

from recipe.phimm.reward.asr_eval import openasr_en_eval
from recipe.phimm.utils.open_asr_normalizer.hf_english_normalizer import (
    _HFEnglishTextNormalizer,
)


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "analyze_word_errors.py"
SPEC = importlib.util.spec_from_file_location("asr_word_error_analyzer", SCRIPT_PATH)
assert SPEC and SPEC.loader
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


def test_default_normalizer_matches_openasr_en_eval(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["analyze_word_errors.py", "--input-path", "input.jsonl"])

    assert analyzer.parse_args().normalizer == "openasr_en"


@pytest.mark.parametrize(
    ("ref", "hyp", "expected_errors", "expected_ref_words"),
    [
        ("ice cream", "icecream", 0, 2),
        ("icecream", "ice cream", 0, 1),
        ("new york city", "newyorkcity", 0, 3),
        ("ice cream is good", "icecream was good", 1, 4),
        ("", "hello", 1, 1),
    ],
)
def test_openasr_en_alignment_matches_reward(ref, hyp, expected_errors, expected_ref_words):
    ref_norm, hyp_norm = analyzer.normalize_pair(ref, hyp, {}, "openasr_en", "", "language")
    alignment = analyzer.align_openasr_en(ref_norm, hyp_norm)
    reward = openasr_en_eval(hyp, ref)

    assert ref_norm == _HFEnglishTextNormalizer()(ref.strip())
    assert hyp_norm == _HFEnglishTextNormalizer()(hyp.strip())
    assert alignment.errors == expected_errors == reward["n_err"]
    assert alignment.ref_words == expected_ref_words == reward["n_ref"]
    assert alignment.wer == reward["wer"]
    assert sum(op.op == "sub" for op in alignment.ops) == alignment.substitutions
    assert sum(op.op == "del" for op in alignment.ops) == alignment.deletions
    assert sum(op.op == "ins" for op in alignment.ops) == alignment.insertions


def test_openasr_en_aggregate_uses_per_utterance_empty_ref_denominator():
    results = analyzer.analyze_all(
        [{"ref": "", "hyp": "hello"}, {"ref": "ice cream", "hyp": "icecream"}],
        "ref",
        "hyp",
        "",
        False,
        "openasr_en",
        "",
        "language",
    )

    aggregate = analyzer.aggregate(results)

    assert aggregate["total_ref_words"] == 3
    assert aggregate["total_errors"] == 1
    assert aggregate["wer"] == pytest.approx(1 / 3)


def test_segmented_rows_are_coalesced_like_long_audio_reward_manager():
    rows = [
        {
            "gts": "hello world",
            "clean_output": "world",
            "output": "raw world",
            "parent_audio_path": "parent.wav",
            "audio_path": "parent.wav#10:20",
            "seg_start": 10.0,
            "n_segments": 2,
        },
        {
            "gts": "hello world",
            "clean_output": "hello",
            "output": "raw hello",
            "parent_audio_path": "parent.wav",
            "audio_path": "parent.wav#0:10",
            "seg_start": 0.0,
            "n_segments": 2,
        },
    ]
    analyzer.remap_verl_schema(rows, "ref", "hyp")

    combined, changed = analyzer.coalesce_segmented_rows(rows, "ref", "hyp", "output")

    assert changed is True
    assert len(combined) == 1
    assert combined[0]["hyp"] == "hello\nworld"
    assert combined[0]["output"] == "raw hello\n\nraw world"
    assert combined[0]["audio_path"] == "parent.wav"
    assert combined[0]["segment_count"] == 2


def test_compact_alignment_keeps_five_words_around_error_and_full_alignment():
    alignment = analyzer.align(
        "l0 l1 l2 l3 l4 l5 l6 wrong r0 r1 r2 r3 r4 r5 r6",
        "l0 l1 l2 l3 l4 l5 l6 fixed r0 r1 r2 r3 r4 r5 r6",
    )

    compact = analyzer._render_alignment_html(alignment, context_words=5)
    full = analyzer._render_alignment_html(alignment)

    assert compact.count('class="context-ellipsis"') == 2
    assert "l0" not in compact and "l1" not in compact and "l2" in compact
    assert "r4" in compact and "r5" not in compact and "r6" not in compact
    assert all(word in full for word in ("l0", "l1", "r5", "r6"))


def test_compact_alignment_merges_overlapping_error_contexts():
    alignment = analyzer.align(
        "l0 l1 l2 l3 l4 l5 l6 bad0 m0 m1 m2 bad1 r0 r1 r2 r3 r4 r5 r6",
        "l0 l1 l2 l3 l4 l5 l6 fix0 m0 m1 m2 fix1 r0 r1 r2 r3 r4 r5 r6",
    )

    compact = analyzer._render_alignment_html(alignment, context_words=5)

    assert compact.count('class="context-ellipsis"') == 2
    assert all(f">{word}</span>" in compact for word in ("m0", "m1", "m2"))


@pytest.mark.parametrize(
    ("ref", "hyp", "error_class"),
    [
        ("a b c d e f g", "a b c inserted d e f g", "err-ins"),
        ("a b c deleted d e f g", "a b c d e f g", "err-del"),
    ],
)
def test_compact_alignment_preserves_insertions_and_deletions(ref, hyp, error_class):
    compact = analyzer._render_alignment_html(analyzer.align(ref, hyp), context_words=1)

    assert error_class in compact


def test_alignment_html_escapes_words_and_keeps_zero_error_text():
    alignment = analyzer.AlignResult(
        ops=[analyzer.AlignOp("ok", "<tag>", "<tag>")],
        ref_words=1,
        hyp_words=1,
    )

    rendered = analyzer._render_alignment_html(alignment, context_words=5)

    assert "&lt;tag&gt;" in rendered
    assert 'class="aligned-word"' in rendered


def test_alignment_html_links_selected_substitution_words():
    rendered = analyzer._render_alignment_html(analyzer.align("ice cream", "nice cream"))

    assert rendered.count('data-align-id="0"') == 2
    assert '>ice</span>' in rendered
    assert '[→nice]</span>' in rendered


def test_report_defaults_to_compact_alignment_with_full_disclosure(tmp_path):
    alignment = analyzer.align(
        "l0 l1 l2 l3 l4 l5 l6 wrong r0 r1 r2 r3 r4 r5 r6",
        "l0 l1 l2 l3 l4 l5 l6 fixed r0 r1 r2 r3 r4 r5 r6",
    )
    result = analyzer.UtteranceResult(
        idx=0,
        utt_id="utt-1",
        ref="ref",
        hyp="hyp",
        ref_norm="ref",
        hyp_norm="hyp",
        alignment=alignment,
        row={},
    )

    analyzer.render_html(
        [result],
        analyzer.aggregate([result]),
        tmp_path,
        "test",
        "input.jsonl",
        top_n=1,
        context_words=5,
    )
    report = (tmp_path / "report.html").read_text()

    assert "Transcript with error highlighting (5-word context)" in report
    assert "Show full alignment" in report
    assert 'class="context-ellipsis"' in report
    assert "selectAlignedWords" in report
    assert ".aligned-word.is-selected" in report
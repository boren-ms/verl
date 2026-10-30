import pytest

from scripts.reconcile_openlong_qwen_alignment import qwen_english_tokens, reconcile_parent


def segment(index, start, end, text):
    return {"id": f"s{index}", "seg_index": index, "n_segments": 2,
            "seg_start": start, "seg_end": end, "text": text,
            "baseline_hypothesis": text, "baseline_n_err": 0,
            "baseline_n_ref": 1, "baseline_wer": 0.0}


def window(text, words, status="ok"):
    return {"window": {"seed_text": text, "core_segment_ids": ["s0", "s1"],
                       "core_segment_indexes": [0, 1], "core_word_start": 0,
                       "core_word_end": len(text.split()), "audio_start": 0, "audio_end": 20},
            "status": status, "words": words}


def timed(word, start, end):
    return {"text": word, "parent_start": start, "parent_end": end}


def test_assigns_overlap_word_to_both_audio_clips():
    rows = [segment(0, 0, 10, "hello shared"), segment(1, 9, 20, "world")]
    result = window("hello shared world", [timed("hello", 1, 1.2),
                                            timed("shared", 9.4, 9.6), timed("world", 12, 12.2)])
    output, qa = reconcile_parent(rows, [result])
    assert [row["text"] for row in output] == ["hello shared", "shared world"]
    assert qa["overlap_words"] == 1 and qa["needs_review"] == 0
    assert all(row["baseline_hypothesis"] for row in output)


def test_untimed_token_is_retained_and_flagged():
    rows = [segment(0, 0, 10, "hello ---"), segment(1, 9, 20, "world")]
    result = window("hello --- world", [timed("hello", 1, 1.2), timed("world", 12, 12.2)])
    output, qa = reconcile_parent(rows, [result])
    assert output[0]["text"] == "hello ---"
    assert "untimed_token" in output[0]["alignment_flags"]
    assert qa["needs_review"] == 1


def test_missing_or_unexpected_results_fail_closed():
    rows = [segment(0, 0, 10, "hello"), segment(1, 9, 20, "world")]
    with pytest.raises(ValueError, match="Missing core"):
        reconcile_parent(rows, [])
    with pytest.raises(ValueError, match="tokenization"):
        reconcile_parent(rows, [window("hello world", [timed("wrong", 1, 2)])])


def test_qwen_english_tokenization_preserves_original_word_mapping():
    assert qwen_english_tokens("can't,") == ["can't"]
    assert qwen_english_tokens("foo中文bar") == ["foo", "中", "文", "bar"]
    assert qwen_english_tokens("---") == []


def test_collapsed_window_keeps_seed_and_requires_review():
    rows = [segment(0, 0, 10, "hello"), segment(1, 9, 20, "world")]
    result = window("hello world", [timed("hello", 0, 0), timed("world", 0, 0)])
    result["quality"] = {"collapsed": True}
    output, qa = reconcile_parent(rows, [result])
    assert [row["text"] for row in output] == ["hello", "world"]
    assert all("collapsed_qwen_timestamps" in row["alignment_flags"] for row in output)
    assert qa["needs_review"] == 2

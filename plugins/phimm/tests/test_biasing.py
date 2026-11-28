# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Static candidates, complete corpus sampling, and prompt placement contracts."""

import json
import random
from dataclasses import replace

import pytest

from recipe.phimm.data import prompts
from recipe.phimm.data.biasing import (
    BiasingConfig,
    BiasingSimulator,
    NoEligibleKeywords,
    matching_keywords,
    normalize_phrase,
)
from recipe.phimm.data.prompts import BIAS_INSTRUCTION, get_task_prompt
from recipe.phimm.data.simulate_biasing import main, read_jsonl, read_librispeech_tsv, write_manifest


@pytest.mark.parametrize("style", ["verb", "lex", "display"])
@pytest.mark.parametrize("lang_hint", [False, True])
def test_positions_change_only_bias_placement(style, lang_hint):
    words = ["Seattle", "New York"]
    plain = get_task_prompt(style, "en", lang_hint=lang_hint)
    before = get_task_prompt(style, "en", lang_hint=lang_hint, biasing="before_audio", biasing_words=words)
    after = get_task_prompt(style, "en", lang_hint=lang_hint, biasing="after_audio", biasing_words=words)
    bias = f"{BIAS_INSTRUCTION} *Seattle*, *New York*."
    assert before.count("<audio>") == after.count("<audio>") == 1
    assert before.index(bias) < before.index("<audio>")
    assert after.index(bias) > after.index("<audio>")
    assert before.replace(" " + bias, "") == plain
    assert after.removesuffix((" " if lang_hint else "\n") + bias) == plain
    assert get_task_prompt(style, "en", lang_hint=lang_hint, biasing=None, biasing_words=words) == plain


@pytest.mark.parametrize("position", ["before_audio", "after_audio"])
@pytest.mark.parametrize("words", [[], None])
def test_empty_candidates_keep_normal_prompt(position, words):
    assert get_task_prompt(biasing=position, biasing_words=words) == get_task_prompt()


def test_no_biasing_skips_context_helper(monkeypatch):
    plain = get_task_prompt()

    def unexpected_call(*args, **kwargs):
        pytest.fail("No biasing must skip add_biasing_context")

    monkeypatch.setattr(prompts, "add_biasing_context", unexpected_call)
    assert get_task_prompt(biasing_words=["Seattle"]) == plain
    assert get_task_prompt(biasing=None, biasing_words=["Seattle"]) == plain
    for position in ("before_audio", "after_audio"):
        assert get_task_prompt(biasing=position, biasing_words=None) == plain
        assert get_task_prompt(biasing=position, biasing_words=[]) == plain


@pytest.mark.parametrize("words", ["Seattle", [""], [42], ["<audio>"]])
def test_invalid_candidates_rejected(words):
    with pytest.raises(ValueError, match="biasing_words"):
        get_task_prompt(biasing="before_audio", biasing_words=words)


@pytest.mark.parametrize("position", ["none", "befor_audio"])
def test_unknown_position_rejected(position):
    with pytest.raises(ValueError, match="biasing position"):
        get_task_prompt(biasing=position, biasing_words=["Seattle"])


def test_ground_truth_matches_replay_phrases_without_subword_false_hits():
    assert matching_keywords(
        "We use Azure, not Seattle. New York is nice.", ["azure", "seat", "Seattle", "New York", "Yorkshire"]
    ) == ["azure", "Seattle", "New York"]
    assert matching_keywords("我们去北京学习", ["北京", "上海", "学习"]) == ["北京", "学习"]
    assert matching_keywords("can't", ["can", "can't"]) == ["can't"]


def full_config():
    return BiasingConfig()


def test_training_keywords_are_exact_hits_and_negatives_do_not_add_extra_hits():
    sampler = BiasingSimulator(replace(full_config(), hit_ratio=(1, 1)), seed=8)
    sampler.prepare_word_pool(["Seattle", "New", "York", "London", "Paris", "Dublin"])
    words, keywords = sampler.sample("Seattle Seattle New York")
    assert keywords == matching_keywords("Seattle Seattle New York", words)
    assert {normalize_phrase(word) for word in keywords} == {"seattle", "new", "york"}
    assert len(words) == len({normalize_phrase(word) for word in words})


def test_rlbr_coverage_counts_unique_usable_transcript_words():
    usable = [f"rare{index}" for index in range(20)]
    text = "the " + " ".join(usable + usable)
    sampler = BiasingSimulator(replace(full_config(), hit_ratio=(0.9, 0.9)), seed=8, common_words=["the"])
    sampler.prepare_word_pool(usable + [f"negative{index}" for index in range(1000)])
    for _ in range(20):
        words, keywords = sampler.sample(text)
        assert len(keywords) == 18
        assert {normalize_phrase(word) for word in keywords} <= set(usable)
        assert keywords == matching_keywords(text, words)
        assert "the" not in {normalize_phrase(word) for word in words}


def test_new_coverage_spans_ten_to_one_hundred_percent_and_keeps_candidates_unique():
    usable = [f"rare{index}" for index in range(100)]
    sampler = BiasingSimulator(replace(full_config()), seed=42, common_words=["the"])
    sampler.prepare_word_pool(usable + [f"negative{index}" for index in range(2000)])
    text = "the " + " ".join(usable + usable)
    hit_counts = []
    for _ in range(100):
        words, keywords = sampler.sample(text)
        hit_counts.append(len(keywords))
        assert 10 <= len(keywords) <= 100
        assert len(keywords) <= len(words) <= 1000
        assert len(words) == len({normalize_phrase(word) for word in words})
        assert keywords == matching_keywords(text, words)
    assert min(hit_counts) < 20
    assert max(hit_counts) > 90


def test_negative_candidates_are_unique_vocabulary_words_with_zero_actual_hits():
    sampler = BiasingSimulator(replace(full_config(), negative_only=True), seed=42, common_words=["the"])
    sampler.prepare_word_pool(["the", "Seattle", "SEATTLE", "New York", "rare"] + [f"negative{i}" for i in range(1200)])
    counts = []
    for text in ("the", "SEATTLE and new york rare", "", "Seattle Seattle"):
        for _ in range(25):
            words, keywords = sampler.sample(text)
            counts.append(len(words))
            assert 1 <= len(words) <= 1000
            assert len(words) == len({normalize_phrase(word) for word in words})
            assert "the" not in words
            assert keywords == matching_keywords(text, words) == []
    assert len(set(counts)) > 50


def test_no_possible_hits_are_saved_separately(tmp_path):
    sampler = BiasingSimulator(replace(full_config()), common_words=["the"])
    sampler.prepare_word_pool(["Seattle", "London"])
    with pytest.raises(NoEligibleKeywords):
        sampler.sample("the")
    path = tmp_path / "data.jsonl"
    result = write_manifest([{"text": "the"}, {"text": "Seattle"}], path, simulator=sampler)
    assert result["source_rows"] == 2
    assert result["rows"] == result["rejected_no_keywords"] == 1
    assert next(read_jsonl(tmp_path / "rejected_no_keywords.jsonl"))["text"] == "the"


def test_seed_reproduces_candidates_without_touching_global_randomness():
    rows = [{"text": "Contoso Seattle New York", "language": "en"}] * 20
    first = BiasingSimulator(full_config(), seed=7)
    second = BiasingSimulator(full_config(), seed=7)
    state = random.getstate()
    first.prepare_word_pool([row["text"] for row in rows])
    second.prepare_word_pool([row["text"] for row in rows])
    assert [first.sample(row["text"]) for row in rows] == [second.sample(row["text"]) for row in rows]
    assert random.getstate() == state


def test_full_sampling_keeps_one_candidate_for_a_single_rare_word():
    sampler = BiasingSimulator(full_config(), seed=2)
    sampler.prepare_word_pool([row["text"] for row in [{"text": "Contoso", "language": "en"}]])
    assert sampler.sample("Contoso") == (["Contoso"], ["Contoso"])


def test_offline_manifest_preserves_text_metadata_and_fixed_annotations(tmp_path):
    row = {
        "id": "a",
        "text": "We use Seattle.",
        "audio_path": "file.flac",
        "metadata": "keep",
        "prompt": "obsolete source prompt",
        "distractors": ["Seattle", "London"],
        "ground_truth": ["Seattle"],
    }
    output = tmp_path / "data.jsonl"
    result = write_manifest([row], output, fixed=True)
    saved = list(read_jsonl(output))[0]
    assert saved["text"] == row["text"]
    assert saved["metadata"] == "keep"
    assert "prompt" not in saved
    assert saved["biasing_words"] == row["distractors"]
    assert saved["keywords"] == row["ground_truth"]
    assert saved["gt_output"].endswith(row["text"])
    assert result["rows"] == result["with_biasing"] == 1
    with pytest.raises(FileExistsError):
        write_manifest([row], output, fixed=True)


def test_invalid_offline_input_does_not_leave_partial_manifest(tmp_path):
    path = tmp_path / "data.jsonl"
    with pytest.raises(ValueError, match="Fixed candidate"):
        write_manifest([{"text": "hello"}], path, fixed=True)
    assert not path.exists()
    assert not path.with_suffix(".jsonl.tmp").exists()


def test_rlbr_tsv_conversion_retains_exact_reference_and_remaps_audio(tmp_path):
    path = tmp_path / "source.tsv"
    original = "/root/data/LibriSpeech/train-clean-100/1/2/1-2-3.flac"
    text = "don't  change punctuation!"
    conversation = [{"messages": [{"role": "assistant", "content": text}]}]
    path.write_text(f"id\t{[original]!r}\t{conversation!r}\n")
    row = next(read_librispeech_tsv(path, "az://bucket/data/LibriSpeech"))
    assert row["text"] == text
    assert row["audio_path"] == "az://bucket/data/LibriSpeech/train-clean-100/1/2/1-2-3.flac"
    assert row["source_audio_path"] == original


@pytest.mark.parametrize("negative_only", [False, True])
def test_cli_generates_positive_or_full_negative_data(tmp_path, monkeypatch, negative_only):
    source = tmp_path / "source.jsonl"
    texts = ["the", "Seattle London", ""]
    source.write_text("".join(json.dumps({"id": str(i), "text": text}) + "\n" for i, text in enumerate(texts)))
    vocabulary = tmp_path / "vocabulary.txt"
    vocabulary.write_text("the 100\nSeattle 10\nLondon 5\nParis 4\nPARIS 3\nDublin 2\n")
    output = tmp_path / "prepared"
    argv = [
        "simulate_biasing",
        "--input-manifest",
        str(source),
        "--output-dir",
        str(output),
        "--candidate-word-file",
        str(vocabulary),
        "--common-word-file",
        str(vocabulary),
        "--common-word-num",
        "1",
    ]
    monkeypatch.setattr("sys.argv", argv + (["--negative-only"] if negative_only else []))
    main()
    summary = json.loads((output / "preparation.json").read_text())
    assert summary["sampler"]["negative_only"] is negative_only
    assert summary["rows"] == summary["with_biasing"] == (3 if negative_only else 1)
    assert summary["with_keywords"] == (0 if negative_only else 1)
    for row in read_jsonl(output / "data.jsonl"):
        assert row["keywords"] == matching_keywords(row["text"], row["biasing_words"])
        assert bool(row["keywords"]) is not negative_only
        assert "prompt" not in row
    if negative_only:
        assert not (output / "rejected_no_keywords.jsonl").exists()
    else:
        assert summary["rejected_no_keywords"] == 2


@pytest.mark.parametrize(
    "kwargs", [{"max_num": 0}, {"max_num": 1001}, {"hit_ratio": (0, 1)}, {"hit_ratio": (0.8, 0.2)}]
)
def test_invalid_sampler_config_rejected(kwargs):
    with pytest.raises(ValueError):
        BiasingConfig(**kwargs)


def test_vocabulary_deduplicates_and_filters_common_words():
    sampler = BiasingSimulator(common_words=["the"])
    sampler.prepare_word_pool(["Seattle", "Seattle", "seattle,", "London", "the"])
    assert sampler.pool == ["Seattle", "London"]
    with pytest.raises(ValueError, match="no usable candidates"):
        sampler.prepare_word_pool(["the"])

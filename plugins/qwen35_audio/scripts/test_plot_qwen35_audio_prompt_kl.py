import json
import math
from types import SimpleNamespace

import pytest

from plot_qwen35_audio_prompt_kl import (
    DEFAULT_INSTRUCTION,
    DEFAULT_KEYWORDS,
    DEFAULT_TRANSCRIPTION,
    build_teacher_instruction,
    build_chat_prefix,
    compute_k2_estimates,
    compute_k3_estimates,
    extract_chosen_logprobs,
    extract_suffix_logprobs,
    normalized_words,
    student_response_ticks,
    transcript_token_fragments,
    validate_keywords,
    visible_token,
    write_outputs,
)


def test_build_teacher_instruction():
    assert build_teacher_instruction(DEFAULT_INSTRUCTION, ["Ada", "New York"]) == (
        DEFAULT_INSTRUCTION
        + "\nPay extra attention to the following phrases/words: *Ada*, *New York*."
    )


def test_build_chat_prefix_uses_generation_prompt():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "instruction"}]
            assert kwargs == {"add_generation_prompt": True, "tokenize": False}
            return "rendered"

    assert build_chat_prefix(Tokenizer(), "instruction") == "rendered"


def test_validate_keywords_normalizes_case_and_whitespace():
    assert validate_keywords("Meet Ada in New   York", [" ada ", "New York"]) == ["ada", "New York"]


def test_default_keywords_come_from_sample_transcription():
    assert validate_keywords(DEFAULT_TRANSCRIPTION, DEFAULT_KEYWORDS) == DEFAULT_KEYWORDS


def test_validate_keywords_rejects_missing_keyword():
    with pytest.raises(ValueError, match="Grace"):
        validate_keywords("Meet Ada", ["Grace"])


def test_extract_suffix_logprobs():
    prompt_token_ids = [10, 11, 20, 21]
    prompt_logprobs = [
        None,
        {11: -0.1},
        {20: SimpleNamespace(logprob=-0.2)},
        {21: SimpleNamespace(logprob=-0.3)},
    ]
    assert extract_suffix_logprobs(prompt_token_ids, prompt_logprobs, [20, 21]) == [-0.2, -0.3]


def test_extract_suffix_logprobs_rejects_misalignment():
    with pytest.raises(ValueError, match="final prompt-token suffix"):
        extract_suffix_logprobs([10, 20, 22], [None, {20: -0.2}, {22: -0.3}], [20, 21])


def test_extract_chosen_logprobs_from_student_completion():
    token_logprobs = [
        {20: SimpleNamespace(logprob=-0.2)},
        {21: SimpleNamespace(logprob=-0.3)},
    ]
    assert extract_chosen_logprobs([20, 21], token_logprobs, "student response") == [-0.2, -0.3]


def test_extract_chosen_logprobs_rejects_missing_score():
    with pytest.raises(ValueError, match="student response position 0"):
        extract_chosen_logprobs([20], [{21: -0.2}], "student response")


def test_compute_k3_estimates_matches_repository_formula():
    values = compute_k3_estimates([-2.0, -1.0], [-1.0, -1.0])
    assert values == pytest.approx([math.e - 2.0, 0.0])
    assert compute_k3_estimates([-30.0], [0.0]) == [10.0]


def test_compute_k2_estimates_matches_repository_formula():
    assert compute_k2_estimates([-2.0, -1.0], [-1.0, -1.0]) == [0.5, 0.0]


def test_visible_token():
    assert visible_token(" hello\n") == r"\shello\n"
    assert visible_token("") == "<empty>"


def test_transcript_token_fragments_exclude_structured_wrapper():
    token_text = ["<", "TXT", ">Ste", "ph", "anos", ".</", "TXT", ">"]
    assert transcript_token_fragments(token_text) == [
        (2, "Ste"),
        (3, "ph"),
        (4, "anos"),
        (5, "."),
    ]


def test_normalized_words_ignores_case_and_punctuation():
    assert normalized_words("Stephanos Dedalos.") == normalized_words("STEPHANOS DEDALOS")


def test_student_response_ticks_group_subtokens_into_words():
    positions, labels = student_response_ticks(["St", "ef", "ano", " St", "url", "a", "."])
    assert positions == [1.0, 4.5]
    assert labels == ["Stefano", "Sturla."]


def test_write_outputs_creates_png_and_json(tmp_path):
    output_path = tmp_path / "report.png"
    report = {
        "transcription": "hello world",
        "keywords": ["hello"],
        "tokens": {
            "text": ["<TXT>", "hello", " world", "</TXT>"],
            "student_logprob": [-0.1, -1.0, -0.5, -0.1],
            "teacher_logprob": [-0.1, -2.0, -0.25, -0.1],
            "teacher_minus_student_logprob": [0.0, -0.5, 0.1, 0.0],
            "k3_estimate": [0.0, 0.1, 0.01, 0.0],
        },
        "summary": {"k3_sum": 0.3, "k3_mean": 0.15},
    }
    json_path = write_outputs(output_path, report)

    assert output_path.stat().st_size > 0
    output_tokens = json.loads(json_path.read_text())["tokens"]
    assert output_tokens["k3_estimate"] == [0.0, 0.1, 0.01, 0.0]
    assert output_tokens["k2_estimate"] == [0.0, 0.5, 0.03125, 0.0]
    assert output_tokens["student_probability"] == pytest.approx(
        [math.exp(-0.1), math.exp(-1.0), math.exp(-0.5), math.exp(-0.1)]
    )
    assert output_tokens["teacher_probability"] == pytest.approx(
        [math.exp(-0.1), math.exp(-2.0), math.exp(-0.25), math.exp(-0.1)]
    )
"""Response-window boundaries for transcript and bias context."""

import pytest

from recipe.phimm.agentic.context_prompt import ContextPrompt, add_context_to_messages


def test_text_window_cleans_orders_and_expires_empty_responses():
    context = ContextPrompt("text", max_history=2)
    context.observe("<src=English><tgt=English>\n<VERBATIM>Alpha.</VERBATIM>")
    context.observe("Beta.")
    assert context.context == "Alpha.\nBeta."
    context.observe("<nonspeech>")
    assert context.context == "Beta."
    context.observe("")
    assert context.context == ""


def test_bias_window_expires_terms_and_preserves_recent_duplicates():
    context = ContextPrompt("bias", max_history=2)
    context.observe("OpenAI, Qwen.")
    context.observe("Microsoft, OPENAI.")
    assert context.context == "*OPENAI*, *Microsoft*, *Qwen*"
    context.observe("NVIDIA.")
    assert context.context == "*NVIDIA*, *OPENAI*, *Microsoft*"
    context.observe("the and uh")
    assert context.context == "*NVIDIA*"
    context.observe("<nonspeech>")
    assert context.context == ""


def test_term_limit_is_independent_of_response_window():
    context = ContextPrompt("bias", max_history=2, max_terms=2)
    context.observe("Alpha, Beta, Gamma.")
    context.observe("Delta.")
    assert context.context == "*Delta*, *Gamma*"
    context.observe("")
    assert context.context == "*Delta*"


@pytest.mark.parametrize("mode", ["text", "bias"])
def test_large_window_expires_at_exact_response_boundary(mode):
    context = ContextPrompt(mode, max_history=1000)
    context.observe("OpenAI.")
    for _ in range(999):
        context.observe("<nonspeech>")
    assert context.context == ("OpenAI." if mode == "text" else "*OpenAI*")
    context.observe("<nonspeech>")
    assert context.context == ""


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("text", "Beta."), ("bias", "*Beta*")],
)
def test_default_window_uses_one_response(mode, expected):
    context = ContextPrompt(mode)
    context.observe("Alpha.")
    context.observe("Beta.")
    assert context.context == expected


@pytest.mark.parametrize("value", [None, 0, -1, True, False, 1.5, "2"])
@pytest.mark.parametrize("mode", ["text", "bias"])
def test_invalid_response_window_is_rejected(mode, value):
    with pytest.raises(ValueError, match="max_history must be a positive integer"):
        ContextPrompt(mode, max_history=value)


@pytest.mark.parametrize("mode", ["prev_text", "prev_bias", "full_bias", "unknown"])
def test_unsupported_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="Unknown ASR context mode"):
        ContextPrompt(mode)


def test_text_window_is_appended_without_changing_original_messages():
    messages = [{"role": "user", "content": "task<audio>"}]
    result = add_context_to_messages(messages, "Alpha.\nBeta.", "text")
    assert result[0]["content"] == "task<audio>\nPrevious context: Alpha.\nBeta."
    assert messages[0]["content"] == "task<audio>"

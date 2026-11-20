"""Response-window boundaries for transcript and bias context."""

import pytest
from recipe.phimm.agentic.context_prompt import ContextPrompt


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


def test_text_term_limit_keeps_last_words_across_cleaned_responses():
    context = ContextPrompt("text", max_history=2, max_terms=4)
    context.observe("<VERBATIM>Old words, Q-wen repeats repeats.</VERBATIM>")
    assert context.context == "words, Q-wen repeats repeats."
    context.observe("Latest\twords!")
    assert context.context == "repeats repeats.\nLatest\twords!"
    context.observe("<nonspeech>")
    assert context.context == "Latest\twords!"
    context.observe("")
    assert context.context == ""


@pytest.mark.parametrize("max_terms", [1, 3, 4])
def test_text_term_limit_preserves_suffix_spacing(max_terms):
    context = ContextPrompt("text", max_terms=max_terms)
    context.observe("Alpha,\tBeta!  Gamma.")
    expected = "Gamma." if max_terms == 1 else "Alpha,\tBeta!  Gamma."
    assert context.context == expected


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


@pytest.mark.parametrize("value", [None, 0, -2, True, False, 1.5, "2"])
@pytest.mark.parametrize("mode", ["text", "bias"])
def test_invalid_term_limit_is_rejected(mode, value):
    with pytest.raises(ValueError, match="max_terms must be -1 or a positive integer"):
        ContextPrompt(mode, max_terms=value)


@pytest.mark.parametrize("options", [{}, {"max_terms": -1}])
@pytest.mark.parametrize("mode", ["text", "bias"])
def test_unlimited_terms_still_respect_response_window(mode, options):
    context = ContextPrompt(mode, max_history=2, **options)
    words = [f"term{i}" for i in range(1100)]
    transcript = " ".join(words)
    context.observe(transcript)
    context.observe("Newest.")
    if mode == "text":
        assert context.context == transcript + "\nNewest."
    else:
        assert context.terms == ("Newest", *reversed(words))
    context.observe("")
    assert context.context == ("Newest." if mode == "text" else "*Newest*")
    context.observe("")
    assert context.context == ""


@pytest.mark.parametrize("mode", ["prev_text", "prev_bias", "full_bias", "unknown"])
def test_unsupported_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="Unknown ASR context mode"):
        ContextPrompt(mode)


def test_text_window_renders_instruction_without_changing_raw_context():
    context = ContextPrompt("text", max_history=2)
    context.observe("Alpha.")
    context.observe("Beta.")
    assert context.prompt == "Previous context: Alpha.\nBeta."
    assert context.context == "Alpha.\nBeta."


@pytest.mark.parametrize("mode", ["text", "bias"])
@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("Previous segment transcript (for context):", "Previous segment transcript (for context): Alpha."),
        ("  Context:\n ", "Context: Alpha."),
        ("", "Alpha."),
    ],
)
def test_custom_prefix_replaces_mode_instruction(mode, prefix, expected):
    context = ContextPrompt(mode, prefix=prefix)
    context.observe("Alpha.")
    if mode == "bias":
        expected = expected.replace("Alpha.", "*Alpha*")
    assert context.prompt == expected


def test_custom_prefix_is_omitted_without_context():
    assert ContextPrompt("text", prefix="Context:").prompt == ""


@pytest.mark.parametrize("prefix", [True, 123, [], {}])
def test_invalid_prefix_is_rejected(prefix):
    with pytest.raises(ValueError, match="prefix must be a string or null"):
        ContextPrompt("text", prefix=prefix)


def test_default_bias_instruction_is_preserved():
    context = ContextPrompt("bias")
    context.observe("OpenAI.")
    assert context.prompt == "Pay extra attention to the following phrases/words: *OpenAI*."

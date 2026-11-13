import pytest
from datasets import Dataset

from recipe.phimm.data.dataset import add_task_info, augment, mask_foreign_language
from recipe.phimm.reward.asr_edge import measure
from recipe.phimm.reward.asr_measure import _parse_response
from recipe.phimm.reward.asr_response import get_hyp_text


TASK = {"task": "lang_asr_verb", "lang_hint": True, "language": "English", "prefix_prob": 0.0}


def test_target_language_only_uses_config_target_instead_of_audio_language():
    ds = Dataset.from_list([{"text": "Hello", "language": "en_fr"}])
    row = add_task_info(ds, **{**TASK, "task": "lang_only_verb"})[0]
    assert row["language"] == "English"
    assert "Only include speech in English." in row["prompt"]
    assert "The target language is English." in row["prompt"]
    assert row["gt_output"] == "<src=English><tgt=English>\n<VERBATIM>\nHello"


def test_target_language_only_preserves_existing_foreign_reference_mask():
    ds = Dataset.from_list([{"text": "Bonjour", "language": "fr"}])
    row = augment(ds, mask_foreign_language={}, add_task_info={**TASK, "task": "lang_only_verb"})[0]
    assert row["text"] == "<foreign_language>"
    assert row["gt_output"].endswith("<foreign_language>")
    assert "Only include speech in English." in row["prompt"]


def test_target_language_only_rejects_disabled_language_hint():
    ds = Dataset.from_list([{"text": "Hello", "language": "en"}])
    with pytest.raises(ValueError, match="requires lang_hint=true"):
        add_task_info(ds, **{**TASK, "task": "lang_only_verb", "lang_hint": False})


@pytest.mark.parametrize("language,text,expected", [
    ("en", "Hello there", "Hello there"),
    ("fr", "Bonjour", "<foreign_language>"),
])
def test_mask_before_prompt_and_ground_truth(language, text, expected):
    ds = Dataset.from_list([{"text": text, "language": language}])
    row = augment(ds, mask_foreign_language={}, add_task_info=TASK, version="2609")[0]
    assert row["text"] == expected
    assert row["language"] == "English"
    assert "The language is English." in row["prompt"]
    assert row["gt_output"].endswith(expected)


@pytest.mark.parametrize("info", [{**TASK, "lang_hint": False}, {**TASK, "task": "lang_asr"}])
def test_noop_without_hint_and_verbatim(info):
    ds = Dataset.from_list([{"text": "Bonjour", "language": "fr"}])
    assert mask_foreign_language(ds, task_info=info)[0] == ds[0]


@pytest.mark.parametrize("source_language", ["en_fr", "en"])
def test_mixed_segments_mask_foreign_text(source_language):
    components = [{"text": "Hello", "language": "en"}, {"text": "bonjour", "language": "fr"},
                  {"text": "world", "language": "en"}]
    ds = Dataset.from_list([{"text": "Hello bonjour world", "language": source_language, "components": components}])
    row = augment(ds, mask_foreign_language={}, add_task_info=TASK, version="2609")[0]
    assert row["text"] == "Hello <foreign_language> world"
    assert row["components"] is None
    assert "The language is English." in row["prompt"]
    assert row["gt_output"].endswith("Hello <foreign_language> world")


def test_mixed_segments_keep_text_spacing_and_punctuation():
    components = [{"text": "Hello,", "language": "en"}, {"text": "안녕", "language": "ko"},
                  {"text": "world!", "language": "en"}]
    ds = Dataset.from_list([{"text": "Hello, 안녕world!", "language": "en_ko", "components": components}])
    assert mask_foreign_language(ds, task_info=TASK)[0]["text"] == "Hello, <foreign_language>world!"


def test_mixed_segments_require_complete_language_boundaries():
    missing = Dataset.from_list([{"text": "Hello bonjour", "language": "en_fr"}])
    with pytest.raises(ValueError, match="requires language-labelled components"):
        mask_foreign_language(missing, task_info=TASK)
    partial = Dataset.from_list([{"text": "Hello bonjour world", "language": "en_fr", "components": [
        {"text": "Hello", "language": "en"}, {"text": "bonjour", "language": "fr"}]}])
    with pytest.raises(ValueError, match="must cover the original transcription"):
        mask_foreign_language(partial, task_info=TASK)


def test_monolingual_components_are_masked_as_one_utterance():
    components = [{"text": "Bon", "language": "fr"}, {"text": "jour", "language": "fr"}]
    ds = Dataset.from_list([{"text": "Bon jour", "language": "fr", "components": components}])
    row = augment(ds, mask_foreign_language={}, add_task_info=TASK, version="2609")[0]
    assert row["text"] == "<foreign_language>"
    assert row["components"] is None
    assert row["gt_output"].endswith("<foreign_language>")


def test_mixed_and_monolingual_rows_can_share_a_dataset():
    rows = [{"text": "Hello bonjour", "language": "en_fr", "components": [
                {"text": "Hello", "language": "en"}, {"text": "bonjour", "language": "fr"}]},
            {"text": "Bonjour", "language": "fr", "components": None}]
    result = mask_foreign_language(Dataset.from_list(rows), task_info=TASK)
    assert result[0]["text"] == "Hello <foreign_language>"
    assert result[1]["text"] == "<foreign_language>"


def test_unknown_language_is_not_guessed():
    ds = Dataset.from_list([{"text": "Hello", "language": "Unknown"}])
    with pytest.raises(ValueError, match="known component language"):
        mask_foreign_language(ds, task_info=TASK)


def test_foreign_reward_does_not_ignore_the_marker():
    ref = "<foreign_language>"
    assert measure(ref, ref, tgt_lang="english", text_norm="lower").n_err == 0
    assert measure("", ref, tgt_lang="english", text_norm="lower").n_err == 1
    assert measure("foreign_language", ref, tgt_lang="english", text_norm="lower").n_err == 1


@pytest.mark.parametrize("text", ["<foreign_language>", "hello <foreign_language> world"])
def test_reward_parser_removes_foreign_tag_from_response(text):
    response = f"<src=English><tgt=English>\n<VERBATIM>\n{text}"
    assert "<foreign_language>" not in get_hyp_text(response, version=2609)
    score = _parse_response(response, text, version=2609, text_norm="lower",
                            extra_info={"language": "English", "keywords": []})
    # The reference is deliberately not rewritten by response cleanup.
    assert score["word"] < 1.0
    assert score["char"] < 1.0
    assert score["fmt"] == 1.0


def test_marker_and_empty_response_receive_same_lexical_score():
    prefix = "<src=English><tgt=English>\n<VERBATIM>\n"
    kwargs = {"version": 2609, "text_norm": "lower", "extra_info": {"language": "English", "keywords": []}}
    with_marker = _parse_response(prefix + "<foreign_language>", "<foreign_language>", **kwargs)
    empty = _parse_response(prefix, "<foreign_language>", **kwargs)
    assert with_marker["word"] == empty["word"]
    assert with_marker["char"] == empty["char"]

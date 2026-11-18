import json

import pytest
from datasets import Dataset
from recipe.phimm.data.dataset import add_task_info, augment, mask_foreign_language
from recipe.phimm.data.prompts import get_task_output, get_task_prompt, resolve_language

TASK = {"style": "verb", "lang_hint": True, "language": "English"}


@pytest.mark.parametrize("language", ["English", "French", "Chinese"])
def test_language_hint_prompt_and_output(language):
    prompt = get_task_prompt("verb", language=language)
    assert "Only include speech" not in prompt
    assert "Omit speech" not in prompt
    assert f"The language is {language}." in prompt
    assert json.loads(get_task_output("verb", language=language, text="sample")) == {
        "language": language,
        "text": "sample",
    }


def test_language_alias_is_normalized():
    assert resolve_language("en") == "English"
    assert "The language is English." in get_task_prompt(language="en")


def test_config_language_overrides_audio_language():
    ds = Dataset.from_list([{"text": "Hello", "language": "en_fr"}])
    row = add_task_info(ds, **TASK)[0]
    assert row["language"] == "English"
    assert "The language is English." in row["prompt"]
    assert json.loads(row["gt_output"]) == {"language": "English", "text": "Hello"}


def test_language_hint_does_not_rewrite_reference():
    ds = Dataset.from_list(
        [
            {
                "text": "Hello bonjour",
                "language": "en_fr",
                "components": [{"text": "Hello", "language": "en"}, {"text": "bonjour", "language": "fr"}],
            }
        ]
    )
    row = add_task_info(ds, **TASK)[0]
    assert row["text"] == "Hello bonjour"
    assert list(map(json.loads, row["gt_output"].splitlines())) == [
        {"language": "English", "text": "Hello"},
        {"language": "French", "text": "bonjour"},
    ]


def test_language_hint_can_be_disabled():
    ds = Dataset.from_list([{"text": "Hello", "language": "en_fr"}])
    row = add_task_info(ds, **{**TASK, "lang_hint": False})[0]
    assert "Only include speech" not in row["prompt"]
    assert "The language is" not in row["prompt"]
    assert row["language"] == "English"
    assert json.loads(row["gt_output"]) == {"language": "English", "text": "Hello"}


@pytest.mark.parametrize(
    "language,text,expected",
    [
        ("en", "Hello there", "Hello there"),
        ("fr", "Bonjour", "<foreign_language>"),
    ],
)
def test_mask_before_prompt_and_ground_truth(language, text, expected):
    ds = Dataset.from_list([{"text": text, "language": language}])
    row = augment(ds, mask_foreign_language={}, add_task_info=TASK)[0]
    assert row["text"] == expected
    assert row["language"] == "English"
    assert "The language is English." in row["prompt"]
    assert json.loads(row["gt_output"])["text"] == expected


@pytest.mark.parametrize("info", [{**TASK, "lang_hint": False}, {**TASK, "style": "lex"}])
def test_noop_without_hint_and_verbatim(info):
    ds = Dataset.from_list([{"text": "Bonjour", "language": "fr"}])
    assert mask_foreign_language(ds, task_info=info)[0] == ds[0]


@pytest.mark.parametrize("source_language", ["en_fr", "en"])
def test_mixed_segments_mask_foreign_text(source_language):
    components = [
        {"text": "Hello", "language": "en"},
        {"text": "bonjour", "language": "fr"},
        {"text": "world", "language": "en"},
    ]
    ds = Dataset.from_list([{"text": "Hello bonjour world", "language": source_language, "components": components}])
    row = augment(ds, mask_foreign_language={}, add_task_info=TASK)[0]
    assert row["text"] == "Hello <foreign_language> world"
    assert row["components"] is None
    assert "The language is English." in row["prompt"]
    assert json.loads(row["gt_output"])["text"] == "Hello <foreign_language> world"


def test_mixed_segments_keep_text_spacing_and_punctuation():
    components = [
        {"text": "Hello,", "language": "en"},
        {"text": "안녕", "language": "ko"},
        {"text": "world!", "language": "en"},
    ]
    ds = Dataset.from_list([{"text": "Hello, 안녕world!", "language": "en_ko", "components": components}])
    assert mask_foreign_language(ds, task_info=TASK)[0]["text"] == "Hello, <foreign_language>world!"


def test_mixed_segments_require_complete_language_boundaries():
    missing = Dataset.from_list([{"text": "Hello bonjour", "language": "en_fr"}])
    with pytest.raises(ValueError, match="requires language-labelled components"):
        mask_foreign_language(missing, task_info=TASK)
    partial = Dataset.from_list(
        [
            {
                "text": "Hello bonjour world",
                "language": "en_fr",
                "components": [{"text": "Hello", "language": "en"}, {"text": "bonjour", "language": "fr"}],
            }
        ]
    )
    with pytest.raises(ValueError, match="must cover the original transcription"):
        mask_foreign_language(partial, task_info=TASK)


def test_monolingual_components_are_masked_as_one_utterance():
    components = [{"text": "Bon", "language": "fr"}, {"text": "jour", "language": "fr"}]
    ds = Dataset.from_list([{"text": "Bon jour", "language": "fr", "components": components}])
    row = augment(ds, mask_foreign_language={}, add_task_info=TASK)[0]
    assert row["text"] == "<foreign_language>"
    assert row["components"] is None
    assert json.loads(row["gt_output"])["text"] == "<foreign_language>"


def test_mixed_and_monolingual_rows_can_share_a_dataset():
    rows = [
        {
            "text": "Hello bonjour",
            "language": "en_fr",
            "components": [{"text": "Hello", "language": "en"}, {"text": "bonjour", "language": "fr"}],
        },
        {"text": "Bonjour", "language": "fr", "components": None},
    ]
    result = mask_foreign_language(Dataset.from_list(rows), task_info=TASK)
    assert result[0]["text"] == "Hello <foreign_language>"
    assert result[1]["text"] == "<foreign_language>"


@pytest.mark.parametrize("language", ["Unknown", "not-a-language"])
def test_unknown_language_is_not_guessed(language):
    ds = Dataset.from_list([{"text": "Hello", "language": language}])
    with pytest.raises(ValueError, match="known component language"):
        mask_foreign_language(ds, task_info=TASK)


def test_invalid_hint_cannot_mask_valid_speech():
    ds = Dataset.from_list([{"text": "Hello", "language": "en"}])
    with pytest.raises(ValueError, match="known hint language"):
        mask_foreign_language(ds, task_info={**TASK, "language": "not-a-language"})

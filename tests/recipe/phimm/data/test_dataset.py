import json
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from datasets import Audio, Dataset, Features, Value
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

from recipe.phimm.data import dataset as dataset_module
from recipe.phimm.data.dataset import (
    _is_bad_fmt,
    _is_bad_lang,
    _has_wrong_numbers,
    add_task_info,
    format_asr_prompt,
)


class MinimalDataset:
    def map(self, function, **kwargs):
        self.example = function({"text": "bonjour"})
        return self


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I (love you)", "I love you"),
        ("I [love you] {very much} <indeed>", "I love you very much indeed"),
        ("I ([love] {you})", "I love you"),
        ("(one)(two)", "onetwo"),
        ("I (love you", "I (love you"),
        ("I love you)", "I love you)"),
        ("(outer [inner]", "(outer inner"),
        ("I ([love)] you", "I (love) you"),
        ("I () love [] you {}", "I  love  you "),
        ("I （love you）", "I love you"),
        ("【你好】，「世界」！", "你好，世界！"),
        ("《book》 〈chapter〉 『quote』", "book chapter quote"),
        ("I （[love] 【you】）", "I love you"),
        ("({[<love you>]})", "love you"),
        ("｟〖『「《〈〔【＜｛［（words）］｝＞】〕〉》」』〗｠", "words"),
        ("(one)[two]{three}<four>（five）【six】", "onetwothreefourfivesix"),
        ("()[]{}<>（）［］｛｝＜＞【】〔〕〈〉《》「」『』〖〗｟｠", ""),
        ("Before (\n[first]\tsecond\n) after", "Before \nfirst\tsecond\n after"),
        ("Say (don't) [stop!] {3.14} <a/b>", "Say don't stop! 3.14 a/b"),
        ("I (love you）", "I (love you）"),
        ("I （love you)", "I （love you)"),
        ("I [love you}", "I [love you}"),
        ("【outer (inner)", "【outer inner"),
        ("【outer (inner】)", "【outer inner】"),
        ("】I 【love you】)", "】I love you)"),
        ("((words)", "(words"),
        ("(words))", "words)"),
        ("No brackets.\nKeep  spaces!", "No brackets.\nKeep  spaces!"),
        ("", ""),
        (None, None),
    ],
)
def test_remove_brackets_preserves_words_and_unmatched_characters(text, expected):
    dataset = Dataset.from_dict({"text": [text], "keywords": [["love you"]], "id": ["sample"]})
    result = dataset_module.process_ds(dataset, remove_brackets={})
    assert result[0]["text"] == expected
    assert result[0]["keywords"] == ["love you"]
    assert result[0]["id"] == "sample"


@pytest.mark.parametrize(
    "pair",
    [
        "()",
        "[]",
        "{}",
        "<>",
        "（）",
        "［］",
        "｛｝",
        "＜＞",
        "【】",
        "〔〕",
        "〈〉",
        "《》",
        "「」",
        "『』",
        "〖〗",
        "｟｠",
    ],
)
@pytest.mark.parametrize(
    "template,expected",
    [
        ("I {open}love you{close}", "I love you"),
        ("{open}{close}", ""),
        ("{open}one{close}{open}two{close}", "onetwo"),
        ("{open}outer {open}inner{close}{close}", "outer inner"),
        ("{open}first\n\tsecond{close}", "first\n\tsecond"),
        ("{open}don't stop! 3.14{close}", "don't stop! 3.14"),
        ("I {open}love you", None),
        ("I love you{close}", None),
    ],
)
def test_remove_brackets_supports_common_bracket_types(pair, template, expected):
    text = template.format(open=pair[0], close=pair[1])
    dataset = Dataset.from_dict({"text": [text]})
    assert dataset_module.remove_brackets(dataset)[0]["text"] == (text if expected is None else expected)


def test_remove_brackets_runs_before_task_output_generation():
    dataset = Dataset.from_dict({"text": ["I (love you)"]})
    result = dataset_module.augment(
        dataset,
        version=2609,
        pre_process={"remove_brackets": {}},
        add_task_info={"task": "lang_asr_verb", "language": "English", "prefix_prob": 0.0, "lang_hint": True},
    )
    assert result[0]["text"] == "I love you"
    assert result[0]["gt_output"] == "<src=English><tgt=English>\n<VERBATIM>\nI love you"


@pytest.mark.parametrize(
    "config_name",
    [
        "train_data/name_en13m",
        "train_data/name_en10k",
        "train_data/name_env37k",
        "val_data/name_1k",
        "val_data/name_10k",
    ],
)
def test_name_data_configs_remove_brackets_before_verbatim_reference_generation(config_name):
    config_root = Path(__file__).parents[4] / "recipe/phimm/config/data"
    config = OmegaConf.to_container(OmegaConf.load(config_root / f"{config_name}.yaml"))[0]
    dataset = Dataset.from_dict({
        "text": ["John (Smith)"],
        "display": ["John [Smith]"],
        "audio_path": ["am_data/example.wav"],
        "keywords": [["Smith"]],
    })
    result = dataset_module.augment(
        dataset,
        version=2609,
        pre_process=config["pre_process"],
        add_task_info=config["add_task_info"],
    )

    assert result[0]["text"] == "John Smith"
    assert result[0]["keywords"] == ["Smith"]
    assert result[0]["gt_output"] == "<src=English><tgt=English>\n<VERBATIM>\nJohn Smith"
    assert "The language is English." in result[0]["prompt"]
    assert result[0]["prefix"] == ""


def test_remove_brackets_supports_parallel_map_and_rename():
    dataset = Dataset.from_dict({"transcript": ["I (love you)", "[hello]"]})
    result = dataset_module.process_ds(
        dataset,
        rename_fields={"mappings": {"text": "transcript"}},
        remove_brackets={},
        num_proc=2,
    )
    assert result["text"] == ["I love you", "hello"]


def test_remove_brackets_rejects_non_text_values():
    with pytest.raises(ValueError, match="string or None"):
        dataset_module.remove_brackets(Dataset.from_dict({"text": [123]}))


def test_remove_brackets_accepts_optional_field():
    dataset = Dataset.from_dict({"text": ["Keep (text)", "Keep (text)"], "transcription": ["I (love you)", None]})
    result = dataset_module.process_ds(dataset, remove_brackets={"field": "transcription"})
    assert result["transcription"] == ["I love you", None]
    assert result["text"] == ["Keep (text)", "Keep (text)"]


@pytest.mark.parametrize("field", ["", None, 123])
def test_remove_brackets_rejects_invalid_field(field):
    with pytest.raises(ValueError, match="nonempty column name"):
        dataset_module.remove_brackets(Dataset.from_dict({"text": ["hello"]}), field=field)


def test_rm0b_v0_recipe_only_changes_openml_training_preprocessing():
    root = Path(__file__).resolve().parents[4] / "recipe/phimm/config/v2609_asr"
    prefix = "remax_2609r2_earning_ml0_verb_hint_e3_bs128_n16_r256_g32_ds0_sfl_avg6_flr1_rm0b"
    with initialize_config_dir(config_dir=str(root), version_base=None):
        original = compose(config_name=prefix)
        variant = compose(config_name=prefix + "_v0")
    original_train = OmegaConf.to_container(original.data.train_data, resolve=True)
    variant_train = OmegaConf.to_container(variant.data.train_data, resolve=True)
    assert len(variant_train) == 1
    assert variant_train[0][0]["post_process"]["add_field"]["fields"]["data_source"] == "openml"
    assert variant_train[0][0].pop("pre_process") == {"remove_brackets": {}}
    assert original_train == variant_train
    source = OmegaConf.to_container(original, resolve=False)
    changed = OmegaConf.to_container(variant, resolve=False)
    assert changed["_ml_train"][0].pop("pre_process") == {"remove_brackets": {}}
    assert source == changed


def test_jsonl_dataset_keeps_date_shaped_keywords_as_strings(tmp_path):
    jsonl_path = tmp_path / "keywords.jsonl"
    records = [
        {"text": "first", "keywords": None},
        {"text": "second", "keywords": ["2026-09-14"]},
    ]
    jsonl_path.write_text("".join(json.dumps(record) + "\n" for record in records))

    dataset = dataset_module.jsonl_dataset(str(jsonl_path))

    assert dataset[1]["keywords"] == ["2026-09-14"]


def test_parquet_dataset_can_keep_embedded_audio_bytes_without_torchcodec(tmp_path):
    audio_bytes = io.BytesIO()
    sf.write(audio_bytes, np.zeros(1600), 16000, format="WAV")
    parquet_path = tmp_path / "audio.parquet"
    Dataset.from_dict(
        {"audio": [{"bytes": audio_bytes.getvalue(), "path": None}], "text": ["hello"]},
        features=Features({"audio": Audio(), "text": Value("string")}),
    ).to_parquet(str(parquet_path))

    dataset = dataset_module.parquet_dataset(str(parquet_path), decode_audio=False)

    assert dataset.features["audio"].decode is False
    assert dataset[0]["audio"]["bytes"] == audio_bytes.getvalue()


def test_format_asr_prompt_uses_2607_audio_placement():
    assert format_asr_prompt("Transcribe.") == "Transcribe.<audio>"


def test_format_asr_prompt_preserves_2609_audio_placement():
    prompt = "Detect the language and transcribe the audio clip into text.<audio>\nThe language is Chinese."

    assert format_asr_prompt(prompt) == prompt


def test_add_task_info_enables_language_prefix_by_default():
    dataset = add_task_info(
        MinimalDataset(), task="lang_asr", language="French", prefix_prob=1.0
    )

    assert dataset.example["prefix"] == "<src=French><tgt=French>\n"
    assert dataset.example["gt_output"] == "bonjour"


def test_add_task_info_supports_2607_prefix_and_completion():
    dataset = add_task_info(
        MinimalDataset(),
        task="lang_asr",
        language="French",
        version=2607,
        prefix_prob=1.0,
    )

    assert dataset.example["prefix"] == "Audio Language: French\n"
    assert dataset.example["gt_output"] == (
        "<ASR><lang=French><TXT>bonjour</TXT></ASR>"
    )


def test_add_task_info_supports_2609_prompt_prefix_and_completion():
    dataset = add_task_info(
        MinimalDataset(),
        task="lang_asr",
        language="French",
        version=2609,
        prefix_prob=1.0,
        lang_hint=True,
    )

    assert dataset.example["prompt"] == (
        "Transcribe the audio clip into text.<audio>\nThe language is French."
    )
    assert dataset.example["prefix"] == "<src=French><tgt=French>\n"
    assert dataset.example["gt_output"] == "bonjour"


def test_add_task_info_supports_2609_detect_language_prompt():
    dataset = add_task_info(
        MinimalDataset(),
        task="lang_asr",
        language="French",
        version=2609,
        prefix_prob=0.0,
    )

    assert dataset.example["prompt"] == (
        "Detect the language and transcribe the audio clip into text.<audio>"
    )
    assert dataset.example["prefix"] == ""
    assert dataset.example["gt_output"] == (
        "<src=French><tgt=French>\nbonjour"
    )


@pytest.mark.parametrize(
    ("task", "prefix_prob", "lang_hint", "expected_prompt", "expected_prefix", "expected_output"),
    [
        (
            "lang_asr_lex",
            1.0,
            True,
            "Transcribe the audio clip into text. Output must be in lexical format.<audio>\n"
            "The language is French.",
            "<src=French><tgt=French>\n",
            "<LEXICAL>\nbonjour",
        ),
        (
            "lang_asr_verb",
            0.0,
            False,
            "Detect the language and transcribe the audio clip into text. "
            "Transcribe verbatim, including all filler words and disfluencies.<audio>",
            "",
            "<src=French><tgt=French>\n<VERBATIM>\nbonjour",
        ),
    ],
)
def test_add_task_info_supports_2609_mode_prompt_and_completion(
    task,
    prefix_prob,
    lang_hint,
    expected_prompt,
    expected_prefix,
    expected_output,
):
    dataset = add_task_info(
        MinimalDataset(),
        task=task,
        language="French",
        version=2609,
        prefix_prob=prefix_prob,
        lang_hint=lang_hint,
    )

    assert dataset.example["prompt"] == expected_prompt
    assert dataset.example["prefix"] == expected_prefix
    assert dataset.example["gt_output"] == expected_output


def test_add_task_info_allows_language_prefix_opt_out():
    dataset = add_task_info(MinimalDataset(), task="lang_asr", language="French", prefix_prob=0.0)

    assert dataset.example["prefix"] == ""


@pytest.mark.parametrize(
    ("version", "expected_prompt"),
    [
        (
            2607,
            "Transcribe the audio clip into text.<audio>\nThe language is French.",
        ),
        (
            2609,
            "Transcribe the audio clip into text.<audio>\nThe language is French.",
        ),
    ],
)
def test_add_task_info_allows_lang_hint_without_prefix(version, expected_prompt):
    dataset = add_task_info(
        MinimalDataset(),
        task="lang_asr",
        language="French",
        version=version,
        prefix_prob=0.0,
        lang_hint=True,
    )

    assert dataset.example["prompt"] == expected_prompt
    assert dataset.example["prefix"] == ""


@pytest.mark.parametrize(
    ("version", "expected_prefix"),
    [
        (2607, "Audio Language: French\n"),
        (2609, "<src=French><tgt=French>\n"),
    ],
)
def test_add_task_info_prefix_does_not_enable_lang_hint(version, expected_prefix):
    dataset = add_task_info(
        MinimalDataset(),
        task="lang_asr",
        language="French",
        version=version,
        prefix_prob=1.0,
        lang_hint=False,
    )

    assert dataset.example["prompt"] == (
        "Detect the language and transcribe the audio clip into text.<audio>"
    )
    assert dataset.example["prefix"] == expected_prefix


def test_add_task_info_uses_2607_multilingual_components():
    dataset = Dataset.from_list(
        [
            {
                "language": "Hebrew Hindi",
                "text": "מאין לך זאת? כבר היית שם? सुप्रीम कोर्ट जल्द करेगा सुनवाई",
                "components": [
                    {"language": "Hebrew", "text": "מאין לך זאת? כבר היית שם?"},
                    {"language": "Hindi", "text": "सुप्रीम कोर्ट जल्द करेगा सुनवाई"},
                ],
            }
        ]
    )

    result = add_task_info(
        dataset,
        task="lang_asr",
        version=2607,
        prefix_prob=0.0,
    )

    assert result[0]["gt_output"] == (
        "Audio Language: Hebrew and Hindi.\n"
        "<ASR><lang=Hebrew><TXT>מאין לך זאת? כבר היית שם?</TXT>\n"
        "<lang=Hindi><TXT>सुप्रीम कोर्ट जल्द करेगा सुनवाई</TXT></ASR>"
    )


def test_bad_format_uses_task_output_format():
    valid = "<src=English><tgt=English>\n<TXT>Hello</TXT>"

    assert not _is_bad_fmt({"raw_response": valid})
    assert _is_bad_fmt({"raw_response": valid + "\n<src=French><tgt=French>\n<VERBAT bonjour"})
    assert not _is_bad_fmt({"raw_response": valid + "\n<src=French><tgt=French>\nbonjour"})
    assert _is_bad_fmt({"raw_response": "Hello"})
    assert not _is_bad_fmt(
        {
            "raw_response": "<ASR><lang=English><TXT>Hello</TXT></ASR>",
            "version": 2607,
        }
    )
    assert _is_bad_fmt({"raw_response": None})


def test_bad_language_uses_task_output_languages():
    mixed = (
        "<src=English><tgt=English>\n<TXT>Hello</TXT>\n"
        "<src=Chinese><tgt=Chinese>\n<TXT>ni hao</TXT>"
    )
    wrong = "<src=French><tgt=French>\n<TXT>Bonjour</TXT>"

    assert not _is_bad_lang({"raw_response": mixed, "language": "English_Chinese"})
    assert _is_bad_lang({"raw_response": wrong, "language": "English"})
    assert _is_bad_lang({"raw_response": "Hello", "language": "English"})


def test_nonspeech_without_language_tags_is_bad_language():
    nonspeech = "<nonspeech>"

    assert _is_bad_lang({"raw_response": nonspeech, "language": "French"})


@pytest.mark.parametrize(
    ("predictions", "expected"),
    [
        (["English", "Chinese"], False),
        (["English"], True),
        (["English", "French"], True),
        (["Chinese", "English"], True),
        (["English", "Chinese", "French"], True),
    ],
)
def test_bad_language_handles_detailed_ordered_scores(predictions, expected):
    response = "\n".join(f"<src={lang}><tgt={lang}>\n<TXT>word</TXT>" for lang in predictions)
    assert _is_bad_lang({"raw_response": response, "language": "English Chinese"}) is expected


def test_wrong_numbers_uses_openasr_english_normalization():
    assert not _has_wrong_numbers(
        {"text": "Revenue was twenty-five million dollars.", "response": "Revenue was $25 million."},
        {},
    )
    assert _has_wrong_numbers(
        {"text": "Revenue was twenty-five million dollars.", "response": "Revenue was $35 million."},
        {},
    )


def test_wrong_numbers_requires_a_number_in_reference_and_supports_custom_fields():
    assert not _has_wrong_numbers({"text": "Revenue increased.", "response": "Revenue increased 5%."}, {})
    assert _has_wrong_numbers(
        {"reference": "Revenue was twenty million dollars.", "hypothesis": "Revenue was $2 million."},
        {"ref_field": "reference", "hyp_field": "hypothesis"},
    )


def test_random_edge_word_cut_keeps_matching_text_and_audio_prefix(monkeypatch):
    dataset = Dataset.from_dict({"text": ["one two three four"], "audio_path": ["sample.wav"]})
    monkeypatch.setattr(dataset_module.random, "randint", lambda start, end: 2)

    result = dataset_module.random_edge_word_cut(dataset)

    assert result[0]["text"] == "one two"
    assert result[0]["audio_path"] == "sample.wav#0%:50%"


def test_add_rare_keywords_supports_rare_file_without_common_file(monkeypatch):
    dataset = Dataset.from_dict({"text": ["common keyword absent"]})
    monkeypatch.setattr(
        dataset_module,
        "read_words",
        lambda file_path, **kwargs: ["keyword", "missing"] if file_path == "rare.txt" else [],
    )

    result = dataset_module.add_rare_keywords(dataset, rare_file="rare.txt")

    assert result[0]["keywords"] == ["keyword"]


def test_random_edge_word_cut_supports_max_words_range(monkeypatch):
    dataset = Dataset.from_dict(
        {"text": ["one two three four five six"], "audio_path": ["sample.wav"]}
    )
    bounds = []

    def fake_randint(start, end):
        bounds.append((start, end))
        return end

    monkeypatch.setattr(dataset_module.random, "randint", fake_randint)

    result = dataset_module.random_edge_word_cut(dataset, max_words=[2, 4])

    assert bounds == [(2, 4)]
    assert result[0]["text"] == "one two three four"
    assert result[0]["audio_path"] == "sample.wav#0%:66.666667%"


def test_random_edge_word_cut_accepts_omegaconf_max_words_range(monkeypatch):
    dataset = Dataset.from_dict({"text": ["one two three four"], "audio_path": ["sample.wav"]})
    config = OmegaConf.create({"max_words": [2, 3]})
    monkeypatch.setattr(dataset_module.random, "randint", lambda start, end: end)

    result = dataset_module.random_edge_word_cut(dataset, max_words=config.max_words)

    assert result[0]["text"] == "one two three"


def test_random_edge_word_cut_caps_max_words_range_to_transcript(monkeypatch):
    dataset = Dataset.from_dict({"text": ["one two three"], "audio_path": ["sample.wav"]})
    bounds = []
    monkeypatch.setattr(
        dataset_module.random,
        "randint",
        lambda start, end: bounds.append((start, end)) or end,
    )

    result = dataset_module.random_edge_word_cut(dataset, max_words=[5, 10])

    assert bounds == [(2, 2)]
    assert result[0]["text"] == "one two"


def test_process_ds_random_edge_word_cut_runs_after_rename_fields(monkeypatch):
    dataset = Dataset.from_dict({"Transcription": ["one two three"], "WavPath": ["sample.wav"]})
    monkeypatch.setattr(dataset_module.random, "randint", lambda start, end: 1)

    result = dataset_module.process_ds(
        dataset,
        rename_fields={"mappings": {"text": "Transcription", "audio_path": "WavPath"}},
        random_edge_word_cut={},
    )

    assert result[0]["text"] == "one"
    assert result[0]["audio_path"] == "sample.wav#0%:33.333333%"


def test_random_edge_audio_cut_generates_relative_segment(monkeypatch):
    dataset = Dataset.from_dict({"audio_path": ["sample.wav"]})
    monkeypatch.setattr(dataset_module.random, "uniform", lambda start, end: end)

    result = dataset_module.random_edge_audio_cut(
        dataset, head_cut=[0, 0.1], tail_cut=[0, 0.1]
    )

    assert result[0]["audio_path"] == "sample.wav#0.1:-0.1"


@pytest.mark.parametrize(
    ("options", "expected"),
    [({"head_cut": 1}, "sample.wav#1:"), ({"tail_cut": 1}, "sample.wav#:-1")],
)
def test_random_edge_audio_cut_supports_fixed_single_edge(options, expected):
    dataset = Dataset.from_dict({"audio_path": ["sample.wav"]})

    result = dataset_module.random_edge_audio_cut(dataset, **options)

    assert result[0]["audio_path"] == expected


def test_random_edge_audio_cut_defaults_to_no_cut():
    dataset = Dataset.from_dict({"audio_path": ["sample.wav"]})

    result = dataset_module.random_edge_audio_cut(dataset)

    assert result[0]["audio_path"] == "sample.wav"


def test_process_ds_random_edge_audio_cut_runs_after_rename_fields(monkeypatch):
    dataset = Dataset.from_dict({"WavPath": ["sample.wav"]})

    result = dataset_module.process_ds(
        dataset,
        rename_fields={"mappings": {"audio_path": "WavPath"}},
        random_edge_audio_cut={"head_cut": 0.1, "tail_cut": 0.1},
    )

    assert result[0]["audio_path"] == "sample.wav#0.1:-0.1"


def test_clean_tagged_text_removes_tags_and_extracts_keywords():
    dataset = Dataset.from_list(
        [
            {
                "text": (
                    "Great. That's uh <ST/> that's helpful, "
                    "<PName> Collar </pname> . Thank you."
                ),
                "keywords": ["existing"],
            }
        ]
    )

    result = dataset_module.clean_tagged_text(dataset)

    assert result[0]["text"] == "Great. That's uh that's helpful, Collar. Thank you."
    assert result[0]["keywords"] == ["Collar"]


@pytest.mark.parametrize(
    "tagged,expected,keywords",
    [
        ("from<NE> Morgan Stanley</NE>.", "from Morgan Stanley.", ["Morgan Stanley"]),
        ("<NE>Morgan Stanley </NE>analyst", "Morgan Stanley analyst", ["Morgan Stanley"]),
        ("to<NE> QDN</NE>'s call", "to QDN's call", ["QDN"]),
        ("with<PName> Nikolaj Lippmann </PName>from", "with Nikolaj Lippmann from", ["Nikolaj Lippmann"]),
        ("from[ENTITY] Morgan Stanley[/ENTITY].", "from Morgan Stanley.", ["Morgan Stanley"]),
        ("from<NE>\nMorgan Stanley\t</NE>today", "from Morgan Stanley today", ["Morgan Stanley"]),
        ("bullion<NE> </NE>on-hand", "bullion on-hand", []),
        ("multi<NE>modal</NE>", "multimodal", ["modal"]),
    ],
)
def test_clean_tagged_text_preserves_word_separators_inside_tags(tagged, expected, keywords):
    dataset = Dataset.from_dict({"text": [tagged]})

    result = dataset_module.clean_tagged_text(dataset)

    assert result[0]["text"] == expected
    assert result[0]["keywords"] == keywords


def test_clean_tagged_text_handles_bracket_tags_and_tagged_keywords():
    dataset = Dataset.from_list(
        [
            {
                "text": (
                    "And the world of [ENTITY]!Vamos![/ENTITY] is a celebration. "
                    "I was just in [ENTITY]Midland, Texas[/ENTITY] last week."
                ),
                "keywords": ["[ENTITY]!Vamos![/ENTITY]"],
            },
            {
                "text": 'He said, "I\'m healed! I\'m healed!" And everybody jumped.',
                "keywords": [],
            },
        ]
    )

    result = dataset_module.clean_tagged_text(dataset)

    assert result[0]["text"] == (
        "And the world of Vamos is a celebration. "
        "I was just in Midland, Texas last week."
    )
    assert result[0]["keywords"] == ["Vamos", "Midland, Texas"]
    assert result[1]["text"] == dataset[1]["text"]
    assert result[1]["keywords"] == []


def test_clean_tagged_text_replaces_fragment_keywords_with_acronym():
    dataset = Dataset.from_list(
        [
            {
                "text": 'If they ever need my [ACRONYM]DNA[/ACRONYM], it is here.',
                "keywords": [
                    "If they ever need my [ACRONYM]DNA[/ACRONYM],",
                    "it is here.",
                ],
            }
        ]
    )

    result = dataset_module.clean_tagged_text(dataset)

    assert result[0]["text"] == "If they ever need my DNA, it is here."
    assert result[0]["keywords"] == ["DNA"]


def test_clean_tagged_text_does_not_cross_repeated_openers():
    dataset = Dataset.from_dict(
        {
            "text": [
                "Read [ENTITY]an unfinished phrase, then "
                "meet [ENTITY]Paola Escobar[/ENTITY] in [ENTITY>Bogota."
            ]
        }
    )

    result = dataset_module.clean_tagged_text(dataset)

    assert result[0]["text"] == (
        "Read an unfinished phrase, then meet Paola Escobar in Bogota."
    )
    assert result[0]["keywords"] == ["Paola Escobar"]


def test_clean_tagged_text_handles_inline_and_orphan_bracket_tags():
    dataset = Dataset.from_dict(
        {
            "text": [
                "Other [ENTITY Texas ] and [ENTITY&lt;Democrats Democrats ] "
                "saw [ENTITY broken [ENTITY words [ENTITY KKK ]."
            ]
        }
    )

    result = dataset_module.clean_tagged_text(dataset)

    assert result[0]["text"] == (
        "Other Texas and Democrats Democrats saw broken words KKK."
    )
    assert result[0]["keywords"] == ["Texas", "Democrats Democrats", "KKK"]


@pytest.mark.parametrize("tag", ["<ST/>", "<UNKNOWN/>", "<FILL/>"])
def test_clean_tagged_text_removes_self_closing_tags_without_keywords(tag):
    dataset = Dataset.from_dict({"text": [f"Before {tag} after."]})

    result = dataset_module.clean_tagged_text(dataset)

    assert result[0]["text"] == "Before after."
    assert result[0]["keywords"] == []


def test_clean_tagged_text_parallel_map_handles_late_keywords():
    dataset = Dataset.from_dict(
        {"text": ["plain text"] * 100 + ["<PName>Ming Luo</PName>"]}
    )

    result = dataset_module.clean_tagged_text(dataset, num_proc=2)

    assert result[-1]["keywords"] == ["Ming Luo"]


def test_process_ds_cleans_tags_after_rename_fields():
    dataset = Dataset.from_dict(
        {"Display": ["Welcome <PName> Ming Luo </PName> from <Org> QDN </Org>."]}
    )

    result = dataset_module.process_ds(
        dataset,
        rename_fields={"mappings": {"text": "Display"}},
        clean_tagged_text={},
    )

    assert result[0]["text"] == "Welcome Ming Luo from QDN."
    assert result[0]["keywords"] == ["Ming Luo", "QDN"]


def test_dataset_load_audio_reads_percentage_range(tmp_path):
    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, np.zeros(1000, dtype=np.float32), 16000)
    dataset = Dataset.from_dict({"audio_path": [f"{audio_path}#10%:30%"]})

    result = dataset_module.load_audio(dataset)

    assert result[0]["sr"] == 16000
    assert len(result[0]["audio"]) == 200

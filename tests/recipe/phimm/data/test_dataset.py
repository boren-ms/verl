import numpy as np
import soundfile as sf
from datasets import Dataset
from omegaconf import OmegaConf

from recipe.phimm.data import dataset as dataset_module
from recipe.phimm.data.dataset import _is_bad_fmt, _is_bad_lang, add_task_info, add_teacher, verl_format_ds


class MinimalDataset:
    def map(self, function, **kwargs):
        self.example = function({"text": "bonjour"})
        return self


class ExampleDataset:
    def __init__(self, example):
        self.input_example = example

    def map(self, function, **kwargs):
        self.example = function(self.input_example)
        return self


def test_add_task_info_enables_language_prefix_by_default():
    dataset = add_task_info(MinimalDataset(), task="lang_asr", language="French")

    assert dataset.example["prefix"] == "Audio Language: French\n"


def test_add_task_info_allows_language_prefix_opt_out():
    dataset = add_task_info(MinimalDataset(), task="lang_asr", language="French", prefix_prob=0.0)

    assert dataset.example["prefix"] == ""


def test_add_teacher_supports_task_specification():
    dataset = add_teacher(MinimalDataset(), task="lang_asr", model_version=2607)

    assert dataset.example["teacher_prompt"] == "Detect the language and transcribe the audio clip into text.<audio>"


def test_add_teacher_appends_starred_keywords_when_biasing():
    dataset = add_teacher(
        ExampleDataset({"text": "hello contoso", "keywords": ["Abate", "Alberto Sordi"]}),
        task="asr",
        model_version=2607,
        biasing=True,
    )

    assert dataset.example["teacher_prompt"] == (
        "Transcribe the audio clip into text.<audio>\n"
        "Pay extra attention to the following phrases/words: *Abate*, *Alberto Sordi*."
    )


def test_add_teacher_ignores_keywords_when_biasing_disabled():
    dataset = add_teacher(
        ExampleDataset({"text": "hello contoso", "keywords": ["Contoso", "Satya Nadella"]}),
        task="asr",
        biasing=False,
    )

    assert dataset.example["teacher_prompt"] == "<audio>\nTranscribe the audio clip into text."


def test_add_teacher_biasing_without_keywords_keeps_base_prompt():
    dataset = add_teacher(ExampleDataset({"text": "hello"}), task="asr", biasing=True)

    assert dataset.example["teacher_prompt"] == "<audio>\nTranscribe the audio clip into text."


def test_verl_format_preserves_teacher_prompt():
    dataset = Dataset.from_dict(
        {
            "prompt": ["<audio>\nStudent prompt"],
            "teacher_prompt": ["<audio>\nTeacher prompt"],
            "text": ["hello"],
        }
    )

    result = verl_format_ds(dataset)

    assert result[0]["teacher_prompt"] == "<audio>\nTeacher prompt"


def test_bad_format_uses_task_output_format():
    valid = "Audio Language: English.\n<ASR><lang=English><TXT>Hello</TXT></ASR>"

    assert not _is_bad_fmt({"raw_response": valid})
    assert _is_bad_fmt({"raw_response": "Hello"})


def test_bad_language_uses_task_output_languages():
    mixed = (
        "Audio Language: English and Chinese.\n"
        "<ASR><lang=English><TXT>Hello</TXT><lang=Chinese><TXT>ni hao</TXT></ASR>"
    )
    wrong = "Audio Language: French.\n<ASR><lang=French><TXT>Bonjour</TXT></ASR>"

    assert not _is_bad_lang({"raw_response": mixed, "language": "English_Chinese"})
    assert _is_bad_lang({"raw_response": wrong, "language": "English"})
    assert _is_bad_lang({"raw_response": "Hello", "language": "English"})


def test_nonspeech_is_not_bad_language():
    nonspeech = "Audio Language: English.\n<ASR><lang=English><TXT><nonspeech></TXT></ASR>"

    assert not _is_bad_lang({"raw_response": nonspeech, "language": "French"})


def test_random_cut_keeps_matching_text_and_audio_prefix(monkeypatch):
    dataset = Dataset.from_dict({"text": ["one two three four"], "audio_path": ["sample.wav"]})
    monkeypatch.setattr(dataset_module.random, "randint", lambda start, end: 2)

    result = dataset_module.random_cut(dataset)

    assert result[0]["text"] == "one two"
    assert result[0]["audio_path"] == "sample.wav#0%:50%"


def test_random_cut_supports_max_words_range(monkeypatch):
    dataset = Dataset.from_dict(
        {"text": ["one two three four five six"], "audio_path": ["sample.wav"]}
    )
    bounds = []

    def fake_randint(start, end):
        bounds.append((start, end))
        return end

    monkeypatch.setattr(dataset_module.random, "randint", fake_randint)

    result = dataset_module.random_cut(dataset, max_words=[2, 4])

    assert bounds == [(2, 4)]
    assert result[0]["text"] == "one two three four"
    assert result[0]["audio_path"] == "sample.wav#0%:66.666667%"


def test_random_cut_accepts_omegaconf_max_words_range(monkeypatch):
    dataset = Dataset.from_dict({"text": ["one two three four"], "audio_path": ["sample.wav"]})
    config = OmegaConf.create({"max_words": [2, 3]})
    monkeypatch.setattr(dataset_module.random, "randint", lambda start, end: end)

    result = dataset_module.random_cut(dataset, max_words=config.max_words)

    assert result[0]["text"] == "one two three"


def test_random_cut_caps_max_words_range_to_transcript(monkeypatch):
    dataset = Dataset.from_dict({"text": ["one two three"], "audio_path": ["sample.wav"]})
    bounds = []
    monkeypatch.setattr(
        dataset_module.random,
        "randint",
        lambda start, end: bounds.append((start, end)) or end,
    )

    result = dataset_module.random_cut(dataset, max_words=[5, 10])

    assert bounds == [(2, 2)]
    assert result[0]["text"] == "one two"


def test_process_ds_random_cut_runs_after_rename_fields(monkeypatch):
    dataset = Dataset.from_dict({"Transcription": ["one two three"], "WavPath": ["sample.wav"]})
    monkeypatch.setattr(dataset_module.random, "randint", lambda start, end: 1)

    result = dataset_module.process_ds(
        dataset,
        rename_fields={"mappings": {"text": "Transcription", "audio_path": "WavPath"}},
        random_cut={},
    )

    assert result[0]["text"] == "one"
    assert result[0]["audio_path"] == "sample.wav#0%:33.333333%"


def test_dataset_load_audio_reads_percentage_range(tmp_path):
    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, np.zeros(1000, dtype=np.float32), 16000)
    dataset = Dataset.from_dict({"audio_path": [f"{audio_path}#10%:30%"]})

    result = dataset_module.load_audio(dataset)

    assert result[0]["sr"] == 16000
    assert len(result[0]["audio"]) == 200
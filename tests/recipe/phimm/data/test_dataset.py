import numpy as np
import soundfile as sf
from datasets import Dataset
from omegaconf import OmegaConf

from recipe.phimm.data import dataset as dataset_module
from recipe.phimm.data.dataset import _is_bad_fmt, _is_bad_lang, add_task_info, format_asr_prompt


class MinimalDataset:
    def map(self, function, **kwargs):
        self.example = function({"text": "bonjour"})
        return self


def test_format_asr_prompt_uses_2607_audio_placement():
    assert format_asr_prompt("Transcribe.") == "Transcribe.<audio>"


def test_add_task_info_enables_language_prefix_by_default():
    dataset = add_task_info(MinimalDataset(), task="lang_asr", language="French")

    assert dataset.example["prefix"] == "<src=French><tgt=French>\n"


def test_add_task_info_allows_language_prefix_opt_out():
    dataset = add_task_info(MinimalDataset(), task="lang_asr", language="French", prefix_prob=0.0)

    assert dataset.example["prefix"] == ""


def test_bad_format_uses_task_output_format():
    valid = "<src=English><tgt=English>\nHello"

    assert not _is_bad_fmt({"raw_response": valid})
    assert _is_bad_fmt({"raw_response": "Hello"})


def test_bad_language_uses_task_output_languages():
    mixed = (
        "<src=English><tgt=English>\nHello\n"
        "<src=Chinese><tgt=Chinese>\nni hao"
    )
    wrong = "<src=French><tgt=French>\nBonjour"

    assert not _is_bad_lang({"raw_response": mixed, "language": "English_Chinese"})
    assert _is_bad_lang({"raw_response": wrong, "language": "English"})
    assert _is_bad_lang({"raw_response": "Hello", "language": "English"})


def test_nonspeech_is_not_bad_language():
    nonspeech = "<src=English><tgt=English>\n<nonspeech>"

    assert not _is_bad_lang({"raw_response": nonspeech, "language": "French"})


def test_random_cut_keeps_matching_text_and_audio_prefix(monkeypatch):
    dataset = Dataset.from_dict({"text": ["one two three four"], "audio_path": ["sample.wav"]})
    monkeypatch.setattr(dataset_module.random, "randint", lambda start, end: 2)

    result = dataset_module.random_cut(dataset)

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
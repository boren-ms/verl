from datasets import Dataset

from recipe.phimm.data.dataset import add_field_ds, verl_format_ds
from recipe.phimm.reward.asr_eval import openasr_hi_in_eval


def test_add_field_overwrites_multiple_constant_fields():
    dataset = Dataset.from_dict(
        {"text": ["original"], "data_source": ["asr"], "prompt": ["transcribe"]}
    )

    result = add_field_ds(dataset, fields={"text": "<nonspeech>", "data_source": "non_speech"})

    assert result[0]["text"] == "<nonspeech>"
    assert result[0]["data_source"] == "non_speech"

    formatted = verl_format_ds(result)
    assert formatted[0]["reward_model"]["ground_truth"] == "<nonspeech>"


def test_verl_format_adds_configured_extra_keys_to_defaults():
    dataset = Dataset.from_dict(
        {
            "text": ["hello"],
            "prompt": ["transcribe"],
            "id": ["sample-1"],
            "language": ["English"],
            "keywords": [["hello"]],
            "prefix": ["Audio Language: English\n"],
            "parent_audio_path": ["call.wav"],
        }
    )

    formatted = verl_format_ds(
        dataset,
        extra_keys=["id", "prefix", "parent_audio_path", "parent_audio_path"],
    )

    assert len(formatted[0]["extra_info"]) == 5
    assert set(formatted[0]["extra_info"]) == {
        "id",
        "language",
        "keywords",
        "prefix",
        "parent_audio_path",
    }
    assert formatted[0]["extra_info"] == {
        "id": "sample-1",
        "language": "English",
        "keywords": ["hello"],
        "prefix": "Audio Language: English\n",
        "parent_audio_path": "call.wav",
    }


def test_verl_format_preserves_hindi_lattice_for_openasr_scoring():
    lattice = [["hello", "world"], ["today"]]
    dataset = Dataset.from_dict(
        {
            "text": ["hello today"],
            "prompt": ["transcribe"],
            "id": ["hindi-sample"],
            "language": ["Hindi"],
            "data_source": ["monsoon_hi_in"],
            "lattice": [lattice],
        }
    )
    row = verl_format_ds(dataset, extra_keys=["lattice"])[0]

    assert row["extra_info"]["lattice"] == lattice
    assert row["extra_info"]["language"] == "Hindi"
    assert row["reward_model"]["ground_truth"] == "hello today"
    result = openasr_hi_in_eval(
        "<src=Hindi><tgt=Hindi>\n<TXT>world today</TXT>",
        row["reward_model"]["ground_truth"],
        version=2609,
        data_source=row["data_source"],
        extra_info=row["extra_info"],
    )
    assert result == {"score": 1.0, "wer": 0.0, "n_err": 0, "n_ref": 2}
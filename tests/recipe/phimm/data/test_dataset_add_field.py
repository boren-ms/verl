from datasets import Dataset

from recipe.phimm.data.dataset import add_field_ds, verl_format_ds


def test_add_field_overwrites_multiple_constant_fields():
    dataset = Dataset.from_dict(
        {"text": ["original"], "data_source": ["asr"], "prompt": ["transcribe"]}
    )

    result = add_field_ds(dataset, fields={"text": "<nonspeech>", "data_source": "non_speech"})

    assert result[0]["text"] == "<nonspeech>"
    assert result[0]["data_source"] == "non_speech"

    formatted = verl_format_ds(result)
    assert formatted[0]["reward_model"]["ground_truth"] == "<nonspeech>"
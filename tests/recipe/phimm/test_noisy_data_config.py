from copy import deepcopy
from pathlib import Path

import pytest
from datasets import Dataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from recipe.phimm.data.dataset import augment


CONFIG_ROOT = Path(__file__).parents[3] / "recipe/phimm/config"
TRAIN_ROOT = CONFIG_ROOT / "data/train_data"
NOISE_RIR = "noisy/enht_v1_verb_lang"
NOISY_LS = "noisy/ls_verb_langhint"
TASK_INFO = {
    "task": "lang_asr_verb",
    "language": "English",
    "prefix_prob": 0.0,
    "lang_hint": True,
}


def load_config(name):
    return OmegaConf.to_container(OmegaConf.load(TRAIN_ROOT / f"{name}.yaml"))


def test_noise_rir_config_preserves_all_augmented_sources_and_processing():
    original = load_config("entity/entity_en_ht_v1_cer")
    expected = deepcopy(original)
    expected[0]["jsonl_paths"] = [
        path for path in original[0]["jsonl_paths"] if "/noise_rir/" in path
    ]
    expected[0]["add_task_info"] = TASK_INFO
    expected[0]["post_process"]["add_field"]["fields"]["data_source"] = "noisy_enht_v1"

    actual = load_config(NOISE_RIR)
    assert actual == expected
    assert len(actual[0]["jsonl_paths"]) == 65
    assert len(set(actual[0]["jsonl_paths"])) == 65

    lines = (TRAIN_ROOT / f"{NOISE_RIR}.yaml").read_text().splitlines()
    for index, line in enumerate(lines):
        if line.startswith("  - az://"):
            comments = []
            previous = index - 1
            while previous >= 0 and lines[previous].startswith("  #"):
                comments.append(lines[previous])
                previous -= 1
            assert any("Declared input examples:" in comment for comment in comments)
            assert any("shards:" in comment for comment in comments)
            assert any(
                "Audio hours/duration: unavailable" in comment or "Upstream summary:" in comment
                for comment in comments
            )
    text = "\n".join(lines)
    assert "Exact manifest rows: 60,958" in text
    assert "Exact manifest rows: 474,655" in text
    assert "NOT a verified row or training count" in text


def test_noisy_ls_config_removes_number_filter_and_sets_source():
    expected = load_config("gen_ls_raw_rp_edge_nodigits_aug_audio")
    expected["add_task_info"] = TASK_INFO
    del expected["post_process"]["filter_text_with_numbers"]
    expected["post_process"]["add_field"]["fields"]["data_source"] = "noisy_ls"
    assert load_config(NOISY_LS) == expected
    text = (TRAIN_ROOT / f"{NOISY_LS}.yaml").read_text()
    assert "20,424 rows" in text
    assert "Distinct noise files used: 100" in text
    assert "NO RIR operation" in text
    assert "Hours are sampled estimates" in text


@pytest.mark.parametrize("name", [NOISE_RIR, NOISY_LS])
def test_noisy_configs_compose_under_train_data(name):
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        composed = compose(config_name=None, overrides=[f"+data/train_data={name}"])
    config = load_config(name)
    assert OmegaConf.to_container(composed.data.train_data) == config


@pytest.mark.parametrize("name", [NOISE_RIR, NOISY_LS])
def test_noisy_configs_generate_verbatim_language_hints_and_training_rows(name):
    config = load_config(name)
    if isinstance(config, list):
        config = config[0]
        examples = [
            {
                "id": "entity",
                "metadata": {
                    "audio_file": "/example.wav",
                    "text": "Um, hello Alice.",
                    "desc": "Um, hello <NE:person>Alice</NE:person>.",
                    "whisper_language": "english",
                },
            },
            {
                "id": "no-entity",
                "metadata": {
                    "audio_file": "/example.wav",
                    "text": "Hello.",
                    "desc": "Hello.",
                    "whisper_language": "english",
                },
            },
        ]
        expected_source = "noisy_enht_v1"
        expected_rows = 1
    else:
        examples = [
            {"id": "ls", "audio_path": "/example.wav", "text": "Um, hello Alice.", "keywords": []},
            {"id": "digits", "audio_path": "/example.wav", "text": "123", "keywords": []},
        ]
        expected_source = "noisy_ls"
        expected_rows = 2

    result = augment(Dataset.from_list(examples), version=2609, **config)
    assert len(result) == expected_rows
    assert all(row["data_source"] == expected_source for row in result)
    if name == NOISY_LS:
        assert result[1]["reward_model"]["ground_truth"] == "123"
        assert result[1]["reward_model"]["gt_output"] == (
            "<src=English><tgt=English>\n<VERBATIM>\n<TXT>123</TXT>"
        )
    row = result[0]
    assert row["data_source"] == expected_source
    assert row["audio_path"] == "/example.wav"
    assert row["extra_info"]["prefix"] == ""
    assert row["reward_model"]["ground_truth"] == "Um, hello Alice."
    assert row["reward_model"]["gt_output"] == (
        "<src=English><tgt=English>\n<VERBATIM>\n<TXT>Um, hello Alice.</TXT>"
    )
    assert row["prompt"][0]["content"] == (
        "Transcribe the audio clip into text. Transcribe verbatim, including all filler "
        "words and disfluencies.<audio>\nThe language is English."
    )

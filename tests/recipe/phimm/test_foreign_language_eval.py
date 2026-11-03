from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from recipe.phimm.data.dataset import augment
from recipe.phimm.reward.asr_eval import foreign_language_eval
from recipe.phimm.reward.long_audio_grouped import LongAudioGroupedRewardManager
from verl import DataProto
from verl.trainer.ppo.metric_utils import process_validation_metrics, update_var2metric2val
from verl.trainer.ppo.reward import get_reward_fn_dispatcher


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "recipe/phimm/config"


def foreign_config(name="remax_foreign_base0_gt_s200_remote_v1"):
    searchpath = f"hydra.searchpath=[file://{CONFIG},file://{ROOT / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(CONFIG / "v2611_dev/foreign"), version_base=None):
        return compose(config_name=name, overrides=[searchpath])


@pytest.mark.parametrize(
    "response,reference,accuracy",
    [
        ("<src=English><tgt=English>\n<VERBATIM>\n<foreign_language>", "<foreign_language>", 1),
        ("<FOREIGN_LANGUAGE>", "<foreign_language>", 1),
        ("<foreign_language> extra words", "<foreign_language>", 1),
        ("<nonspeech>", "<foreign_language>", 0),
        ("", "<foreign_language>", 1),
        ("   ", "<foreign_language>", 1),
        ("foreign language", "<foreign_language>", 0),
        ("</foreign_language>", "<foreign_language>", 1),
        ("</FOREIGN_LANGUAGE>", None, 1),
        ("<foreign_language>", "English transcript", 1),
        ("English transcript", "English transcript", 0),
        ("<foreign_language>", None, 1),
        ("", None, 1),
    ],
)
def test_foreign_language_marker_accuracy(response, reference, accuracy):
    result = foreign_language_eval(response, reference, version=2609)
    assert result == {"score": float(accuracy), "n_err": 1 - accuracy, "n_ref": 1}


def test_foreign_language_marker_does_not_require_ground_truth():
    assert foreign_language_eval("<foreign_language>", version=2609)["score"] == 1


def test_foreign_language_marker_is_checked_in_parsed_text_not_header():
    reference = "<foreign_language>"
    response = "Audio Language: <foreign_language>\n<ASR><lang=English><TXT>hello</TXT></ASR>"
    assert foreign_language_eval(response, reference, version=2607)["score"] == 0
    response = "Audio Language: English\n<ASR><lang=English><TXT><foreign_language></TXT></ASR>"
    assert foreign_language_eval(response, reference, version=2607)["score"] == 1


def test_foreign_language_uses_solution_str_not_extra_info():
    result = foreign_language_eval("<foreign_language>", "<foreign_language>", version=2609,
                                   extra_info={"raw_response": "<nonspeech>"})
    assert result["score"] == 1


@pytest.mark.parametrize("name", ["remax_foreign_base0_gt_s200_remote_v1", "remax_foreign_base0_gt_s200_1k_v1"])
def test_heldout_data_uses_shared_foreign_source_with_english_hint(name):
    cfg = foreign_config(name)
    english, foreign = [OmegaConf.to_container(item, resolve=True) for item in cfg.data.val_data[1]]
    languages = ["en", "de", "es", "fr", "it", "nl", "pl", "pt", "zh", "ja", "ko"]
    rows = [{"id": lang, "text": f"sample {lang}", "language": lang,
             "source_language": lang, "data_source": "unmodified_source",
             "audio_path": f"{lang}.wav"} for lang in languages]
    english_output = augment(Dataset.from_list(rows[:1]), version="2609", **english)
    foreign_output = augment(Dataset.from_list(rows[1:]), version="2609", **foreign)

    assert english_output[0]["data_source"] == "enhint_en"
    assert english_output[0]["reward_model"]["ground_truth"] == "sample en"
    assert all(row["data_source"] == "enhint_noen" for row in foreign_output)
    assert all(row["reward_model"]["ground_truth"] == "<foreign_language>"
               for row in foreign_output)
    outputs = list(english_output) + list(foreign_output)
    assert [row["extra_info"]["id"] for row in outputs] == languages
    assert all("source_language" not in row["extra_info"] for row in outputs)
    assert all("The language is English." in row["prompt"][0]["content"] for row in outputs)
    assert all("Transcribe verbatim" in row["prompt"][0]["content"] for row in outputs)


def test_validation_manager_scores_cleaned_foreign_and_grouped_english():
    cfg = foreign_config()
    scorer = get_reward_fn_dispatcher(cfg.val_reward)
    calls = []

    def compute_score(**kwargs):
        calls.append(kwargs)
        return scorer(**kwargs)

    prefix = "<src=English><tgt=English>\n<VERBATIM>\n"
    responses = {1: prefix + "<foreign_language>", 2: prefix + "<nonspeech>",
                 3: prefix + "bonjour", 4: prefix,
                 5: prefix + "hello world", 6: prefix + "hello", 7: prefix + "world"}

    class Tokenizer:
        eos_token = None

        def decode(self, token_ids, skip_special_tokens=True):
            return responses[int(token_ids[0])]

    sources = ["enhint_noen", "enhint_noen", "enhint_noen", "enhint_noen", "enhint_en",
               "earnings22_parent4", "earnings22_parent4"]
    info = [{"id": str(i)} for i in range(5)] + [
        {"id": "parent-0", "parent_audio_path": "parent.wav", "seg_start": 0},
        {"id": "parent-1", "parent_audio_path": "parent.wav", "seg_start": 1},
    ]
    data = DataProto.from_dict(
        tensors={"prompts": torch.ones((7, 1), dtype=torch.long),
                 "responses": torch.arange(1, 8).reshape(7, 1),
                 "attention_mask": torch.ones((7, 2), dtype=torch.long)},
        non_tensors={"extra_info": np.array(info, dtype=object),
                     "reward_model": np.array([{"ground_truth": text} for text in
                         ["<foreign_language>"] * 4 + ["hello world"] * 3],
                         dtype=object),
                     "data_source": np.array(sources, dtype=object)},
    )
    manager = LongAudioGroupedRewardManager(
        Tokenizer(), 0, compute_score=compute_score,
        **OmegaConf.to_container(cfg.val_reward.reward_kwargs, resolve=True),
    )
    result = manager(data, return_dict=True)
    assert [call["solution_str"] for call in calls] == [
        "", "", "bonjour", "", "hello world", "hello\nworld",
    ]
    assert all("raw_response" not in call["extra_info"] for call in calls)
    extra = result["reward_extra_info"]
    assert "accuracy" not in extra
    assert extra["n_err"] == [0, 0, 1, 0, 0, 0, None]
    assert extra["n_ref"] == [1, 1, 1, 1, 2, 2, None]
    assert result["reward_tensor"].flatten().tolist() == [1, 1, 0, 1, 1, 1, 1]

    metrics = process_validation_metrics(sources, list("abcdefg"), extra)
    foreign = update_var2metric2val(metrics["enhint_noen"])
    assert foreign["p_err"]["mean@1"] == 0.25
    assert update_var2metric2val(metrics["earnings22_parent4"])["p_err"]["mean@1"] == 0.0

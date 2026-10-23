from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from recipe.phimm.data.keyword_output import PROTOCOL
from recipe.phimm.data.prompts import get_task_output
from recipe.phimm.reward.asr_keywords import eval_score
from recipe.phimm.reward.keyword_manager import ASRKeywordsRewardManager


class CharTokenizer:
    eos_token = "<EOS>"
    eos_token_id = 999

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(int(t)) for t in tokens if int(t) not in (0, 999))


@pytest.mark.parametrize("finish,valid", [("stop", 1), ("length", 0), ("abort", 0)])
def test_native_termination_wins_over_full_budget_without_eos(finish, valid):
    answer = get_task_output(task="lang_asr_verb", text="the", version=2609, output_keywords=True, keywords=[])
    response = torch.tensor([list(answer.encode())])
    extra = {"output_protocol": PROTOCOL, "keyword_annotation_valid": True, "keywords": []}
    data = DataProto.from_dict(
        tensors={
            "prompts": torch.ones((1, 1), dtype=torch.long),
            "responses": response,
            "attention_mask": torch.ones((1, 1 + response.shape[-1]), dtype=torch.long),
        },
        non_tensors={
            "extra_info": np.array([extra], dtype=object),
            "reward_model": np.array([{"ground_truth": "the"}], dtype=object),
            "data_source": np.array(["ls_clean"], dtype=object),
            "rollout_finish_reason": np.array([finish], dtype=object),
        },
    )
    manager = ASRKeywordsRewardManager(
        CharTokenizer(),
        0,
        compute_score=eval_score,
        max_resp_len=response.shape[-1],
        overlong_buffer_cfg=SimpleNamespace(enable=False, len=0),
    )
    result = manager(data, return_dict=True)
    assert result["reward_extra_info"]["kw_list_f1"] == [valid]
    assert extra["generation_finish_reason_source"] == "vllm"


def test_candidate_metadata_does_not_alias_and_validation_dump_is_enriched():
    answer = get_task_output(task="lang_asr_verb", text="rare", version=2609, output_keywords=True, keywords=["rare"])
    # First candidate is complete; second uses its entire budget without EOS.
    good = list(answer.encode()) + [999]
    bad = list(answer.encode()) + [ord("x")]
    response = torch.tensor([good, bad])
    shared = {"output_protocol": PROTOCOL, "keyword_annotation_valid": True, "keywords": ["rare"]}
    data = DataProto.from_dict(
        tensors={
            "prompts": torch.ones((2, 1), dtype=torch.long),
            "responses": response,
            "attention_mask": torch.ones((2, 1 + len(good)), dtype=torch.long),
        },
        non_tensors={
            "extra_info": np.array([shared, shared], dtype=object),
            "reward_model": np.array([{"ground_truth": "rare"}] * 2, dtype=object),
            "data_source": np.array(["ls_clean"] * 2, dtype=object),
        },
    )
    manager = ASRKeywordsRewardManager(
        CharTokenizer(),
        0,
        compute_score=lambda **kw: eval_score(**kw),
        max_resp_len=len(good),
        overlong_buffer_cfg=SimpleNamespace(enable=False, len=0),
    )
    result = manager(data, return_dict=True)
    assert result["reward_extra_info"]["kw_list_f1"] == [1, 0]
    assert result["reward_extra_info"]["kw_truncated"] == [0, 1]
    assert data.non_tensor_batch["extra_info"][0] is not data.non_tensor_batch["extra_info"][1]
    assert data.non_tensor_batch["extra_info"][0]["predicted_keywords"] == ["rare"]
    # A unique validation dictionary must keep its identity for already-collected dumps.
    unique = data.non_tensor_batch["extra_info"][0]
    manager(data, return_dict=True)
    assert data.non_tensor_batch["extra_info"][0] is unique
    assert unique["parsed_transcription"] == "rare"


def test_composed_config_loads_actual_train_and_validation_dispatchers():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    from verl.trainer.ppo.reward import load_reward_manager

    root = Path(__file__).resolve().parents[4] / "recipe/phimm/config"
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(config_dir=str(root), version_base=None):
        config = compose(config_name="2609_kw/training/remax_ls_joint_pilot")
        OmegaConf.resolve(config)
    answer = get_task_output(task="lang_asr_verb", text="rare", version=2609, output_keywords=True, keywords=["rare"])
    extra = {"output_protocol": PROTOCOL, "keyword_annotation_valid": True, "keywords": ["rare"]}
    train = load_reward_manager(
        config, CharTokenizer(), 0, max_resp_len=768, overlong_buffer_cfg=config.reward_model.overlong_buffer
    )
    train_result = train.compute_score(
        data_source="ls_joint_train", solution_str=answer, ground_truth="rare", extra_info=extra
    )
    assert train_result["kw_list_f1"] == 1
    assert train_result["score"] == train_result["asr_score"] + 0.6
    # Same overrides as TaskRunner, exercising the real file-based dispatcher.
    validation_config = OmegaConf.merge(config)
    with open_dict(validation_config):
        for key in ("reward_functions", "reward_function_by_data_source"):
            validation_config[key] = config.val_reward[key]
    validation = load_reward_manager(
        validation_config,
        CharTokenizer(),
        0,
        max_resp_len=768,
        overlong_buffer_cfg=config.reward_model.overlong_buffer,
    )
    for source in ("ls_clean", "ls_other", "ls_dev", "ls_internal"):
        result = validation.compute_score(
            data_source=source, solution_str=answer, ground_truth="rare", extra_info=extra
        )
        assert result["n_err"] == result["nb_err"] == result["nu_err"] == 0
        assert result["kw_list_f1"] == 1

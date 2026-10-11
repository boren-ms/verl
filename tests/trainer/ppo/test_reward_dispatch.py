import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import (
    capture_remax_metric_baselines,
    compute_remax_metric_accumulation,
    compute_reward_component_metrics,
)
from verl.trainer.ppo.ray_trainer import _compute_reward_metrics
from verl.trainer.ppo.reward import get_reward_fn_dispatcher, load_reward_manager
from verl.trainer.ppo.filter_groups import compute_remax_advantage_metric
from verl.workers.reward_manager.naive import NaiveRewardManager


def test_reward_dispatches_by_data_source_and_preserves_kwargs(tmp_path):
    reward_file = tmp_path / "reward.py"
    reward_file.write_text(
        "def compute_score(solution_str, ground_truth, label, **kwargs):\n"
        "    return {'score': label, 'data_source': kwargs['data_source']}\n"
    )
    config = OmegaConf.create(
        {
            "reward_functions": {
                "mix_lang": {
                    "path": str(reward_file),
                    "name": "compute_score",
                    "reward_kwargs": {"label": 1.0},
                },
            },
            "reward_function_by_data_source": {
                "mix_cv15_all": "mix_lang",
                "mix_cv15_tier1": "mix_lang",
            },
            "custom_reward_function": {
                "path": str(reward_file),
                "name": "compute_score",
                "reward_kwargs": {"label": 0.5},
            },
        }
    )

    reward_fn = get_reward_fn_dispatcher(config)

    assert reward_fn(data_source="mix_cv15_all", solution_str="response", ground_truth="reference") == {
        "score": 1.0,
        "data_source": "mix_cv15_all",
    }
    assert reward_fn(data_source="mix_cv15_tier1", solution_str="response", ground_truth="reference") == {
        "score": 1.0,
        "data_source": "mix_cv15_tier1",
    }
    assert reward_fn(data_source="other_source", solution_str="response", ground_truth="reference") == {
        "score": 0.5,
        "data_source": "other_source",
    }


def test_load_naive_reward_manager_accepts_shared_manager_kwargs():
    config = OmegaConf.create(
        {
            "reward_model": {"reward_manager": "naive", "sandbox_fusion": {}},
            "data": {"reward_fn_key": "data_source"},
        }
    )

    reward_manager = load_reward_manager(
        config,
        tokenizer=object(),
        num_examine=0,
        max_resp_len=1024,
        overlong_buffer_cfg=None,
    )

    assert isinstance(reward_manager, NaiveRewardManager)
    assert reward_manager.reward_kwargs == {"max_resp_len": 1024, "overlong_buffer_cfg": None}


@pytest.mark.parametrize("manager", ["naive", "dapo", "naive_parallel", "dapo_parallel", "batch"])
@pytest.mark.parametrize("explicit_none", [False, True])
def test_remax_missing_source_metrics_survive_reward_managers(tmp_path, manager, explicit_none):
    reward_file = tmp_path / "optional_reward.py"
    reward_file.write_text(
        "def compute_score(data_source=None, data_sources=None, **kwargs):\n"
        "    def score(source):\n"
        f"        result = {{'char': None, 'word': None}} if {explicit_none!r} else {{}}\n"
        "        result.update({'score': 0.5, 'char': 0.875} if source == 'char_source' else "
        "{'score': 0.5, 'word': 0.875})\n"
        "        return result\n"
        "    return [score(source) for source in data_sources] if data_sources is not None else score(data_source)\n"
    )
    algorithm = {
        "adv_estimator": "remax",
        "remax_measures": {
            "char": {"delta": 0.25},
            "word": {"delta": 0.25},
            "absent": {"delta": 0.25},
        },
    }
    config = OmegaConf.create(
        {
            "algorithm": algorithm,
            "reward_model": {"reward_manager": manager},
            "data": {"reward_fn_key": "data_source"},
            "custom_reward_function": {"path": str(reward_file), "name": "compute_score"},
        }
    )

    class Tokenizer:
        eos_token = "<eos>"

        def decode(self, *args, **kwargs):
            return "response"

    manager_kwargs = {}
    if manager.startswith("dapo"):
        manager_kwargs = {"max_resp_len": 2, "overlong_buffer_cfg": OmegaConf.create({"enable": False, "len": 0})}
    reward_manager = load_reward_manager(config, Tokenizer(), num_examine=0, **manager_kwargs)
    data = DataProto.from_dict(
        tensors={
            "prompts": torch.ones((2, 1), dtype=torch.long),
            "responses": torch.ones((2, 2), dtype=torch.long),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
        },
        non_tensors={
            "data_source": np.array(["char_source", "word_source"]),
            "reward_model": np.array([{"ground_truth": "ref"}] * 2, dtype=object),
            "extra_info": np.array([{}, {}], dtype=object),
        },
    )
    result = reward_manager(data, return_dict=True)
    sampled = result["reward_extra_info"]
    assert sampled["char"] == [0.875, None]
    assert sampled["word"] == [None, 0.875]
    assert "absent" not in sampled
    assert not any(key.startswith("remax_metric_present_") for key in sampled)
    baseline_extra = {**sampled, "char": [0.5, None], "word": [None, 0.5]}
    batch = {}
    algo = AlgoConfig(**algorithm)
    capture_remax_metric_baselines(algo, baseline_extra, batch, 2)
    # Availability tensors must travel with their metric baseline values.
    repeated = DataProto.from_dict(tensors=batch).repeat(2, interleave=True)
    repeated.reorder(torch.tensor([3, 0, 2, 1]))
    assert repeated.batch["reward_baselines_char_present"].tolist() == [False, True, False, True]
    score, _ = compute_remax_metric_accumulation(algo, sampled, batch, torch.ones((2, 2)))
    torch.testing.assert_close(score, torch.full((2,), 0.375))
    filtered = compute_remax_advantage_metric(
        np.zeros(2),
        np.zeros(2),
        config=algo,
        non_tensor_batch=sampled,
        batch=batch,
        response_mask=torch.ones((2, 2)),
    )
    np.testing.assert_allclose(filtered, [0.375, 0.375])
    metrics = compute_reward_component_metrics(algo, sampled, batch)
    assert metrics["gdpo/char/mean"] == 0.875
    assert metrics["gdpo/word/base_std"] == 0
    assert not any("absent" in key for key in metrics)
    details = _compute_reward_metrics(sampled)
    assert details["reward-core/char/mean"] == 0.875
    assert not any("present" in key or "absent" in key for key in details)

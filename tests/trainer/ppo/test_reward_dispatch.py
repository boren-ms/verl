from omegaconf import OmegaConf

from verl.trainer.ppo.reward import get_reward_fn_dispatcher, load_reward_manager
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
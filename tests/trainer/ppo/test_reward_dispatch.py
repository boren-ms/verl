from omegaconf import OmegaConf

from verl.trainer.ppo.reward import get_reward_fn_dispatcher


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
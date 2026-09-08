from omegaconf import OmegaConf

from recipe.phimm.reward.config import reward_manager_kwargs
from recipe.phimm.reward.long_audio_grouped import _record_group_result


def test_reward_manager_kwargs_forwards_configured_values():
    config = OmegaConf.create(
        {
            "val_reward": {
                "reward_kwargs": {
                    "version": 2607,
                    "custom_option": "enabled",
                }
            }
        }
    )

    assert reward_manager_kwargs(config, "val_reward") == {
        "version": 2607,
        "custom_option": "enabled",
    }


def test_reward_manager_kwargs_defaults_to_empty():
    assert reward_manager_kwargs(OmegaConf.create({}), "val_reward") == {}


def test_group_result_is_recorded_once_per_parent():
    reward_extra_info = {}

    _record_group_result(
        reward_extra_info,
        {"score": 0.75, "n_err": 2, "n_ref": 8},
        row_count=4,
        head_index=1,
    )

    assert reward_extra_info == {
        "score": [None, 0.75, None, None],
        "n_err": [None, 2, None, None],
        "n_ref": [None, 8, None, None],
    }

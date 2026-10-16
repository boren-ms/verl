from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from recipe.phimm.reward.asr_vote import compute_score


@pytest.mark.parametrize("config_name", ["remax_dev_base0_v0", "remax_dev_base0_v0a", "remax_dev_base0_v0b"])
def test_base0_vote_preserves_training_setup_and_configures_vote_rewards(config_name):
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2611_dev/dev"), version_base=None):
        original = compose(config_name="remax_dev_base0_r0", overrides=[searchpath])
        config = compose(config_name=config_name, overrides=[searchpath])
        v0 = compose(config_name="remax_dev_base0_v0", overrides=[searchpath])

    for section in (
        "actor_rollout_ref",
        "algorithm",
        "data",
        "trainer",
        "val_reward",
        "reward_function_by_data_source",
    ):
        assert OmegaConf.to_container(config[section], resolve=False) == OmegaConf.to_container(
            original[section], resolve=False
        )

    reward = config.reward_functions.librispeech
    assert reward.path == "recipe/phimm/reward/asr_vote.py"
    assert reward.name == "compute_score"
    kwargs = OmegaConf.to_container(reward.reward_kwargs, resolve=True)
    if config_name in ("remax_dev_base0_v0a", "remax_dev_base0_v0b"):
        expected = OmegaConf.to_container(v0, resolve=False)
        for cfg in expected["reward_functions"]["librispeech"]["reward_kwargs"]["measures"].values():
            cfg.update(low=-0.1, high=0.1)
        if config_name == "remax_dev_base0_v0b":
            expected["reward_functions"]["librispeech"]["reward_kwargs"]["measures"]["lang"]["beta"] = 1.0
        assert OmegaConf.to_container(config, resolve=False) == expected
    assert kwargs["version"] == 2609
    assert kwargs["reduce"] == {"mode": "average", "total": 3}
    assert "delta" not in kwargs
    assert set(kwargs["measures"]) == set(original.reward_functions.librispeech.reward_kwargs.measures)
    for name, cfg in kwargs["measures"].items():
        full_weight = name in ("char", "word") or (name == "lang" and config_name == "remax_dev_base0_v0b")
        assert cfg["beta"] == (1.0 if full_weight else 0.25)
        assert cfg["delta"] == ("abs" if name == "lang" else "rel")
        assert cfg["low"] < 0 < cfg["high"]

    kwargs["text_norm"] = "identity"
    baseline = compute_score("<src=English><tgt=English>\n<TXT>cat</TXT>", "cat dog", **kwargs)
    assert baseline["score"] == 0.0
    assert not any(name.startswith("d_") for name in baseline)
    result = compute_score(
        "<src=English><tgt=English>\n<TXT>cat dog</TXT>",
        "cat dog",
        extra_info={"greedy_reward": baseline},
        **kwargs,
    )
    assert result["score"] == pytest.approx(2.25 / 3)
    assert result["d_word"] == 1.0
    assert result["d_lang"] == 0.0
    assert "d_score" not in result

    language_result = compute_score(
        "<src=English><tgt=French>\n<TXT>cat</TXT>",
        "cat dog",
        extra_info={"greedy_reward": baseline},
        **kwargs,
    )
    assert language_result["score"] == pytest.approx(-kwargs["measures"]["lang"]["beta"] / 3)
    assert language_result["v_lang"] == -1.0

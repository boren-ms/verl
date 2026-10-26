"""Check temperature-scaled rollout log-probs in every 2611 ReMax recipe."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


ROOT = Path(__file__).parents[3]
CONFIG_ROOT = ROOT / "recipe/phimm/config"
RECIPE_ROOT = CONFIG_ROOT / "v2611_dev"
REMAX_RECIPES = sorted(RECIPE_ROOT.rglob("remax*.yaml"))


@pytest.mark.parametrize(
    "recipe_path",
    REMAX_RECIPES,
    ids=[str(path.relative_to(RECIPE_ROOT).with_suffix("")) for path in REMAX_RECIPES],
)
def test_remax_recipes_use_expected_behavior_log_probs(recipe_path):
    # Compose from each recipe's own folder so its relative defaults resolve.
    # All recipes inherit temperature-scaled generated-token scores.
    searchpath = f"hydra.searchpath=[file://{CONFIG_ROOT},file://{ROOT / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(recipe_path.parent), version_base=None):
        config = compose(config_name=recipe_path.stem, overrides=[searchpath])

    rollout = config.actor_rollout_ref.rollout
    assert config.algorithm.adv_estimator == "remax"
    assert rollout.engine_kwargs.vllm.get("logprobs_mode", "raw_logprobs") == "raw_logprobs"
    tags = recipe_path.stem.lower().split("_")
    temperature_one = (
        "t1" in tags
        or "base0b" in tags
        or recipe_path.stem == "remax_base0_v1_e1"
    )
    expected_temperature = 2.0 if "t2" in tags else 1.0 if temperature_one else 1.2
    assert rollout.temperature == pytest.approx(expected_temperature)
    assert rollout.calculate_log_probs is True

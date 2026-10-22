"""Check default and opt-in log-prob settings of every 2611 ReMax recipe."""

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
    # Preserve the original vLLM default unless a recipe explicitly opts in.
    searchpath = f"hydra.searchpath=[file://{CONFIG_ROOT},file://{ROOT / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(recipe_path.parent), version_base=None):
        config = compose(config_name=recipe_path.stem, overrides=[searchpath])

    rollout = config.actor_rollout_ref.rollout
    assert config.algorithm.adv_estimator == "remax"
    expected_mode = "processed_logprobs" if recipe_path.stem == "remax_dev_base0_v1h2a_fix_L2_lp" else None
    assert rollout.engine_kwargs.vllm.get("logprobs_mode") == expected_mode
    assert rollout.temperature == pytest.approx(1.2)
    assert rollout.calculate_log_probs is True

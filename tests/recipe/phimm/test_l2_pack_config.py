"""Keep the packing experiment a single-setting comparison with L2."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_l2_pack_changes_only_linear_attention_packing():
    root = Path(__file__).parents[3]
    config_root = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config_root},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config_root / "v2611_dev/dev"), version_base=None):
        baseline = compose(config_name="remax_dev_base0_v1h2a_fix_L2", overrides=[searchpath])
        packed = compose(config_name="remax_dev_base0_v1h2a_fix_L2_pack", overrides=[searchpath])

    assert baseline.actor_rollout_ref.model.use_remove_padding is True
    assert baseline.actor_rollout_ref.model.use_linear_attention_packing is False
    assert packed.actor_rollout_ref.model.use_linear_attention_packing is True
    packed.actor_rollout_ref.model.use_linear_attention_packing = False
    assert OmegaConf.to_container(packed, resolve=False) == OmegaConf.to_container(baseline, resolve=False)

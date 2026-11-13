from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_foreign_base1_uses_own_target_language_data_configs_except_nonenglish_validation():
    root = Path(__file__).resolve().parents[3]
    config = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config / "v2611_dev/foreign"), version_base=None):
        base = compose(config_name="remax_foreign_base0_2611r2", overrides=[searchpath])
        candidate = compose(config_name="remax_foreign_base1_2611r2", overrides=[searchpath])

    # Check every effective source rather than unused Hydra packages.
    for split, expected_count in [("train_data", 2), ("val_data", 7)]:
        before = OmegaConf.to_container(base.data[split], resolve=True)
        after = OmegaConf.to_container(candidate.data[split], resolve=True)
        before_flat = [source for group in before for source in group]
        after_flat = [source for group in after for source in group]
        assert len(after_flat) == expected_count
        for original, targeted in zip(before_flat, after_flat, strict=True):
            assert original["add_task_info"]["task"] == "lang_asr_verb"
            if split == "val_data" and original["post_process"]["add_field"]["fields"]["data_source"] == "enhint_noen":
                assert targeted == original
                continue
            assert targeted["add_task_info"]["task"] == "lang_only_verb"
            assert targeted["add_task_info"]["language"] == "English"
            assert targeted["add_task_info"]["lang_hint"] is True
            targeted["add_task_info"] = original["add_task_info"]
            # Prompt caches use a separate namespace for the new task.
            if "cache_name" in original:
                assert targeted["cache_name"].startswith("auto_foreign_base1_")
                targeted["cache_name"] = original["cache_name"]
            assert targeted == original

    assert [(item.name, item.sample) for item in candidate.data.mix[0].datasource] == [
        ("foreign_1k", 6700), ("foreign_mixed_10k", 6100)
    ]
    for key in ("actor_rollout_ref", "algorithm", "reward_functions", "reward_function_by_data_source", "val_reward"):
        assert OmegaConf.to_container(candidate[key], resolve=False) == OmegaConf.to_container(base[key], resolve=False)
    assert candidate.trainer.experiment_name == "remax_foreign_base1_2611r2"
    candidate.trainer.experiment_name = base.trainer.experiment_name
    assert OmegaConf.to_container(candidate.trainer, resolve=False) == OmegaConf.to_container(base.trainer, resolve=False)
    assert candidate.trainer.total_epochs == 2
    assert candidate.trainer.total_training_steps is None
    assert candidate.data.train_batch_size == 128
    assert candidate.actor_rollout_ref.rollout.gt_rollout is False

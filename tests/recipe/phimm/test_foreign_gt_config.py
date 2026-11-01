from pathlib import Path
from collections import Counter
import json
import unicodedata

from hydra import compose, initialize_config_dir
import pytest


def test_foreign_gt_uses_remote_training_and_both_heldout_validations():
    root = Path(__file__).resolve().parents[3]
    config = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config / "v2611_dev/foreign"), version_base=None):
        cfg = compose(config_name="remax_foreign_base0_gt_s200_remote", overrides=[searchpath])

    assert cfg.actor_rollout_ref.rollout.gt_rollout is True
    assert cfg.actor_rollout_ref.rollout.n == 8
    assert cfg.trainer.total_training_steps == 200
    assert cfg.trainer.total_epochs == 29
    assert cfg.trainer.experiment_name is None
    assert [(item.name, item.sample) for item in cfg.data.mix[0].datasource] == [
        ("noisy_ls", 100), ("mls_foreign", 700), ("cjk_foreign", 100),
    ]
    assert all(cfg.reward_function_by_data_source[source] == "librispeech"
               for source in ("noisy_ls", "mls_foreign", "cjk_foreign"))
    assert "foreign_language" not in cfg.reward_functions
    train = [item for group in cfg.data.train_data for item in group]
    val = [item for group in cfg.data.val_data for item in group]
    assert len(train) == 3 and len(val) == 2
    assert all(item.jsonl_paths.startswith("az://orngwus2cresco/") for item in train + val)
    assert val[0].jsonl_paths.endswith("/earnings22_parent4_segments.jsonl")
    assert val[0].val_batch_size == -1
    assert set(("parent_audio_path", "seg_index", "seg_start", "n_segments")).issubset(
        val[0].post_process.verl_format.extra_keys
    )
    assert cfg.val_reward.reward_manager == "long_audio_grouped"
    assert cfg.actor_rollout_ref.rollout.val_kwargs.n == 1
    assert val[1].jsonl_paths.endswith("/english_hint_heldout.jsonl")
    sources = ["earnings22_parent4"] + [
        f"english_hint_{lang}" for lang in ("en", "de", "es", "fr", "it", "nl", "pl", "pt", "zh", "ja", "ko")
    ]
    assert all(cfg.val_reward.reward_function_by_data_source[source] == "openasr_en" for source in sources)


def test_foreign_gt_1k_uses_remote_manifest_with_english_verbatim_hint():
    root = Path(__file__).resolve().parents[3]
    config = root / "recipe/phimm/config"
    searchpath = f"hydra.searchpath=[file://{config},file://{root / 'verl/trainer/config'}]"
    with initialize_config_dir(config_dir=str(config / "v2611_dev/foreign"), version_base=None):
        cfg = compose(config_name="remax_foreign_base0_gt_s200_1k", overrides=[searchpath])
    item = cfg.data.train_data[0][0]
    assert item.jsonl_paths.startswith("az://orngwus2cresco/")
    assert item.jsonl_paths.endswith("/train_1k.jsonl")
    assert item.add_task_info.task == "lang_asr_verb"
    assert item.add_task_info.language == "English"
    assert item.add_task_info.lang_hint is True
    assert "mask_foreign_language" in item
    assert cfg.data.mix[0].datasource[0].name == "foreign_1k"
    assert cfg.data.mix[0].datasource[0].sample == 11000
    assert cfg.reward_function_by_data_source.foreign_1k == "librispeech"
    assert cfg.actor_rollout_ref.rollout.gt_rollout is True
    assert cfg.trainer.total_training_steps == 200
    assert cfg.trainer.experiment_name is None


_DATA_1K = Path.home() / "data/foreign_language_data/foreign_1k_v1"
_HELDOUT = Path.home() / "data/foreign_language_data/data/english_hint_heldout/all.jsonl"


@pytest.mark.skipif(not (_DATA_1K / "train_1k.jsonl").exists() or not _HELDOUT.exists(),
                    reason="local training and validation audit data is unavailable")
def test_foreign_gt_1k_has_exact_counts_and_no_heldout_text_overlap():
    rows = [json.loads(line) for line in (_DATA_1K / "train_1k.jsonl").read_text().splitlines()]
    validation = [json.loads(line) for line in _HELDOUT.read_text().splitlines()]
    def normalize(text):
        return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())
    assert Counter(row["source_language"] for row in rows) == dict.fromkeys(
        ("en", "de", "es", "fr", "it", "nl", "pl", "pt", "zh", "ja", "ko"), 1000
    )
    assert len({row["id"] for row in rows}) == 11000
    assert not {normalize(row["text"]) for row in rows} & {normalize(row["source_text"]) for row in validation}
    assert not {row["id"] for row in rows} & {row["id"] for row in validation}

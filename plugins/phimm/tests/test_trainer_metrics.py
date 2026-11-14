"""Keep ASR corpus metrics compatible with upstream validation accounting."""

import json

import pytest
from omegaconf import OmegaConf


@pytest.mark.parametrize("missing_sessions", [False, True])
def test_validation_accounts_for_missing_sessions_and_preserves_corpus_counts(tmp_path, missing_sessions):
    from recipe.phimm.agentic.trainer import ASRTrainerSync

    trainer = ASRTrainerSync.__new__(ASRTrainerSync)
    trainer.config = OmegaConf.create(
        {"trainer": {"validation_data_dir": str(tmp_path)}, "asr": {"validation_remote_dir": None}}
    )
    trainer.global_steps = 20
    expected = {("source", "prompt1"): 4, ("source", "prompt2"): 4} if missing_sessions else None
    metrics = trainer._val_metrics_update(
        ["source", "source"],
        ["prompt1", "prompt1"],
        {
            "reward": [0.4, 0.6],
            "acc": [1.0, 1.0],
            "n_err": [1, 0],
            "n_ref": [10, 10],
            "asr_data_source": ["source", "source"],
        },
        [2, 2],
        expected_acc_counts=expected,
    )

    assert metrics["val-core/source/p_err/mean@1"] == 0.05
    assert metrics[f"val-core/source/acc/mean@{4 if missing_sessions else 2}"] == (
        0.25 if missing_sessions else 1.0
    )
    saved = json.loads((tmp_path / "20.metrics.json").read_text())
    assert saved == {"step": 20, "metrics": metrics}

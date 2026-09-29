import numpy as np
import pytest

from verl.trainer.ppo.ray_trainer import _compute_reward_metrics


def test_compute_reward_metrics_splits_core_and_aux_values():
    metrics = _compute_reward_metrics(
        {
            "score": [0.8, 0.6, None],
            "wer": np.array([0.2, 0.4, np.nan]),
            "is_valid": [True, False, True],
            "detail": [{"error": "substitution"}, {"error": "deletion"}],
        }
    )

    assert metrics == pytest.approx(
        {
            "reward-core/score/mean": 0.7,
            "reward-core/score/std": 0.1,
            "reward-aux/score/max": 0.8,
            "reward-aux/score/min": 0.6,
            "reward-core/wer/mean": 0.3,
            "reward-core/wer/std": 0.1,
            "reward-aux/wer/max": 0.4,
            "reward-aux/wer/min": 0.2,
            "reward-core/is_valid/mean": 2 / 3,
            "reward-core/is_valid/std": np.std([1.0, 0.0, 1.0]),
            "reward-aux/is_valid/max": 1.0,
            "reward-aux/is_valid/min": 0.0,
        }
    )

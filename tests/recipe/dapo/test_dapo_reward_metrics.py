from types import SimpleNamespace

import numpy as np
import pytest

from recipe.dapo.dapo_ray_trainer import _compute_retained_reward_metrics


def test_compute_retained_reward_metrics_uses_final_training_batch():
    batch = SimpleNamespace(
        non_tensor_batch={
            "score": np.array([0.9, 0.5]),
            "wer": np.array([0.1, 0.3]),
            "data_source": np.array(["mix", "openml"]),
        }
    )

    metrics = _compute_retained_reward_metrics(batch, {"score", "wer", "missing"})

    assert metrics == pytest.approx(
        {
            "reward-core/score/mean": 0.7,
            "reward-core/score/std": 0.2,
            "reward-aux/score/max": 0.9,
            "reward-aux/score/min": 0.5,
            "reward-core/wer/mean": 0.2,
            "reward-core/wer/std": 0.1,
            "reward-aux/wer/max": 0.3,
            "reward-aux/wer/min": 0.1,
        }
    )

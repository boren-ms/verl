# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import torch

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.filter_groups import compute_remax_advantage_metric, select_prompt_uids_by_metric


def test_compute_remax_advantage_metric_subtracts_greedy_baseline():
    sampled_rewards = np.array([0.7, 0.4, 0.5, 0.5])
    reward_baselines = np.array([0.5, 0.5, 0.5, 0.5])

    metric_values = compute_remax_advantage_metric(sampled_rewards, reward_baselines)

    np.testing.assert_allclose(metric_values, [0.2, -0.1, 0.0, 0.0])


def test_variance_mode_preserves_existing_behavior():
    prompt_uids = np.array(["varying", "varying", "equal", "equal", "single"])
    metric_values = np.array([0.0, 1.0, 2.0, 2.0, 0.0])

    kept = select_prompt_uids_by_metric(prompt_uids, metric_values)

    assert kept == ["varying", "single"]


def test_nonzero_mode_keeps_any_nonzero_advantage():
    prompt_uids = np.array(["all_zero", "all_zero", "one_nonzero", "one_nonzero", "equal_nonzero", "equal_nonzero"])
    metric_values = np.array([0.0, 1e-7, 0.0, -0.25, 0.5, 0.5])

    kept = select_prompt_uids_by_metric(prompt_uids, metric_values, mode="nonzero", atol=1e-6)

    assert kept == ["one_nonzero", "equal_nonzero"]


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="Unsupported filter group mode"):
        select_prompt_uids_by_metric(np.array(["prompt"]), np.array([1.0]), mode="unknown")
    with pytest.raises(ValueError, match="atol must be non-negative"):
        select_prompt_uids_by_metric(np.array(["prompt"]), np.array([1.0]), atol=-1.0)


@pytest.mark.parametrize("mode", ["variance", "nonzero"])
def test_missing_filter_values_are_not_fabricated(mode):
    assert select_prompt_uids_by_metric(
        np.array(["absent", "absent", "useful", "useful"]),
        np.array([None, None, None, 0.5], dtype=object),
        mode=mode,
    ) == ["useful"]
    with pytest.raises(ValueError, match="finite"):
        select_prompt_uids_by_metric(np.array(["bad"]), np.array([float("nan")]), mode=mode)


def test_metric_filter_uses_raw_stopped_deltas_not_scalar_rewards_or_transforms():
    config = AlgoConfig(
        adv_estimator="remax",
        binary_adv=True,
        adv_scale=100,
        norm_adv_in_remax="l2",
        remax_measures={"char": {"delta": 0.25}, "word": {"beta": 2, "delta": 0}},
    )
    sampled = {"char": np.array([0.875, 0.125, 0.5, 0.875]), "word": np.ones(4)}
    batch = {"reward_baselines_char": torch.full((4,), 0.5), "reward_baselines_word": torch.ones(4)}
    score = compute_remax_advantage_metric(
        np.ones(4),
        np.zeros(4),
        config=config,
        non_tensor_batch=sampled,
        batch=batch,
        response_mask=torch.tensor([[1], [1], [1], [0]]),
    )
    np.testing.assert_allclose(score, [0.375, -0.375, 0, 0])
    assert select_prompt_uids_by_metric(np.array(["useful", "useful", "empty", "empty"]), score, mode="nonzero") == [
        "useful"
    ]


def test_metric_filter_requires_metric_data_and_response_mask():
    config = AlgoConfig(adv_estimator="remax", remax_measures={"char": {"delta": 0}})
    with pytest.raises(ValueError, match="response mask"):
        compute_remax_advantage_metric(np.ones(1), np.zeros(1), config=config)
    with pytest.raises(ValueError, match="sampled metric data"):
        compute_remax_advantage_metric(np.ones(1), np.zeros(1), config=config, response_mask=torch.ones((1, 1)))

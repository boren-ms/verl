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

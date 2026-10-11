# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import unittest
from dataclasses import replace

import numpy as np
import pytest
import torch

import verl.trainer.ppo.core_algos
from verl import DataProto
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import (
    capture_remax_metric_baselines,
    compute_gae_advantage_return,
    compute_gdpo_outcome_advantage,
    compute_grpo_outcome_advantage,
    compute_remax_disagreement_mask,
    compute_remax_metric_accumulation,
    compute_remax_outcome_advantage,
    filter_nonfinite_log_probs,
    get_adv_estimator_fn,
    get_remax_reward_keys,
    kl_penalty,
    register_adv_est,
)


@pytest.mark.parametrize(
    "greedy,sampled,expected",
    [
        ([1, 2, 9], [1, 2, 9], [0, 0, 0]),
        ([1, 2, 9], [1, 3, 9], [0, 1, 0]),
        ([1, 2, 9], [1, 3, 2, 9], [0, 1, 0, 0]),
        ([1, 2, 9], [2, 9], [1, 0]),
        ([1, 2, 3, 9], [1, 3, 9], [0, 1, 0]),
        ([1, 2, 9], [1, 9], [0, 1]),
        ([1, 2, 9], [1], [1]),
        ([1, 2, 3, 9], [1, 4, 9], [0, 1, 1]),
        ([1, 2, 9], [1, 3, 4, 9], [0, 1, 1, 0]),
        ([1, 2, 9], [9], [1]),
        ([1, 1, 2, 9], [1, 2, 9], [1, 0, 0]),
        ([101, 1, 102, 9], [103, 1, 102, 9], [1, 0, 0, 0]),
        ([], [1, 9], [1, 1]),
        ([1, 9], [], []),
        ([], [], []),
    ],
)
def test_remax_disagreement_mask_deletion_boundaries(greedy, sampled, expected):
    # Deliberately use different widths and padding IDs that also occur in valid positions.
    response_ids = torch.tensor([sampled + [9, 9]], dtype=torch.long)
    response_mask = torch.tensor([[1] * len(sampled) + [0, 0]])
    baseline_ids = torch.tensor([greedy + [1]], dtype=torch.long)
    baseline_mask = torch.tensor([[1] * len(greedy) + [0]])

    actual = compute_remax_disagreement_mask(response_ids, response_mask, baseline_ids, baseline_mask)
    assert actual.tolist() == [expected + [0, 0]]
    assert actual.dtype == response_mask.dtype
    assert actual.device == response_mask.device


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("edit_boundary", [False, False, True, False]),
        ("first_diff", [False, False, True, True]),
    ],
)
def test_remax_disagreement_mask_noncontiguous_valid_positions(mode, expected):
    actual = compute_remax_disagreement_mask(
        torch.tensor([[1, 99, 3, 9]]),
        torch.tensor([[True, False, True, True]]),
        torch.tensor([[1, 2, 3, 9, 0]]),
        torch.tensor([[1, 1, 1, 1, 0]]),
        mode=mode,
    )
    assert actual.tolist() == [expected]


@pytest.mark.parametrize(
    "greedy,sampled,expected",
    [
        ([1, 2, 9], [1, 2, 9], [0, 0, 0]),
        ([1, 2, 9], [1, 3, 9], [0, 1, 1]),
        ([1, 2, 9], [1, 3, 2, 9], [0, 1, 1, 1]),
        ([1, 2, 9], [2, 9], [1, 1]),
        ([1, 2, 3, 9], [1, 3, 9], [0, 1, 1]),
        ([1, 2, 9], [1, 9], [0, 1]),
        ([1, 2, 9], [1], [0]),
        ([1, 2], [1, 2, 9], [0, 0, 1]),
        ([1, 1, 2, 9], [1, 2, 9], [0, 1, 1]),
        ([101, 1, 102, 9], [103, 1, 102, 9], [1, 1, 1, 1]),
        ([], [1, 9], [1, 1]),
        ([1, 9], [], []),
        ([], [], []),
    ],
)
def test_remax_disagreement_mask_from_first_difference(greedy, sampled, expected):
    response_ids = torch.tensor([sampled + [9, 9]], dtype=torch.long)
    response_mask = torch.tensor([[1] * len(sampled) + [0, 0]])
    baseline_ids = torch.tensor([greedy + [1]], dtype=torch.long)
    baseline_mask = torch.tensor([[1] * len(greedy) + [0]])

    actual = compute_remax_disagreement_mask(
        response_ids, response_mask, baseline_ids, baseline_mask, mode="first_diff"
    )
    assert actual.tolist() == [expected + [0, 0]]
    assert actual.dtype == response_mask.dtype
    assert actual.device == response_mask.device


def test_remax_disagreement_mask_explicit_edit_mode_preserves_default():
    ids = torch.tensor([[1, 3, 9]])
    mask = torch.ones_like(ids)
    baseline = torch.tensor([[1, 2, 9]])
    default = compute_remax_disagreement_mask(ids, mask, baseline, mask)
    explicit = compute_remax_disagreement_mask(ids, mask, baseline, mask, mode="edit_boundary")
    assert torch.equal(default, explicit)


@pytest.mark.parametrize("mode", ["unknown", "", None, "first_difference", "1stdiff"])
def test_remax_disagreement_mask_rejects_invalid_mode(mode):
    ids = torch.ones((1, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="advantage mask mode"):
        compute_remax_disagreement_mask(ids, ids, ids, ids, mode=mode)


@pytest.mark.parametrize("index", [np.array([-1]), np.array([1]), np.array([0.0]), np.array([0, 0])])
def test_remax_disagreement_mask_rejects_invalid_baseline_index(index):
    with pytest.raises(ValueError, match="baseline_index"):
        compute_remax_disagreement_mask(
            torch.ones((1, 2)), torch.ones((1, 2)), torch.ones((1, 3)), torch.ones((1, 3)), index
        )


@pytest.mark.parametrize("invalid_side", ["response", "baseline"])
def test_remax_disagreement_mask_rejects_invalid_shapes(invalid_side):
    mask = torch.ones((1, 2))
    with pytest.raises(ValueError, match="two-dimensional shape"):
        compute_remax_disagreement_mask(
            torch.ones((1, 3)) if invalid_side == "response" else mask,
            mask,
            torch.ones(2) if invalid_side == "baseline" else mask,
            mask,
        )


def test_filter_nonfinite_log_probs_keeps_finite_trajectories_and_dp_alignment():
    batch = DataProto.from_dict(
        tensors={
            "response_mask": torch.tensor([[1, 0]] * 5),
            "old_log_probs": torch.tensor(
                [[0.0, float("nan")], [float("nan"), 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            ),
            "ref_log_prob": torch.tensor([[0.0, 0.0]] * 5),
            "rollout_log_probs": torch.tensor([[0.0, 0.0]] * 5),
            "sample_id": torch.arange(5),
        },
        non_tensors={"uid": np.array(["a", "b", "c", "d", "e"], dtype=object)},
    )
    batch.batch["ref_log_prob"][3, 0] = float("inf")

    invalid_counts = {}
    filtered, n_invalid, n_removed = filter_nonfinite_log_probs(batch, dp_size=2, invalid_counts=invalid_counts)

    assert (n_invalid, n_removed) == (2, 3)
    assert invalid_counts == {"old": 1, "ref": 1, "rollout": 0}
    assert filtered.batch["sample_id"].tolist() == [0, 2]
    assert filtered.non_tensor_batch["uid"].tolist() == ["a", "c"]
    assert len(batch) == 5


def test_filter_nonfinite_log_probs_counts_overlapping_sources_once_per_trajectory():
    batch = DataProto.from_dict(
        tensors={
            "response_mask": torch.tensor([[1, 1], [1, 1], [1, 0], [1, 1], [1, 1]]),
            "old_log_probs": torch.tensor(
                [[float("nan"), float("-inf")], [0.0, 0.0], [0.0, float("nan")], [0.0, 0.0], [0.0, 0.0]]
            ),
            "ref_log_prob": torch.tensor(
                [[float("inf"), 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            ),
            "rollout_log_probs": torch.tensor(
                [[0.0, float("nan")], [float("-inf"), 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            ),
            "sample_id": torch.arange(5),
        }
    )
    invalid_counts = {}

    filtered, n_invalid, n_removed = filter_nonfinite_log_probs(batch, dp_size=2, invalid_counts=invalid_counts)

    assert invalid_counts == {"old": 1, "ref": 1, "rollout": 2}
    assert (n_invalid, n_removed) == (2, 3)
    assert filtered.batch["sample_id"].tolist() == [2, 3]


@pytest.mark.parametrize(
    "key,source", [("old_log_probs", "old"), ("ref_log_prob", "ref"), ("rollout_log_probs", "rollout")]
)
@pytest.mark.parametrize("nonfinite", [None, float("nan"), float("inf"), float("-inf")])
def test_filter_nonfinite_log_probs_counts_missing_and_finite_sources(key, source, nonfinite):
    batch = DataProto.from_dict(
        tensors={
            "response_mask": torch.ones(2, 1),
            key: torch.tensor([[0.0 if nonfinite is None else nonfinite], [0.0]]),
        }
    )
    invalid_counts = {"old": 99, "ref": 99, "rollout": 99}

    filtered, n_invalid, n_removed = filter_nonfinite_log_probs(batch, invalid_counts=invalid_counts)

    expected = {"old": 0, "ref": 0, "rollout": 0}
    expected[source] = int(nonfinite is not None)
    assert invalid_counts == expected
    assert n_invalid == n_removed == int(nonfinite is not None)
    if nonfinite is None:
        assert filtered is batch
    else:
        assert len(filtered) == 1


def test_filter_nonfinite_log_probs_rejects_empty_dp_batch():
    batch = DataProto.from_dict(
        tensors={
            "response_mask": torch.ones(2, 1),
            "old_log_probs": torch.tensor([[float("nan")], [0.0]]),
        }
    )

    with pytest.raises(ValueError, match="No complete DP batch remains"):
        filter_nonfinite_log_probs(batch, dp_size=2)


def mock_test_fn():
    pass


def test_kl_penalty_uses_fp32_for_bf16_inputs():
    logprob = torch.tensor([-1.0, -2.0], dtype=torch.bfloat16)
    ref_logprob = torch.tensor([-1.5, -1.5], dtype=torch.bfloat16)

    result = kl_penalty(logprob, ref_logprob, "low_var_kl")

    assert result.dtype == torch.float32


class TestRegisterAdvEst(unittest.TestCase):
    def setUp(self):
        """Clear the registry before each test"""
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = {
            "gae": lambda x: x * 2,
            "vtrace": lambda x: x + 1,
        }
        self.ADV_ESTIMATOR_REGISTRY = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY

    def tearDown(self) -> None:
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        return super().tearDown()

    def test_register_new_function(self):
        """Test registering a new function with a string name"""

        @register_adv_est("test_estimator")
        def test_fn():
            pass

        self.assertIn("test_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_estimator"], test_fn)

    def test_register_with_enum(self):
        """Test registering with an enum value (assuming AdvantageEstimator exists)"""
        from enum import Enum

        class AdvantageEstimator(Enum):
            TEST = "test_enum_estimator"

        @register_adv_est(AdvantageEstimator.TEST)
        def test_fn():
            pass

        self.assertIn("test_enum_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_enum_estimator"], test_fn)

    def test_duplicate_registration_same_function(self):
        """Test that registering the same function twice doesn't raise an error"""
        register_adv_est("duplicate_test")(mock_test_fn)
        register_adv_est("duplicate_test")(mock_test_fn)

        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["duplicate_test"], mock_test_fn)

    def test_duplicate_registration_different_function(self):
        """Test that registering different functions with same name raises ValueError"""

        @register_adv_est("conflict_test")
        def test_fn1():
            pass

        with self.assertRaises(ValueError):

            @register_adv_est("conflict_test")
            def test_fn2():
                pass

    def test_decorator_preserves_function(self):
        """Test that the decorator returns the original function"""

        def test_fn():
            return "original"

        decorated = register_adv_est("preserve_test")(test_fn)
        self.assertEqual(decorated(), "original")

    def test_multiple_registrations(self):
        """Test registering multiple different functions"""
        init_adv_count = len(self.ADV_ESTIMATOR_REGISTRY)

        @register_adv_est("estimator1")
        def fn1():
            pass

        @register_adv_est("estimator2")
        def fn2():
            pass

        self.assertEqual(len(self.ADV_ESTIMATOR_REGISTRY), 2 + init_adv_count)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator1"], fn1)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator2"], fn2)

    def test_get_adv_estimator_fn_valid_names(self):
        """Test that valid names return the correct function from registry."""
        # Test GAE
        gae_fn = get_adv_estimator_fn("gae")
        assert gae_fn(5) == 10  # 5 * 2 = 10

        # Test Vtrace
        vtrace_fn = get_adv_estimator_fn("vtrace")
        assert vtrace_fn(5) == 6  # 5 + 1 = 6

    def test_get_adv_estimator_fn_invalid_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_adv_estimator_fn("invalid_name")
        assert "Unknown advantage estimator simply: invalid_name" in str(excinfo.value)

    def test_get_adv_estimator_fn_case_sensitive(self):
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_adv_estimator_fn("GAE")  # Different case


def test_multi_turn_compute_gae_advantage_return():
    """Test multi-turn GAE skip observation tokens."""
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)

    rewards = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0]], dtype=torch.float)

    values1 = torch.tensor(
        [
            [
                random.uniform(-100.0, 100.0),
                random.random(),
                4.0,
                5.0,
                6.0,
                random.uniform(-100.0, 0),
                random.random(),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    values2 = torch.tensor(
        [
            [
                random.random(),
                random.uniform(-100.0, 100.0),
                4.0,
                5.0,
                6.0,
                random.random(),
                random.uniform(0.0, 100.0),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float)

    adv1, ret1 = compute_gae_advantage_return(rewards, values1, response_mask, gamma, lam)
    adv2, ret2 = compute_gae_advantage_return(rewards, values2, response_mask, gamma, lam)

    ret1 *= response_mask
    ret2 *= response_mask
    assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    print(f" [CORRECT] \n\n{adv1=}, \n\n{ret1=}")


def test_compute_gdpo_skips_missing_reward_component():
    token_level_rewards = torch.zeros((4, 2), dtype=torch.float)
    response_mask = torch.tensor([[0.0, 1.0]] * 4)
    index = np.array([0, 0, 1, 1])
    batch = {
        "prompts": torch.zeros((4, 1), dtype=torch.long),
        "attention_mask": torch.ones((4, 3), dtype=torch.long),
    }
    non_tensor_batch = {
        "char": np.array([0.0, 1.0, 0.0, 1.0]),
        "lang": np.array([1.0, 0.0, 0.0, 1.0]),
    }

    advantages, returns = compute_gdpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        config=AlgoConfig(
            adv_estimator="gdpo",
            gdpo_reward_keys=["char", "digit_char", "lang"],
            gdpo_reward_weights=[1.0, 100.0, 3.0],
        ),
        non_tensor_batch=non_tensor_batch,
        batch=batch,
    )
    expected_advantages, expected_returns = compute_gdpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        config=AlgoConfig(
            adv_estimator="gdpo",
            gdpo_reward_keys=["char", "lang"],
            gdpo_reward_weights=[1.0, 3.0],
        ),
        non_tensor_batch=non_tensor_batch,
        batch=batch,
    )

    assert torch.allclose(advantages, expected_advantages)
    assert torch.allclose(returns, expected_returns)


@pytest.mark.parametrize(("norm_mode", "expected_scale"), [("l2", 1.0), ("rms", 2**0.5)])
def test_compute_remax_normalizes_weighted_aggregate(norm_mode, expected_scale):
    response_mask = torch.ones((4, 2), dtype=torch.float)
    index = np.array([0, 0, 1, 1])
    batch = {
        "prompts": torch.zeros((4, 1), dtype=torch.long),
        "attention_mask": torch.ones((4, 3), dtype=torch.long),
        "reward_baselines_char": torch.zeros(4),
        "reward_baselines_lang": torch.zeros(4),
    }
    non_tensor_batch = {
        "char": np.array([3.0, 4.0, 0.0, 5.0]),
        "lang": np.array([0.0, 10.0, 12.0, 5.0]),
    }

    advantages, returns = compute_remax_outcome_advantage(
        token_level_rewards=torch.zeros_like(response_mask),
        reward_baselines=torch.zeros(4),
        response_mask=response_mask,
        index=index,
        config=AlgoConfig(
            adv_estimator="remax",
            norm_adv_in_remax=norm_mode,
            remax_measures={"char": {"delta": 0}, "lang": {"beta": 2.0, "delta": 0}},
        ),
        non_tensor_batch=non_tensor_batch,
        batch=batch,
    )

    expected_advantages = (
        torch.tensor([[3 / 585**0.5] * 2, [24 / 585**0.5] * 2, [24 / 801**0.5] * 2, [15 / 801**0.5] * 2])
        * expected_scale
    )
    expected_returns = torch.tensor([[3.0, 3.0], [24.0, 24.0], [24.0, 24.0], [15.0, 15.0]])
    assert torch.allclose(advantages, expected_advantages)
    assert torch.allclose(returns, expected_returns)


def _metric_remax_inputs():
    # Exactly representable deltas make strict threshold equality unambiguous.
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=torch.float)
    sampled = {
        "char": np.array([0.875, 0.125, 0.75, 0.625, 0.875]),
        "word": np.array([0.125, 0.875, 0.875, 0.625, 0.875]),
        "char_del": np.array([1.0, 1.0, 1.0, 0.625, 1.0]),
    }
    batch = {f"reward_baselines_{key}": torch.full((5,), 0.5) for key in sampled}
    config = AlgoConfig(
        adv_estimator="remax",
        remax_measures={key: {"delta": 0.25} for key in sampled},
    )
    return mask, sampled, batch, config


@pytest.mark.parametrize("norm", [None, "l2", "rms"])
@pytest.mark.parametrize("binary", [False, True])
def test_remax_metrics_stop_per_row_and_preserve_returns(norm, binary):
    mask, sampled, batch, config = _metric_remax_inputs()
    config = replace(config, norm_adv_in_remax=norm, binary_adv=binary, adv_scale={"source": 2.0})
    advantages, returns = compute_remax_outcome_advantage(
        torch.zeros_like(mask),
        torch.zeros(5),
        mask,
        config=config,
        index=np.array([0, 0, 0, 0, 0]),
        data_sources=np.array(["source"] * 5),
        non_tensor_batch=sampled,
        batch=batch,
    )
    expected = torch.tensor([0.375, -0.375, 0.625, 0.375, 0.0])
    if norm:
        divisor = expected.norm() if norm == "l2" else expected.square().mean().sqrt()
        expected /= divisor
    if binary:
        expected = expected.sign()
    torch.testing.assert_close(advantages, expected[:, None] * mask * 2)
    torch.testing.assert_close(returns, torch.tensor([0.875, 0.125, 1.625, 1.875, 0.0])[:, None] * mask)


def test_remax_zero_threshold_accumulates_all_metrics_with_weights():
    mask, sampled, batch, config = _metric_remax_inputs()
    config = replace(
        config,
        remax_measures={
            "char": {"beta": 2.0, "delta": 0},
            "word": {"beta": -1.0, "delta": 0},
            "char_del": {"beta": 0.0, "delta": 0},
        },
    )
    advantages, returns = compute_remax_outcome_advantage(
        torch.zeros_like(mask),
        torch.zeros(5),
        mask,
        config=config,
        non_tensor_batch=sampled,
        batch=batch,
    )
    expected_rewards = torch.as_tensor(2 * sampled["char"] - sampled["word"], dtype=torch.float)
    torch.testing.assert_close(advantages, (expected_rewards - 0.5)[:, None] * mask)
    torch.testing.assert_close(returns, expected_rewards[:, None] * mask)


def test_remax_stopping_uses_raw_delta_not_beta_or_accumulated_score():
    mask, sampled, batch, config = _metric_remax_inputs()
    config = replace(
        config,
        remax_measures={
            "char": {"beta": 0.0, "delta": 0.25},
            "word": {"beta": 100.0, "delta": 0},
            "char_del": {"delta": 0.25},
        },
    )
    score, _ = compute_remax_metric_accumulation(config, sampled, batch, mask)
    torch.testing.assert_close(score, torch.tensor([0.0, 0.0, 38.0, 12.625, 0.0]))
    config = replace(config, remax_measures={"word": {"delta": 0.25}, "char": {"delta": 0.25}})
    score, _ = compute_remax_metric_accumulation(config, sampled, batch, mask)
    torch.testing.assert_close(score, torch.tensor([-0.375, 0.375, 0.375, 0.25, 0.0]))


@pytest.mark.parametrize(
    "measures",
    [
        {},
        [],
        {"char": None},
        {"char": {"delta": -1}},
        {"char": {"delta": float("nan")}},
        {"char": {"delta": float("inf")}},
        {"char": {"delta": None}},
        {"char": {"delta": True}},
        {"char": {"delta": 0, "beta": float("inf")}},
        {"char": {"delta": 0, "unknown": 1}},
    ],
)
def test_remax_rejects_invalid_measure_configuration(measures):
    with pytest.raises(ValueError, match="remax_measures"):
        get_remax_reward_keys(AlgoConfig(adv_estimator="remax", remax_measures=measures))


@pytest.mark.parametrize("config_kind", ["dict", "dataclass", "omegaconf"])
def test_remax_omitted_delta_defaults_to_zero(config_kind):
    mask, sampled, batch, _ = _metric_remax_inputs()
    values = {
        "adv_estimator": "remax",
        "remax_measures": {"char": {}, "word": {"beta": 2.0}, "char_del": {}},
    }
    if config_kind == "dataclass":
        config = AlgoConfig(**values)
    elif config_kind == "omegaconf":
        from omegaconf import OmegaConf

        config = OmegaConf.create(values)
    else:
        config = values
    assert get_remax_reward_keys(config) == ["char", "word", "char_del"]
    actual = compute_remax_metric_accumulation(config, sampled, batch, mask)
    explicit = AlgoConfig(
        adv_estimator="remax",
        remax_measures={"char": {"delta": 0}, "word": {"beta": 2.0, "delta": 0}, "char_del": {"delta": 0}},
    )
    expected = compute_remax_metric_accumulation(explicit, sampled, batch, mask)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference)
    torch.testing.assert_close(actual[0], torch.tensor([0.125, 0.875, 1.5, 0.5, 0.0]))


@pytest.mark.parametrize("legacy", ["gdpo_reward_keys", "gdpo_reward_weights"])
@pytest.mark.parametrize("new_mapping", [None, {"char": {"delta": 0}}])
def test_remax_legacy_keys_are_gdpo_only(legacy, new_mapping):
    config = AlgoConfig(adv_estimator="remax", remax_measures=new_mapping, **{legacy: ["char"]})
    with pytest.raises(ValueError, match="GDPO-only"):
        get_remax_reward_keys(config)
    assert get_remax_reward_keys(AlgoConfig(adv_estimator="gdpo", **{legacy: ["char"]})) == []


@pytest.mark.parametrize("invalid", ["missing_sample", "missing_base", "sample_shape", "base_shape", "nan", "inf"])
def test_remax_rejects_invalid_metric_data_even_after_stopping(invalid):
    mask, sampled, batch, config = _metric_remax_inputs()
    if invalid == "missing_sample":
        del sampled["char_del"]
    elif invalid == "missing_base":
        del batch["reward_baselines_char_del"]
    elif invalid == "sample_shape":
        sampled["char_del"] = np.ones((5, 1))
    elif invalid == "base_shape":
        batch["reward_baselines_char_del"] = torch.zeros(1)
    elif invalid == "nan":
        sampled["char_del"][0] = float("nan")
    else:
        batch["reward_baselines_char_del"][0] = float("inf")
    with pytest.raises(ValueError, match="ReMax|Reward component"):
        compute_remax_metric_accumulation(config, sampled, batch, mask)


def test_gdpo_missing_dimensions_are_excluded_from_group_and_batch_normalization():
    mask = torch.ones((6, 2))
    config = AlgoConfig(
        adv_estimator="gdpo", gdpo_reward_keys=["char", "word", "absent"], gdpo_reward_weights=[1.0, 3.0, 100.0]
    )
    advantages, returns = compute_gdpo_outcome_advantage(
        torch.full_like(mask, 100.0),
        mask,
        np.zeros(6),
        epsilon=0,
        config=config,
        non_tensor_batch={
            "char": np.array([0.0, 1.0, None, None, None, None], dtype=object),
            "word": np.array([None, None, 2.0, 4.0, None, None], dtype=object),
        },
    )
    raw = torch.tensor([-1.0, 1.0, -3.0, 3.0])[:, None].expand(4, 2) / 2**0.5
    expected = torch.zeros_like(mask)
    expected[:4] = raw / raw.flatten().std()
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_gdpo_complete_components_preserve_decoupled_weighted_normalization():
    mask = torch.tensor([[1.0, 1.0, 0.0]] * 4)
    index = np.array([0, 0, 1, 1])
    char = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.8, 0.0], [0.0, 0.4, 0.0], [0.0, 0.9, 0.0]])
    word = torch.tensor([[0.0, 0.7, 0.0], [0.0, 0.3, 0.0], [0.0, 0.1, 0.0], [0.0, 0.5, 0.0]])
    char_adv, _ = compute_grpo_outcome_advantage(char, mask, index)
    word_adv, _ = compute_grpo_outcome_advantage(word, mask, index)
    expected = verl.trainer.ppo.core_algos.verl_F.masked_whiten(char_adv + 3 * word_adv, mask) * mask
    actual, _ = compute_gdpo_outcome_advantage(
        char + word,
        mask,
        index,
        config=AlgoConfig(adv_estimator="gdpo", gdpo_reward_keys=["char", "word"], gdpo_reward_weights=[1, 3]),
        non_tensor_batch={"char": char.sum(-1).numpy(), "word": word.sum(-1).numpy()},
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("missing", [{}, {"char": [None, None]}])
def test_gdpo_all_components_missing_never_uses_scalar_fallback(missing):
    mask = torch.ones((2, 2))
    advantages, returns = compute_gdpo_outcome_advantage(
        torch.full_like(mask, 100.0),
        mask,
        np.array([0, 0]),
        config=AlgoConfig(adv_estimator="gdpo", gdpo_reward_keys=["char"]),
        non_tensor_batch=missing,
    )
    assert torch.equal(advantages, torch.zeros_like(mask))
    assert torch.equal(returns, torch.zeros_like(mask))


def test_gdpo_single_present_token_and_empty_response_are_finite():
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    advantages, _ = compute_gdpo_outcome_advantage(
        torch.ones_like(mask),
        mask,
        np.array([0, 0, 0]),
        config=AlgoConfig(adv_estimator="gdpo", gdpo_reward_keys=["char"]),
        non_tensor_batch={"char": [0.5, None, 1.0]},
    )
    torch.testing.assert_close(advantages, torch.tensor([[0.5 / (1 + 1e-6), 0.0], [0.0, 0.0], [0.0, 0.0]]))


@pytest.mark.parametrize("values", [[float("nan"), None], [None, float("inf")], [[1.0], [None]]])
def test_gdpo_rejects_invalid_present_components(values):
    with pytest.raises(ValueError, match="Reward component"):
        compute_gdpo_outcome_advantage(
            torch.zeros((2, 2)),
            torch.ones((2, 2)),
            np.array([0, 0]),
            config=AlgoConfig(adv_estimator="gdpo", gdpo_reward_keys=["char"]),
            non_tensor_batch={"char": values},
        )


def test_remax_skips_metrics_absent_on_both_sides():
    mask, sampled, batch, config = _metric_remax_inputs()
    del sampled["char"]
    del batch["reward_baselines_char"]
    score, rewards = compute_remax_metric_accumulation(config, sampled, batch, mask)
    torch.testing.assert_close(score, torch.tensor([-0.375, 0.375, 0.375, 0.25, 0.0]))
    torch.testing.assert_close(rewards, torch.tensor([0.125, 0.875, 0.875, 1.25, 0.0]))
    score, rewards = compute_remax_metric_accumulation(config, {}, {}, mask)
    assert torch.equal(score, torch.zeros(5))
    assert torch.equal(rewards, torch.zeros(5))


def test_remax_mixed_source_availability_is_row_aligned():
    config = AlgoConfig(
        adv_estimator="remax",
        remax_measures={
            "char": {"delta": 0.25},
            "word": {"delta": 0.25},
            "absent": {"delta": 0.25},
        },
    )
    sampled = {"char": [None, 0.875, 0.5], "word": [0.875, None, 0.625]}
    baselines = {"char": [None, 0.5, 0.5], "word": [0.5, None, 0.5]}
    batch = {}
    capture_remax_metric_baselines(config, baselines, batch, 3)
    mask = torch.ones((3, 2))
    score, rewards = compute_remax_metric_accumulation(config, sampled, batch, mask)
    torch.testing.assert_close(score, torch.tensor([0.375, 0.375, 0.125]))
    torch.testing.assert_close(rewards, torch.tensor([0.875, 0.875, 1.125]))
    assert batch["reward_baselines_absent_present"].tolist() == [False] * 3
    sampled["char"][0] = 0.5
    with pytest.raises(ValueError, match="only one side"):
        compute_remax_metric_accumulation(config, sampled, batch, mask)


@pytest.mark.parametrize("missing", ["sampled", "baseline"])
def test_remax_metrics_never_fall_back_to_scalar_rewards(missing):
    mask, sampled, batch, config = _metric_remax_inputs()
    with pytest.raises(ValueError, match="sampled metric data"):
        compute_remax_outcome_advantage(
            torch.ones_like(mask),
            torch.zeros(5),
            mask,
            config=config,
            non_tensor_batch=None if missing == "sampled" else sampled,
            batch=None if missing == "baseline" else batch,
        )


@pytest.mark.parametrize(
    ("norm_mode", "expected_divisor"),
    [("l2", 0.18**0.5), ("rms", 0.06**0.5)],
)
def test_compute_remax_normalizes_single_reward_by_configured_mode(norm_mode, expected_divisor):
    token_level_rewards = torch.tensor([[0.0, 0.8], [0.0, 0.2], [0.0, 0.5]])
    response_mask = torch.ones((3, 2))

    advantages, _ = compute_remax_outcome_advantage(
        token_level_rewards=token_level_rewards,
        reward_baselines=torch.full((3,), 0.5),
        response_mask=response_mask,
        index=np.array([0, 0, 0]),
        config=AlgoConfig(
            adv_estimator="remax",
            norm_adv_in_remax=norm_mode,
        ),
    )

    expected_advantages = torch.tensor([[0.3, 0.3], [-0.3, -0.3], [0.0, 0.0]]) / expected_divisor
    assert torch.allclose(advantages, expected_advantages)


def test_compute_remax_rejects_unknown_normalization_mode():
    with pytest.raises(ValueError, match="Unsupported ReMax advantage normalization mode: std"):
        compute_remax_outcome_advantage(
            token_level_rewards=torch.zeros((1, 1)),
            reward_baselines=torch.zeros(1),
            response_mask=torch.ones((1, 1)),
            config=AlgoConfig(adv_estimator="remax", norm_adv_in_remax="std"),
        )


def test_compute_remax_outcome_advantage_binary_adv():
    token_level_rewards = torch.tensor(
        [
            [0.0, 0.8, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=torch.float,
    )
    reward_baselines = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float)
    response_mask = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float,
    )

    advantages, returns = compute_remax_outcome_advantage(
        token_level_rewards=token_level_rewards,
        reward_baselines=reward_baselines,
        response_mask=response_mask,
        config=AlgoConfig(adv_estimator="remax", binary_adv=True),
    )

    expected_advantages = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float,
    )
    expected_returns = torch.tensor(
        [
            [0.8, 0.8, 0.0],
            [0.2, 0.2, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=torch.float,
    )

    assert torch.equal(advantages, expected_advantages)
    assert torch.equal(returns, expected_returns)


def test_compute_remax_outcome_advantage_binary_adv_with_adv_scale():
    token_level_rewards = torch.tensor(
        [
            [0.0, 0.8, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=torch.float,
    )
    reward_baselines = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float)
    response_mask = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float,
    )

    advantages, returns = compute_remax_outcome_advantage(
        token_level_rewards=token_level_rewards,
        reward_baselines=reward_baselines,
        response_mask=response_mask,
        config=AlgoConfig(adv_estimator="remax", binary_adv=True, adv_scale=2.0),
    )

    expected_advantages = torch.tensor(
        [
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float,
    )
    expected_returns = torch.tensor(
        [
            [0.8, 0.8, 0.0],
            [0.2, 0.2, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=torch.float,
    )

    assert torch.equal(advantages, expected_advantages)
    assert torch.equal(returns, expected_returns)


def test_compute_remax_outcome_advantage_datasource_adv_scale():
    token_level_rewards = torch.tensor(
        [
            [0.0, 0.8, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.8, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.8, 0.0],
        ],
        dtype=torch.float,
    )
    reward_baselines = torch.full((5,), 0.5, dtype=torch.float)
    response_mask = torch.tensor([[0.0, 1.0, 0.0]] * 5, dtype=torch.float)
    data_sources = np.array(["mix_cv15_all", "enus_digits_chunk_100", "openml", "openml", "unknown"])

    advantages, _ = compute_remax_outcome_advantage(
        token_level_rewards=token_level_rewards,
        reward_baselines=reward_baselines,
        response_mask=response_mask,
        data_sources=data_sources,
        config=AlgoConfig(
            adv_estimator="remax",
            adv_scale={
                "mix_cv15_all": 0.5,
                "enus_digits_chunk_100": 2.0,
                "openml": 3.0,
                "default": 4.0,
            },
        ),
    )

    expected_advantages = torch.tensor(
        [
            [0.4, 0.15, 0.0],
            [0.4, -0.6, 0.0],
            [2.4, 0.9, 0.0],
            [0.6, -0.9, 0.0],
            [3.2, 1.2, 0.0],
        ],
        dtype=torch.float,
    )
    assert torch.allclose(advantages, expected_advantages)


if __name__ == "__main__":
    unittest.main()

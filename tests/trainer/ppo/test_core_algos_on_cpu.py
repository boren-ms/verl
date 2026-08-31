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

import numpy as np
import pytest
import torch

import verl.trainer.ppo.core_algos
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import (
    compute_topk_log_probs,
    compute_gae_advantage_return,
    compute_gdpo_outcome_advantage,
    compute_remax_outcome_advantage,
    get_adv_estimator_fn,
    kl_penalty,
    register_adv_est,
    topk_distill_kl,
)


def mock_test_fn():
    pass


def test_kl_penalty_uses_fp32_for_bf16_inputs():
    logprob = torch.tensor([-1.0, -2.0], dtype=torch.bfloat16)
    ref_logprob = torch.tensor([-1.5, -1.5], dtype=torch.bfloat16)

    result = kl_penalty(logprob, ref_logprob, "low_var_kl")

    assert result.dtype == torch.float32


def test_k3_straight_through_uses_k3_value_and_k2_gradient():
    logprob = torch.tensor([-1.0, -2.0], requires_grad=True)
    ref_logprob = torch.tensor([-1.5, -1.5])

    result = kl_penalty(logprob, ref_logprob, "k3+")
    expected_value = kl_penalty(logprob.detach(), ref_logprob, "k3")
    result.sum().backward()

    torch.testing.assert_close(result.detach(), expected_value)
    torch.testing.assert_close(logprob.grad, logprob.detach() - ref_logprob)


def test_topk_distill_kl_matches_grouped_distribution_kl():
    student_logits = torch.tensor([[[1.0, 0.0, -1.0, 2.0]]], requires_grad=True)
    teacher_logits = torch.tensor([[[2.0, 1.0, 0.0, -1.0]]], requires_grad=True)
    topk_indices = torch.tensor([[[0, 1]]])
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    student_topk = torch.gather(student_log_probs, -1, topk_indices)
    teacher_topk = torch.gather(teacher_log_probs, -1, topk_indices)
    student_tail = torch.logsumexp(student_log_probs[..., 2:], dim=-1)
    teacher_tail = torch.logsumexp(teacher_log_probs[..., 2:], dim=-1)

    result = topk_distill_kl(student_topk, student_tail, teacher_topk, teacher_tail)
    teacher_grouped = torch.cat([teacher_topk.exp(), teacher_tail.exp().unsqueeze(-1)], dim=-1)
    student_grouped = torch.cat([student_topk.exp(), student_tail.exp().unsqueeze(-1)], dim=-1)
    expected = (teacher_grouped * (teacher_grouped.log() - student_grouped.log())).sum(dim=-1)

    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert student_logits.grad is not None
    assert teacher_logits.grad is None


def test_topk_distill_kl_applies_temperature_squared_scaling():
    student_topk = torch.log(torch.tensor([[[0.2, 0.3]]]))
    teacher_topk = torch.log(torch.tensor([[[0.4, 0.1]]]))
    student_tail = torch.log(torch.tensor([[0.5]]))
    teacher_tail = torch.log(torch.tensor([[0.5]]))

    unscaled = topk_distill_kl(student_topk, student_tail, teacher_topk, teacher_tail)
    scaled = topk_distill_kl(student_topk, student_tail, teacher_topk, teacher_tail, temperature=2.0)

    torch.testing.assert_close(scaled, unscaled * 4)


def test_compute_topk_log_probs_uses_full_vocab_normalization_and_requested_indices():
    logits = torch.tensor([[[1.0, 4.0, 2.0, 3.0]]], requires_grad=True)

    topk_log_probs, tail_log_prob, topk_indices = compute_topk_log_probs(
        logits, topk=2, temperature=2.0, vocab_chunk_size=2
    )
    expected_full = torch.log_softmax(logits.float() / 2.0, dim=-1)

    torch.testing.assert_close(topk_indices, torch.tensor([[[1, 3]]]))
    torch.testing.assert_close(topk_log_probs, torch.gather(expected_full, -1, topk_indices))
    expected_tail = torch.logsumexp(expected_full[..., [0, 2]], dim=-1)
    torch.testing.assert_close(tail_log_prob, expected_tail)

    requested_indices = torch.tensor([[[0, 2]]])
    selected_log_probs, selected_tail, returned_indices = compute_topk_log_probs(
        logits, topk=2, temperature=2.0, topk_indices=requested_indices, vocab_chunk_size=2
    )
    torch.testing.assert_close(selected_log_probs, torch.gather(expected_full, -1, requested_indices))
    torch.testing.assert_close(selected_tail, torch.logsumexp(expected_full[..., [1, 3]], dim=-1))
    torch.testing.assert_close(returned_indices, requested_indices)


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
def test_compute_remax_normalizes_each_reward_dimension_before_weighting(norm_mode, expected_scale):
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
            gdpo_reward_keys=["char", "lang"],
            gdpo_reward_weights=[1.0, 2.0],
        ),
        non_tensor_batch=non_tensor_batch,
        batch=batch,
    )

    expected_advantages = torch.tensor(
        [[0.6, 0.6], [2.8, 2.8], [24 / 13, 24 / 13], [23 / 13, 23 / 13]]
    ) * expected_scale
    expected_returns = torch.tensor([[3.0, 3.0], [24.0, 24.0], [24.0, 24.0], [15.0, 15.0]])
    assert torch.allclose(advantages, expected_advantages)
    assert torch.allclose(returns, expected_returns)


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

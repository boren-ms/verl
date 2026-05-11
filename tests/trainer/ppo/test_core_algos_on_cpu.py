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
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import verl.trainer.ppo.core_algos
from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    compute_self_distillation_loss,
    compute_token_level_rollout_is_weights,
    get_adv_estimator_fn,
    register_adv_est,
)


def mock_test_fn():
    pass


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


def test_compute_self_distillation_loss_token_masking():
    cfg = SimpleNamespace(
        full_logit_distillation=False,
        alpha=1.0,
        distillation_topk=None,
        distillation_add_tail=True,
        is_clip=None,
    )
    student_log_probs = torch.tensor([[-0.3, -0.5, -0.7], [-0.4, -0.6, -0.8]])
    teacher_log_probs = torch.tensor([[-0.2, -0.8, -0.6], [-0.5, -0.5, -0.7]])
    response_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    self_distillation_mask = torch.tensor([1.0, 0.0])

    loss, metrics = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=cfg,
        self_distillation_mask=self_distillation_mask,
    )

    per_token = (student_log_probs - teacher_log_probs).detach() * student_log_probs
    expected_mask = response_mask * self_distillation_mask.unsqueeze(1)
    expected = (per_token * expected_mask).sum() / expected_mask.sum()
    assert torch.allclose(loss, expected)
    assert metrics["self_distillation/variant_code"] == 3.0
    assert metrics["self_distillation/mask_fraction"] == expected_mask.mean().item()


def test_compute_self_distillation_loss_full_logits_zero_when_equal():
    cfg = SimpleNamespace(
        full_logit_distillation=True,
        alpha=0.5,
        distillation_topk=None,
        distillation_add_tail=True,
        is_clip=None,
    )
    logits = torch.randn(2, 3, 5)
    log_probs = F.log_softmax(logits, dim=-1)
    response_mask = torch.ones(2, 3)
    token_log_probs = torch.zeros(2, 3)

    loss, metrics = compute_self_distillation_loss(
        student_log_probs=token_log_probs,
        teacher_log_probs=token_log_probs,
        response_mask=response_mask,
        self_distillation_config=cfg,
        student_all_log_probs=log_probs,
        teacher_all_log_probs=log_probs,
    )

    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)
    assert metrics["self_distillation/variant_code"] == 0.0
    assert metrics["self_distillation/kl_type_code"] == 2.0


def test_compute_self_distillation_loss_topk_alpha_half():
    cfg = SimpleNamespace(
        full_logit_distillation=True,
        alpha=0.5,
        distillation_topk=2,
        distillation_add_tail=True,
        is_clip=None,
    )
    student_topk = torch.log(torch.tensor([[[0.6, 0.2], [0.5, 0.3]]]))
    teacher_topk = torch.log(torch.tensor([[[0.4, 0.4], [0.2, 0.6]]]))
    response_mask = torch.tensor([[1.0, 0.0]])
    token_log_probs = torch.zeros(1, 2)

    loss, metrics = compute_self_distillation_loss(
        student_log_probs=token_log_probs,
        teacher_log_probs=token_log_probs,
        response_mask=response_mask,
        self_distillation_config=cfg,
        student_topk_log_probs=student_topk,
        teacher_topk_log_probs=teacher_topk,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert metrics["self_distillation/variant_code"] == 1.0
    assert metrics["self_distillation/kl_type_code"] == 2.0


def test_compute_token_level_rollout_is_weights_clips_by_threshold():
    old_log_probs = torch.log(torch.tensor([[4.0, 1.0, 0.5]]))
    rollout_log_probs = torch.log(torch.tensor([[1.0, 1.0, 1.0]]))
    response_mask = torch.tensor([[1.0, 1.0, 0.0]])

    weights, metrics = compute_token_level_rollout_is_weights(
        old_log_probs=old_log_probs,
        rollout_log_probs=rollout_log_probs,
        response_mask=response_mask,
        threshold=2.0,
    )

    assert torch.allclose(weights, torch.tensor([[2.0, 1.0, 0.0]]))
    assert metrics["rollout_corr/is_weight_mean"] == pytest.approx(1.5)
    assert metrics["rollout_corr/is_clip_fraction"] == pytest.approx(0.5)


if __name__ == "__main__":
    unittest.main()

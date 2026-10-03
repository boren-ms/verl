from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from recipe.dapo.dapo_ray_trainer import _attach_remax_disagreement_mask
from verl import DataProto
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss_vanilla, deduplicate_rollout_responses, kl_penalty
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _validate_remax_advantage_mask,
    compute_advantage,
    compute_response_mask,
)


def _rollouts(responses):
    responses = torch.tensor(responses, dtype=torch.long)
    mask = (responses != 0).long()
    return DataProto.from_dict(
        tensors={
            "responses": responses,
            "attention_mask": torch.cat([torch.ones((len(responses), 1), dtype=torch.long), mask], dim=1),
        }
    )


@pytest.mark.parametrize("mode", ["edit_boundary", "first_diff"])
def test_greedy_masks_follow_interleaved_rows_and_batch_operations(mode):
    baseline = _rollouts([[1, 2, 9, 0], [3, 4, 9, 0]])
    sampled = _rollouts([[1, 5, 9, 0], [1, 9, 0, 0]] * 4 + [[3, 4, 9, 0], [3, 6, 9, 0]] * 4)
    sampled.batch["row_id"] = torch.arange(16)
    _attach_remax_disagreement_mask(sampled, baseline, 8, mode=mode)
    substitution = [0, 1, 1, 0] if mode == "first_diff" else [0, 1, 0, 0]
    expected = torch.tensor([substitution, [0, 1, 0, 0]] * 4 + [[0, 0, 0, 0], substitution] * 4)
    assert torch.equal(sampled.batch["remax_advantage_mask"], expected)

    prompts = DataProto.from_dict(
        tensors={"prompts": torch.ones((2, 1), dtype=torch.long)},
        non_tensors={"uid": np.array(["a", "b"], dtype=object)},
    )
    sampled = prompts.repeat(8, interleave=True).union(sampled)
    sampled.batch["response_mask"] = compute_response_mask(sampled)
    retained = sampled[torch.tensor([0, 1, 1, 8, 9])]
    merged = DataProto.concat([retained[:2], retained[2:]])
    merged.reorder(torch.tensor([4, 1, 2, 0, 3]))
    merged, _, _, _ = deduplicate_rollout_responses(merged, dp_size=1)
    assert len(merged) == 4
    assert torch.equal(merged.batch["remax_advantage_mask"], expected[merged.batch["row_id"]])


def test_advantage_mask_always_includes_deletion_anchor():
    baseline = _rollouts([[1, 2, 9]])
    sampled = _rollouts([[1, 9, 0]])
    _attach_remax_disagreement_mask(sampled, baseline, 1)
    assert sampled.batch["remax_advantage_mask"].tolist() == [[0, 1, 0]]


@pytest.mark.parametrize("rollout_n", [0, 1, 3])
def test_greedy_masks_reject_incorrect_prompt_mapping(rollout_n):
    with pytest.raises(ValueError, match="interleaved"):
        _attach_remax_disagreement_mask(_rollouts([[1, 9], [2, 9]]), _rollouts([[1, 9]]), rollout_n)


def _advantage_batch():
    return DataProto.from_dict(
        tensors={
            "prompts": torch.ones((2, 1), dtype=torch.long),
            "responses": torch.tensor([[1, 2, 9, 0], [1, 3, 9, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]] * 2),
            "token_level_scores": torch.tensor([[0.0, 0.0, 0.8, 0.0], [0.0, 0.0, 0.2, 0.0]]),
            "token_level_rewards": torch.tensor([[0.0, 0.0, 0.8, 0.0], [0.0, 0.0, 0.2, 0.0]]),
            "reward_baselines": torch.tensor([0.5, 0.5]),
            "reward_baselines_char": torch.tensor([0.5, 0.5]),
            "reward_baselines_lang": torch.tensor([1.0, 1.0]),
            "remax_advantage_mask": torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0]]),
        },
        non_tensors={
            "uid": np.array(["a", "a"], dtype=object),
            "data_source": np.array(["openml", "openml"], dtype=object),
            "char": np.array([0.8, 0.2]),
            "lang": np.array([1.0, 1.0]),
        },
    )


@pytest.mark.parametrize("norm", [None, "l2", "rms"])
@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("multi_reward", [False, True])
@pytest.mark.parametrize("mode", ["edit_boundary", "first_diff"])
def test_masks_final_advantages_only(norm, binary, multi_reward, mode):
    config = AlgoConfig(
        adv_estimator="remax",
        norm_adv_in_remax=norm,
        binary_adv=binary,
        adv_scale={"openml": 2.0},
        gdpo_reward_keys=["char", "lang"] if multi_reward else None,
        gdpo_reward_weights=[0.2, 0.1] if multi_reward else None,
    )
    original = compute_advantage(_advantage_batch(), "remax", config=config)
    masked_config = replace(config, remax_advantage_mask=mode)
    masked = compute_advantage(_advantage_batch(), "remax", config=masked_config)
    assert torch.equal(
        masked.batch["advantages"], original.batch["advantages"] * masked.batch["remax_advantage_mask"]
    )
    for key in ("returns", "response_mask", "token_level_scores", "token_level_rewards", "reward_baselines"):
        assert torch.equal(masked.batch[key], original.batch[key])
    assert masked.batch["advantages"][0, 1] > 0
    assert masked.batch["advantages"][1, 1] < 0


@pytest.mark.parametrize("all_zero", [False, True])
def test_masked_policy_gradient_preserves_regularization_and_denominator(all_zero):
    batch = _advantage_batch()
    if all_zero:
        batch.batch["remax_advantage_mask"].zero_()
    batch = compute_advantage(
        batch, "remax", config=AlgoConfig(adv_estimator="remax", remax_advantage_mask="edit_boundary")
    )
    valid = batch.batch["response_mask"]
    selected = batch.batch["remax_advantage_mask"].bool()
    log_probs = torch.full((2, 4), -0.9, requires_grad=True)
    old = torch.full_like(log_probs, -1.0)
    config = SimpleNamespace(clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.28, tis_imp_ratio_cap=5)
    config.get = lambda name, default: getattr(config, name, default)
    loss, *_ = compute_policy_loss_vanilla(
        old, log_probs, batch.batch["advantages"], valid, config=config, rollout_log_probs=old
    )
    expected = (-batch.batch["advantages"] * torch.exp(log_probs - old) * valid).sum() / valid.sum()
    torch.testing.assert_close(loss, expected)
    gradient = torch.autograd.grad(loss, log_probs, retain_graph=True)[0]
    assert torch.equal(gradient[~selected], torch.zeros_like(gradient[~selected]))
    if all_zero:
        assert loss.item() == 0
    else:
        assert torch.all(gradient[selected] != 0)

    kl = agg_loss(kl_penalty(log_probs, old, "low_var_kl"), valid, "token-mean")
    entropy = agg_loss(-log_probs, valid, "token-mean")
    kl_gradient = torch.autograd.grad(kl, log_probs, retain_graph=True)[0]
    entropy_gradient = torch.autograd.grad(entropy, log_probs)[0]
    matched = valid.bool() & ~selected
    assert torch.all(kl_gradient[matched] != 0)
    assert torch.all(entropy_gradient[matched] != 0)
    for value in (loss, kl, entropy):
        assert torch.isfinite(value)


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"adv_estimator": "grpo"}, "adv_estimator=remax"),
        ({"use_kl_in_reward": True}, "actor-side KL"),
        ({"remax_advantage_mask": "unknown"}, "remax_advantage_mask"),
        ({"remax_advantage_mask": ""}, "remax_advantage_mask"),
        ({"remax_advantage_mask": True}, "remax_advantage_mask"),
        ({"remax_advantage_mask": False}, "remax_advantage_mask"),
    ],
)
def test_rejects_incompatible_configuration(overrides, error):
    config = AlgoConfig(**{"adv_estimator": "remax", "remax_advantage_mask": "edit_boundary", **overrides})
    with pytest.raises(ValueError, match=error):
        _validate_remax_advantage_mask(config, config.adv_estimator)
    with pytest.raises(ValueError, match=error):
        compute_advantage(_advantage_batch(), config.adv_estimator, config=config)


def test_null_advantage_mask_disables_masking():
    batch = _advantage_batch()
    del batch.batch["remax_advantage_mask"]
    unmasked = compute_advantage(batch, "remax", config=AlgoConfig(adv_estimator="remax", remax_advantage_mask=None))
    expected = compute_advantage(_advantage_batch(), "remax", config=AlgoConfig(adv_estimator="remax"))
    assert torch.equal(unmasked.batch["advantages"], expected.batch["advantages"])


def test_rejects_sequence_level_policy_loss_before_trainer_startup():
    config = SimpleNamespace(
        algorithm=AlgoConfig(adv_estimator="remax", remax_advantage_mask="edit_boundary"),
        actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(policy_loss={"loss_mode": "gspo"})),
    )
    with pytest.raises(ValueError, match="token-level vanilla"):
        RayPPOTrainer(config, tokenizer=None, role_worker_mapping={}, resource_pool_manager=None)


@pytest.mark.parametrize("invalid", ["missing", "shape", "nan", "fractional", "padding"])
def test_rejects_invalid_advantage_masks(invalid):
    batch = _advantage_batch()
    if invalid == "missing":
        del batch.batch["remax_advantage_mask"]
    elif invalid == "shape":
        batch.batch["remax_advantage_mask"] = torch.ones((2, 3))
    elif invalid in ("nan", "fractional"):
        batch.batch["remax_advantage_mask"] = batch.batch["remax_advantage_mask"].float()
        batch.batch["remax_advantage_mask"][0, 0] = float("nan") if invalid == "nan" else 0.5
    elif invalid == "padding":
        batch.batch["remax_advantage_mask"][0, -1] = 1
    with pytest.raises(ValueError, match="remax_advantage_mask"):
        compute_advantage(
            batch, "remax", config=AlgoConfig(adv_estimator="remax", remax_advantage_mask="edit_boundary")
        )

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

from collections import defaultdict

import numpy as np

from verl.trainer.ppo.core_algos import compute_remax_metric_accumulation, get_remax_reward_keys


def compute_remax_advantage_metric(
    sampled_rewards: np.ndarray,
    reward_baselines: np.ndarray,
    *,
    config=None,
    non_tensor_batch=None,
    batch=None,
    response_mask=None,
) -> np.ndarray:
    """Compute raw ReMax signal before normalization, binarization, and scaling."""
    if get_remax_reward_keys(config, "remax"):
        if response_mask is None:
            raise ValueError("ReMax metric filtering requires a response mask")
        score, _ = compute_remax_metric_accumulation(config, non_tensor_batch, batch, response_mask)
        return score.detach().cpu().numpy()
    sampled_rewards = np.asarray(sampled_rewards)
    reward_baselines = np.asarray(reward_baselines)
    if sampled_rewards.shape != reward_baselines.shape:
        raise ValueError(
            "Sampled rewards and ReMax reward baselines must have matching shapes, "
            f"got {sampled_rewards.shape} and {reward_baselines.shape}"
        )
    return sampled_rewards - reward_baselines


def select_prompt_uids_by_metric(
    prompt_uids: np.ndarray,
    metric_values: np.ndarray,
    mode: str = "variance",
    atol: float = 0.0,
) -> list:
    """Select prompt groups using their per-trajectory scalar metric values."""
    if mode not in ("variance", "nonzero"):
        raise ValueError(f"Unsupported filter group mode: {mode}")
    if atol < 0:
        raise ValueError(f"Filter group atol must be non-negative, got {atol}")

    prompt_uid2metric_vals = defaultdict(list)
    for uid, metric_value in zip(prompt_uids, metric_values, strict=True):
        if metric_value is None:
            continue
        value = np.asarray(metric_value)
        if value.size != 1:
            raise ValueError(f"Filter group metric must be scalar, got shape {value.shape}")
        scalar = float(value.item())
        if not np.isfinite(scalar):
            raise ValueError("Filter group metric must be finite when present")
        prompt_uid2metric_vals[uid].append(scalar)

    kept_prompt_uids = []
    for uid, values in prompt_uid2metric_vals.items():
        if mode == "variance":
            keep = len(values) == 1 or np.std(values) > atol
        else:
            keep = np.any(np.abs(values) > atol)
        if keep:
            kept_prompt_uids.append(uid)
    return kept_prompt_uids

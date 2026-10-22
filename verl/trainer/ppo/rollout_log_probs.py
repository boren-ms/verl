"""Align teacher-forced rollout scores with the actor's probability convention."""

import math

import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


def align_gt_rollout_log_probs(data: DataProto, actor_rollout_wg) -> None:
    """Replace only GT scores with actor scores at the configured rollout temperature.

    vLLM's ``processed_logprobs`` applies to generated tokens. Its GT
    ``prompt_logprobs`` remain unscaled, so their selected-token values cannot
    be converted to another temperature without the full vocabulary. Score
    only these teacher-forced rows with the actor instead. This does not turn
    GT examples into samples from the behavior policy.
    """
    if "gt_response_mask" not in data.batch or "rollout_log_probs" not in data.batch:
        return
    gt_mask = data.batch["gt_response_mask"]
    if gt_mask.dtype != torch.bool or gt_mask.shape != (len(data),):
        raise ValueError("gt_response_mask must be a boolean vector with one entry per trajectory")
    indices = gt_mask.nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        return

    temperature = data.meta_info.get("temperature")
    if temperature is None or not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("GT log-prob alignment requires a finite positive actor temperature")

    gt_data, pad_size = pad_dataproto_to_divisor(data[indices], actor_rollout_wg.world_size)
    # DataProto row selection shares metadata; workers add their scoring options.
    gt_data.meta_info = gt_data.meta_info.copy()
    scored = unpad_dataproto(actor_rollout_wg.compute_log_prob(gt_data), pad_size)
    scored_temperature = scored.meta_info.get("temperature")
    if scored_temperature is None or not math.isclose(float(scored_temperature), float(temperature)):
        raise ValueError("GT log-prob scoring must use the same temperature as the actor update")

    rollout_log_probs = data.batch["rollout_log_probs"]
    gt_log_probs = scored.batch["old_log_probs"]
    if gt_log_probs.shape != rollout_log_probs[indices].shape:
        raise ValueError("GT actor log-prob scores must match the selected rollout rows")
    rollout_log_probs[indices] = gt_log_probs.to(device=rollout_log_probs.device, dtype=rollout_log_probs.dtype)

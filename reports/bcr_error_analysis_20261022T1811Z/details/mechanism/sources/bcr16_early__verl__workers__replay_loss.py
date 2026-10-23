# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Loss normalization for variable-size replay batches padded across FSDP ranks."""

import torch
import torch.distributed as dist


class ReplayLossNormalizer:
    """Turn microbatch token means into the real global minibatch objective.

    All ranks must construct one normalizer for each optimizer minibatch, including
    ranks containing only copied padding rows. FSDP averages their gradients, so
    each local loss is multiplied by the number of ranks. Forward/backward still
    runs on padding rows to keep FSDP collectives in the same order.

    This contract is for FSDP with no sequence parallelism. ``valid_rows`` marks
    real trajectories, independently of their token-level response mask.
    """

    MODES = (
        "token-mean",
        "seq-mean-token-sum",
        "seq-mean-token-mean",
        "seq-mean-token-sum-norm",
    )

    def __init__(self, response_mask, valid_rows, loss_agg_mode, device):
        if loss_agg_mode not in self.MODES:
            raise ValueError(f"Unsupported replay loss aggregation mode: {loss_agg_mode}")
        self._validate(response_mask, valid_rows)
        self.loss_agg_mode = loss_agg_mode
        self.response_length = response_mask.shape[-1]
        counts = torch.stack(
            [
                (response_mask * valid_rows[:, None]).sum(dtype=torch.float64),
                valid_rows.sum(dtype=torch.float64),
            ]
        ).to(device)
        self.world_size = 1
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        token_count, row_count = counts.cpu().tolist()
        if loss_agg_mode == "token-mean":
            self.denominator = token_count
        elif loss_agg_mode == "seq-mean-token-sum-norm":
            # Preserve this aggregation's sum, rather than mean, over sequences.
            self.denominator = self.response_length
        else:
            self.denominator = row_count

    @staticmethod
    def _validate(response_mask, valid_rows):
        if response_mask.ndim != 2 or valid_rows.shape != response_mask.shape[:1]:
            raise ValueError("Replay valid-row mask must have one value per response row")
        if valid_rows.dtype != torch.bool:
            raise ValueError("Replay valid-row mask must be boolean")

    def weights_and_scale(self, response_mask, valid_rows):
        """Return token weights and the scale for a token-mean microbatch loss.

        Row-normalizing token weights implements sequence-mean-token-mean without
        dividing padding rows by zero. Other modes recover token sums from the
        microbatch mean and apply their corresponding global denominator.
        """
        self._validate(response_mask, valid_rows)
        weights = response_mask.float() * valid_rows[:, None]
        if self.loss_agg_mode == "seq-mean-token-mean":
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1)
        if self.denominator == 0:
            return weights, 0.0
        scale = weights.sum().item() * self.world_size / self.denominator
        return weights, scale

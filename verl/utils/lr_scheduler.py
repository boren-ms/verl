# Copyright 2026 Microsoft Corporation
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

import math
from bisect import bisect_right
from collections.abc import Callable, Sequence
from itertools import accumulate
from numbers import Real
from typing import Optional

from torch.optim import Optimizer


def validate_lr_cycle_config(cycle_steps: Optional[Sequence[int]], cycle_lrs: Optional[Sequence[float]]) -> None:
    """Validate optional restart intervals and absolute per-cycle peak learning rates."""
    if cycle_steps is None:
        if cycle_lrs is not None:
            raise ValueError("cycle_lrs requires cycle_steps.")
        return
    if (
        not isinstance(cycle_steps, Sequence)
        or isinstance(cycle_steps, (str, bytes))
        or not cycle_steps
        or any(isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0 for steps in cycle_steps)
    ):
        raise ValueError("cycle_steps must be a non-empty sequence of positive integers.")
    if cycle_lrs is not None:
        if not isinstance(cycle_lrs, Sequence) or isinstance(cycle_lrs, (str, bytes)) or not cycle_lrs:
            raise ValueError("cycle_lrs must be a non-empty sequence of learning rates.")
        if any(isinstance(lr, bool) or not isinstance(lr, Real) or not math.isfinite(lr) or lr < 0 for lr in cycle_lrs):
            raise ValueError("cycle_lrs must contain finite, non-negative learning rates.")


def get_cycle_lr_lambdas(
    optimizer: Optimizer,
    lr_lambda: Callable[[int, Optional[int]], float],
    num_warmup_steps: int,
    cycle_steps: Optional[Sequence[int]],
    cycle_lrs: Optional[Sequence[float]],
) -> Callable[[int], float]:
    """Wrap a within-cycle LR multiplier with restarts, preserving parameter-group LR ratios."""
    validate_lr_cycle_config(cycle_steps, cycle_lrs)
    if cycle_steps is None:
        return lambda current_step: lr_lambda(current_step, None)
    if not 0 <= num_warmup_steps < min(cycle_steps):
        raise ValueError("Cyclic schedules require 0 <= num_warmup_steps < every cycle length.")

    base_lrs = [group.get("initial_lr", group["lr"]) for group in optimizer.param_groups]
    if cycle_lrs is not None and (not math.isfinite(base_lrs[0]) or base_lrs[0] <= 0):
        raise ValueError("Per-cycle learning rates require a positive, finite optimizer base LR.")
    peak_ratios = tuple(lr / base_lrs[0] for lr in cycle_lrs) if cycle_lrs is not None else (1.0,)
    lengths = tuple(cycle_steps)
    boundaries = tuple(accumulate(lengths))

    def cyclic_lr_lambda(current_step: int) -> float:
        cycle_index = bisect_right(boundaries, current_step)
        if cycle_index < len(lengths):
            cycle_step = current_step - (boundaries[cycle_index - 1] if cycle_index else 0)
            cycle_length = lengths[cycle_index]
        else:
            later_cycle, cycle_step = divmod(current_step - boundaries[-1], lengths[-1])
            cycle_index += later_cycle
            cycle_length = lengths[-1]
        return peak_ratios[min(cycle_index, len(peak_ratios) - 1)] * lr_lambda(cycle_step, cycle_length)

    return cyclic_lr_lambda

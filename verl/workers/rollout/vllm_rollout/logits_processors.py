# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Custom logits processors for vLLM V1 rollout.

Provides an ``AdapterLogitsProcessor`` subclass registered at engine init via
the ``logits_processors`` engine kwarg.  Per-request parameters are read from
``SamplingParams.extra_args``.

Reference: https://github.com/vllm-project/vllm/issues/757
"""

import torch
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


# ---------------------------------------------------------------------------
# Core: n-gram banning logic
# ---------------------------------------------------------------------------

def _calc_banned_ngram_tokens(
    ngram_size: int, output_ids: list[int], window_size: int
) -> set[int]:
    """Return token ids that would create a repeated n-gram if generated next."""
    if len(output_ids) < ngram_size:
        return set()

    current_prefix = tuple(output_ids[-(ngram_size - 1) :])
    search_start = max(0, len(output_ids) - window_size)
    search_end = len(output_ids) - ngram_size + 1

    banned: set[int] = set()
    for i in range(search_start, search_end):
        ngram = tuple(output_ids[i : i + ngram_size])
        if ngram[:-1] == current_prefix:
            banned.add(ngram[-1])
    return banned


# ---------------------------------------------------------------------------
# V1 model-level adapter  (registered via engine logits_processors kwarg)
# ---------------------------------------------------------------------------

class NoRepeatNGramV1Adapter(AdapterLogitsProcessor):
    """vLLM V1 model-level adapter that creates per-request n-gram processors.

    Per-request parameters are read from ``SamplingParams.extra_args``:
      - ``ngram_size`` (int): n-gram size. 0 or absent → disabled.
      - ``ngram_window`` (int, default 100): search window.
    """

    def is_argmax_invariant(self) -> bool:
        # Banning tokens changes which token gets highest probability.
        return False

    def new_req_logits_processor(self, params):
        extra = params.extra_args or {}
        ngram_size = extra.get("ngram_size", 0)
        if ngram_size <= 0:
            return None
        window_size = extra.get("ngram_window", 100)
        return _NoRepeatNGramPerRequest(ngram_size, window_size)


class _NoRepeatNGramPerRequest:
    """Per-request processor with ``(output_ids, logits) -> logits`` signature."""

    def __init__(self, ngram_size: int, window_size: int):
        self.ngram_size = ngram_size
        self.window_size = window_size

    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        banned = _calc_banned_ngram_tokens(self.ngram_size, output_ids, self.window_size)
        if banned:
            logits[list(banned)] = -float("inf")
        return logits


# FQCN for passing to engine kwargs (logits_processors=[...])
NOREPEAT_NGRAM_V1_FQCN = (
    "verl.workers.rollout.vllm_rollout.logits_processors:NoRepeatNGramV1Adapter"
)

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
"""Custom logits processors for vLLM rollout.

Based on vLLM's NoRepeatNGramLogitsProcessor from deepseek_ocr.py:
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/deepseek_ocr.py#L134-L185
"""

import torch


class NoRepeatNGramLogitsProcessor:
    """Logits processor that prevents repeated n-grams during generation.

    Compatible with vLLM's logits_processors interface:
        (prompt_token_ids: list[int], past_token_ids: list[int], scores: torch.Tensor) -> torch.Tensor

    Args:
        ngram_size: Size of the n-gram to prevent repeating. Must be > 0.
        window_size: Only search for repeated n-grams within the last
            ``window_size`` generated tokens. Limits cost on long sequences.
        whitelist_token_ids: Optional set of token ids that are never banned
            even if they would complete a repeated n-gram.
    """

    def __init__(
        self,
        ngram_size: int,
        window_size: int = 100,
        whitelist_token_ids: set[int] | None = None,
    ):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, got {ngram_size}.")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"`window_size` has to be a strictly positive integer, got {window_size}.")
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()

    def __call__(
        self, prompt_token_ids: list[int], past_token_ids: list[int], scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # Only look at generated (past) tokens, not the prompt.
        output_ids = list(past_token_ids)
        if len(output_ids) < self.ngram_size:
            return scores

        current_prefix = tuple(output_ids[-(self.ngram_size - 1) :])

        search_start = max(0, len(output_ids) - self.window_size)
        search_end = len(output_ids) - self.ngram_size + 1

        banned_tokens: set[int] = set()
        for i in range(search_start, search_end):
            ngram = tuple(output_ids[i : i + self.ngram_size])
            if ngram[:-1] == current_prefix:
                banned_tokens.add(ngram[-1])

        banned_tokens = banned_tokens - self.whitelist_token_ids

        if banned_tokens:
            scores[list(banned_tokens)] = -float("inf")

        return scores

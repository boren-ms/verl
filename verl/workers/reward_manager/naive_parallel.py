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

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive_parallel")
class NaiveParallelRewardManager(AbstractRewardManager):
    """Same behavior as :class:`NaiveRewardManager`, but runs ``compute_score``
    concurrently across samples with a thread pool.

    Use this when ``compute_score`` is dominated by I/O-bound long calls
    (e.g. HTTP requests to an LLM judge server). Threads release the GIL
    during network I/O, so a single pool of N threads can fan out N
    in-flight judge calls without process/pickling overhead.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        num_workers: int = 32,
    ) -> None:
        """
        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, ``default_compute_score`` is used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data.
            num_workers: Number of worker threads used to parallelize ``compute_score`` calls.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.num_workers = max(1, int(num_workers))

    def _prepare_item(self, data_item):
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_source = data_item.non_tensor_batch[self.reward_fn_key]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        extra_info["num_turns"] = num_turns

        return {
            "prompt_str": prompt_str,
            "response_str": response_str,
            "ground_truth": ground_truth,
            "data_source": data_source,
            "extra_info": extra_info,
            "valid_response_length": int(valid_response_length),
        }

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list] = defaultdict(list)

        # Pre-decode and gather per-sample inputs serially (cheap, tokenizer calls).
        items = [self._prepare_item(data[i]) for i in range(len(data))]

        def _score_one(item):
            return self.compute_score(
                data_source=item["data_source"],
                solution_str=item["response_str"],
                ground_truth=item["ground_truth"],
                extra_info=item["extra_info"],
            )

        # Fan out the (potentially slow) compute_score calls. Results stay
        # in index order because ThreadPoolExecutor.map preserves order.
        n = len(items)
        if n == 0:
            scores: list = []
        else:
            workers = min(self.num_workers, n)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                scores = list(pool.map(_score_one, items))

        already_print_data_sources: dict = {}

        for i, (item, score) in enumerate(zip(items, scores, strict=True)):
            if isinstance(score, dict):
                reward = score["score"]
                for key in reward_extra_info.keys() - score.keys():
                    reward_extra_info[key].append(1.0)
                for key, value in score.items():
                    if key not in reward_extra_info:
                        reward_extra_info[key].extend([1.0] * i)
                    reward_extra_info[key].append(value)
            else:
                reward = score
                for key in reward_extra_info:
                    reward_extra_info[key].append(1.0)

            valid_response_length = item["valid_response_length"]
            reward_tensor[i, valid_response_length - 1] = reward

            data_source = item["data_source"]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", item["prompt_str"])
                print("[response]", item["response_str"])
                print("[ground_truth]", item["ground_truth"])
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

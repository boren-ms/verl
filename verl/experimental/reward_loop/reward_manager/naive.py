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

import inspect

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score


@register("naive")
class NaiveRewardManager(RewardManagerBase):
    """The reward manager."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        # Number of decoded train samples to print per data source (debugging).
        # Sourced from data.train_num_examine; 0 disables example logging.
        data_cfg = getattr(config, "data", None)
        self.num_examine = int((data_cfg.get("train_num_examine", 0) if data_cfg is not None else 0) or 0)
        self._already_print_data_sources: dict[str, int] = {}

    async def run_single(self, data: DataProto) -> dict:
        data = data[-1:]  # for multi-sequence outputs, we only compute reward based on the last sequence
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())
        sample_extra_info = dict(extra_info)
        skip_examine = data_item.meta_info.get("skip_examine", False)
        sample_extra_info["baseline_score"] = data_item.batch["reward_baselines"].item() if "reward_baselines" in data_item.batch else None
        sample_extra_info["baseline_response"] = data_item.non_tensor_batch.get("baseline_response", None)

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            sample_extra_info.update(result.pop("extra_info", {}))
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score
        self._maybe_log_example(
            data_item=data_item,
            data_source=data_source,
            response_str=response_str,
            ground_truth=ground_truth,
            score=score,
            result=result,
            extra_info=sample_extra_info,
            skip_examine=skip_examine,
        )

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}

    def _maybe_log_example(
        self,
        *,
        data_item,
        data_source,
        response_str,
        ground_truth,
        score,
        result,
        extra_info,
        skip_examine,
    ):
        """Print a few decoded train samples per data source for debugging.

        Mirrors the example-logging behaviour of the legacy DAPO reward manager,
        gated by ``data.train_num_examine``. Counting is per ``data_source`` and
        capped at ``self.num_examine``.
        """
        if self.num_examine <= 0 or skip_examine:
            return

        printed = self._already_print_data_sources.get(data_source, 0)
        if printed >= self.num_examine:
            return
        self._already_print_data_sources[data_source] = printed + 1

        prompt_str = None
        try:
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        except Exception:  # noqa: BLE001 - logging must never break reward computation
            prompt_str = None

        pfx = f"[{self._already_print_data_sources[data_source]}]"
        print(f"====== train sample {pfx} (data_source={data_source}) ======")
        if prompt_str is not None:
            print(f"{pfx}[prompt]", prompt_str)
        for key, value in extra_info.items():
            print(f"{pfx}[{key}]", value)
        print(f"{pfx}[ground_truth]", ground_truth)
        print(f"{pfx}[response]", response_str)
        scores = []
        if isinstance(result, dict):
            for key, value in result.items():
                scores.append(f"{key}={value}")
        else:
            scores.append(f"score={score}")
        print(pfx, "; ".join(scores))

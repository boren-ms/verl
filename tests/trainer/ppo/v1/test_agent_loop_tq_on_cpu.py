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

import asyncio
from unittest.mock import patch

import torch

from verl.trainer.ppo.v1.agent_loop_tq import (
    _attach_remax_baseline,
    _settle_session_tasks,
    apply_greedy_sampling_params,
)


def test_greedy_sampling_disables_top_k_and_temperature():
    params = {"top_p": 0.9, "top_k": -1, "temperature": 0.7}

    apply_greedy_sampling_params(params)

    assert params == {"top_p": 1.0, "top_k": -1, "temperature": 0.0, "min_tokens": 1}


def test_settle_session_tasks_waits_for_siblings_after_failure():
    async def run():
        settled = asyncio.Event()

        async def fail():
            raise RuntimeError("session failed")

        async def finish_later():
            await asyncio.sleep(0.01)
            settled.set()

        tasks = [asyncio.create_task(fail()), asyncio.create_task(finish_later())]
        errors = await _settle_session_tasks(tasks)

        assert settled.is_set()
        assert all(task.done() for task in tasks)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    asyncio.run(run())


def test_attach_remax_baseline_writes_sampled_rewards_and_clears_baseline():
    class _NestedRewards:
        def to_padded_tensor(self, padding):
            assert padding == 0
            return torch.tensor([[0.25, 0.75]])

    with (
        patch(
            "verl.trainer.ppo.v1.agent_loop_tq.tq.kv_batch_get",
            return_value={"rm_scores": _NestedRewards()},
        ) as get_batch,
        patch("verl.trainer.ppo.v1.agent_loop_tq.tq.kv_batch_put") as put_batch,
        patch("verl.trainer.ppo.v1.agent_loop_tq.tq.kv_clear") as clear,
    ):
        asyncio.run(
            _attach_remax_baseline(
                sampled_keys=["uid_0_0", "uid_1_0"],
                baseline_keys=["uid_2_0"],
                partition_id="train",
            )
        )

    get_batch.assert_called_once_with(
        keys=["uid_2_0"],
        partition_id="train",
        select_fields=["rm_scores"],
    )
    fields = put_batch.call_args.kwargs["fields"]
    assert torch.equal(fields["reward_baselines"], torch.tensor([1.0, 1.0]))
    clear.assert_called_once_with(keys=["uid_2_0"], partition_id="train")

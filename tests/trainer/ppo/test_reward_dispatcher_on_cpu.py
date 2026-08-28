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

import asyncio
import inspect

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import get_reward_fn_dispatcher, get_val_reward_fn


def _config(**overrides):
    config = {
        "reward": {"custom_reward_function": {}},
        **overrides,
    }
    return OmegaConf.create(config)


@pytest.fixture
def external_rewards(monkeypatch):
    calls = []

    def first_reward(*, data_source, solution_str, scale=1):
        calls.append(("first", data_source, solution_str, scale))
        return scale

    def second_reward(*, data_source, solution_str, scale=1):
        calls.append(("second", data_source, solution_str, scale))
        return scale * 2

    async def async_reward(*, data_source, solution_str, scale=1):
        calls.append(("async", data_source, solution_str, scale))
        return scale * 3

    rewards = {"first.py": first_reward, "second.py": second_reward, "async.py": async_reward}
    monkeypatch.setattr(
        "verl.utils.import_utils.load_extern_object",
        lambda module_path, object_name: rewards[module_path],
    )
    return calls


def test_dispatches_named_reward_functions_by_data_source(external_rewards):
    config = _config(
        reward_functions={
            "shared": {"path": "first.py", "name": "score", "reward_kwargs": {"scale": 4}},
            "other": {"path": "second.py", "name": "score"},
        },
        reward_function_by_data_source={"source_a": "shared", "source_b": "other"},
    )

    reward_fn = get_reward_fn_dispatcher(config)

    assert reward_fn(data_source="source_a", solution_str="a") == 4
    assert reward_fn(data_source="source_b", solution_str="b", scale=5) == 10
    assert external_rewards == [("first", "source_a", "a", 4), ("second", "source_b", "b", 5)]


def test_rejects_unknown_reward_registration(external_rewards):
    config = _config(
        reward_functions={"known": {"path": "first.py", "name": "score"}},
        reward_function_by_data_source={"source": "missing"},
    )

    with pytest.raises(ValueError, match="Unknown reward function registrations: \\['missing'\\]"):
        get_reward_fn_dispatcher(config)


def test_uses_custom_reward_function_as_fallback(external_rewards):
    config = _config(
        reward={"custom_reward_function": {"path": "second.py", "name": "score"}},
        reward_functions={"source_a": {"path": "first.py", "name": "score"}},
    )

    reward_fn = get_reward_fn_dispatcher(config)

    assert reward_fn(data_source="unregistered", solution_str="fallback") == 2
    assert external_rewards == [("second", "unregistered", "fallback", 1)]


def test_validation_dispatches_by_data_source(external_rewards):
    config = _config(
        val_reward={
            "reward_functions": {
                "first": {"path": "first.py", "name": "score"},
                "second": {"path": "second.py", "name": "score"},
            },
            "reward_function_by_data_source": {"val_a": "first", "val_b": "second"},
        }
    )

    reward_fn = get_val_reward_fn(config)

    assert reward_fn(data_source="val_a", solution_str="a") == 1
    assert reward_fn(data_source="val_b", solution_str="b") == 2


def test_dispatcher_preserves_async_reward_functions(external_rewards):
    config = _config(
        reward_functions={"async": {"path": "async.py", "name": "score", "reward_kwargs": {"scale": 2}}},
        reward_function_by_data_source={"source": "async"},
    )

    reward_fn = get_reward_fn_dispatcher(config)

    assert inspect.iscoroutinefunction(reward_fn)
    assert asyncio.run(reward_fn(data_source="source", solution_str="async")) == 6
# Copyright 2025 Individual Contributor: Thibaut Barroyer
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
from __future__ import annotations

import inspect
import multiprocessing
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, cast

from verl import DataProto
from verl.utils.reward_score import get_default_compute_score

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from verl.experimental.reward_loop.reward_manager.base import RawRewardFn, RewardManagerBase
    from verl.trainer.config.config import ModuleConfig
    from verl.workers.config.reward import RewardManagerConfig


def _call_with_kwargs(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.

    This function is used to merge additional keyword arguments with the original function's arguments.
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return raw_fn(*args, **merged_kwargs)


async def _call_with_kwargs_async(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.

    This function is used to merge additional keyword arguments with the original function's arguments.
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return await raw_fn(*args, **merged_kwargs)


def get_custom_reward_fn(
    config: DictConfig, reward_fn_config: Optional[DictConfig | dict] = None
) -> Optional[RawRewardFn]:
    """Load and return a custom reward function from external file.

    Dynamically imports a reward function from a specified file path and wraps
    it with additional keyword arguments from the configuration.

    Args:
        config (dict): Configuration dictionary containing custom_reward_function
                      settings with 'path', 'name', and 'reward_kwargs' fields.

    Returns:
        callable or None: Wrapped reward function with merged kwargs, or None
                         if no custom reward function is configured.

    Raises:
        FileNotFoundError: If the specified reward function file doesn't exist.
        RuntimeError: If there's an error loading the module from file.
        AttributeError: If the specified function name isn't found in the module.
    """

    if reward_fn_config is None:
        reward_fn_config = config.reward.get("custom_reward_function") or {}
    module_path = reward_fn_config.get("path")
    if not module_path:
        return None

    fn_name = reward_fn_config.get("name")
    assert fn_name is not None

    from verl.utils.import_utils import load_extern_object

    raw_fn = load_extern_object(module_path=module_path, object_name=fn_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))
    if not inspect.iscoroutinefunction(raw_fn):
        return partial(_call_with_kwargs, raw_fn, reward_kwargs)
    else:
        return partial(_call_with_kwargs_async, raw_fn, reward_kwargs)


def _get_reward_fn_dispatcher(
    config: DictConfig,
    dispatcher_config: DictConfig | dict,
    fallback_reward_fn_config: Optional[DictConfig | dict] = None,
) -> Optional[RawRewardFn]:
    """Load named reward functions and dispatch calls by data source."""
    reward_fn_configs = dispatcher_config.get("reward_functions") or {}
    if not reward_fn_configs:
        return get_custom_reward_fn(config, fallback_reward_fn_config)

    reward_function_by_data_source = dispatcher_config.get("reward_function_by_data_source") or {}
    if reward_function_by_data_source:
        registered_reward_functions = {
            name: get_custom_reward_fn(config, reward_fn_config)
            for name, reward_fn_config in reward_fn_configs.items()
        }
        unknown = sorted(set(reward_function_by_data_source.values()) - set(registered_reward_functions))
        if unknown:
            raise ValueError(f"Unknown reward function registrations: {unknown}")
        reward_functions = {
            data_source: registered_reward_functions[registration]
            for data_source, registration in reward_function_by_data_source.items()
        }
    else:
        # Backward-compatible form: reward_functions is keyed directly by data source.
        reward_functions = {
            data_source: get_custom_reward_fn(config, reward_fn_config)
            for data_source, reward_fn_config in reward_fn_configs.items()
        }

    missing = [data_source for data_source, reward_fn in reward_functions.items() if reward_fn is None]
    if missing:
        raise ValueError(f"Reward function registrations require a path: {missing}")

    fallback_reward_fn = get_custom_reward_fn(config, fallback_reward_fn_config)

    def resolve_reward_fn(data_source):
        reward_fn = reward_functions.get(data_source, fallback_reward_fn)
        if reward_fn is None:
            available = ", ".join(sorted(reward_functions))
            raise ValueError(
                f"No reward function registered for data source {data_source!r}. "
                f"Available registrations: {available}. Configure custom_reward_function as a fallback."
            )
        return reward_fn

    if any(inspect.iscoroutinefunction(reward_fn) for reward_fn in [*reward_functions.values(), fallback_reward_fn]):

        async def dispatch_reward(data_source, *args, **kwargs):
            result = resolve_reward_fn(data_source)(*args, data_source=data_source, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

    else:

        def dispatch_reward(data_source, *args, **kwargs):
            return resolve_reward_fn(data_source)(*args, data_source=data_source, **kwargs)

    return dispatch_reward


def get_reward_fn_dispatcher(config: DictConfig) -> Optional[RawRewardFn]:
    """Load training reward functions registered for individual data sources."""
    fallback_reward_fn_config = config.reward.get("custom_reward_function") or {}
    return _get_reward_fn_dispatcher(config, config, fallback_reward_fn_config)


def get_val_reward_fn(config: DictConfig) -> Optional[RawRewardFn]:
    """Load and return a validation reward function from config.val_reward.custom_reward_function.

    If config.val_reward is not set, returns None (use the same reward as training).
    """
    val_reward_config = config.get("val_reward")
    if val_reward_config is None:
        return None

    fallback_reward_fn_config = val_reward_config.get("custom_reward_function") or {}
    return _get_reward_fn_dispatcher(config, val_reward_config, fallback_reward_fn_config)


def resolve_reward_manager_cls(config: DictConfig) -> type[RewardManagerBase]:
    """Resolve the reward manager class from ``config`` without instantiating it."""
    reward_manager_cfg: RewardManagerConfig = config.reward.reward_manager
    if reward_manager_cfg.source == "register":
        from verl.experimental.reward_loop.reward_manager import get_reward_manager_cls

        return get_reward_manager_cls(reward_manager_cfg.name)
    elif reward_manager_cfg.source == "importlib":
        from verl.utils.import_utils import load_extern_object

        module_cfg: ModuleConfig | None = reward_manager_cfg.module
        assert module_cfg is not None and module_cfg.path is not None, (
            f"Module path is required when {reward_manager_cfg.source=}, but got {module_cfg=}"
        )
        return cast(
            "type[RewardManagerBase]",
            load_extern_object(module_path=module_cfg.path, object_name=reward_manager_cfg.name),
        )
    else:
        raise ValueError(f"Unknown reward manager source: {reward_manager_cfg.source}")


def load_reward_manager(config: DictConfig, tokenizer: Any, **reward_kwargs: Any) -> RewardManagerBase:
    """
    Load and initialize a reward manager based on the configuration.

    Args:
        config: PPO trainer configuration object containing reward_model fields.
        tokenizer: Tokenizer object used for processing text.
        **reward_kwargs: Additional keyword arguments for the reward manager.

    Returns:
        An instance of the specified reward manager class.
    """

    # Try to get a custom reward function based on the configuration
    # user defined reward manager can be registered in custom_reward_fn
    compute_score = get_reward_fn_dispatcher(config)
    final_compute_score = compute_score

    reward_manager_cfg: RewardManagerConfig = config.reward.reward_manager
    reward_manager_cls = resolve_reward_manager_cls(config)

    default_compute_score_ = get_default_compute_score(reward_manager_cfg.name)

    if compute_score is None:
        sandbox_config = config.reward.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        memory_limit_mb = sandbox_config.get("memory_limit_mb", 1024) if sandbox_config else 1024
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            # Create a semaphore to control concurrent access to the sandbox
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            final_compute_score = partial(
                default_compute_score_,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=_concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
            )
        else:
            final_compute_score = default_compute_score_

    # Instantiate and return the reward manager with the specified parameters
    return reward_manager_cls(
        config=config,
        tokenizer=tokenizer,
        compute_score=final_compute_score,
        **reward_kwargs,
    )


def extract_reward(batch: DataProto):
    """
    Extract reward tensor and extra info from batch data.
    """
    reward_tensor = batch.batch["rm_scores"]
    reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
    reward_extra_infos_dict = {key: batch.non_tensor_batch[key] for key in reward_extra_keys}
    return reward_tensor, reward_extra_infos_dict

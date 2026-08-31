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

import warnings
from copy import deepcopy
from enum import Enum

import torch
from omegaconf import DictConfig, OmegaConf

from verl import DataProto
from verl.single_controller.base import Worker

WorkerType = type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


def need_reference_policy(
    role_worker_mapping: dict[Role, WorkerType],
) -> bool:
    """Given a role worker mapping, do we need ref policy."""
    return Role.RefPolicy in role_worker_mapping


def use_actor_as_reference(config: DictConfig) -> bool:
    """Use the adapter-disabled actor as reference only when no external reference model is configured."""
    lora_rank = OmegaConf.select(config, "actor_rollout_ref.model.lora_rank", default=0) or 0
    ref_model_path = OmegaConf.select(config, "actor_rollout_ref.ref.model.path", default=None)
    return lora_rank > 0 and not ref_model_path


def build_teacher_context_batch(data: DataProto) -> DataProto:
    """Replace the student prompt with a pre-tokenized teacher prompt while preserving sampled responses."""
    if "teacher_input_ids" not in data.batch:
        return data

    teacher_data = deepcopy(data)
    responses = data.batch["responses"]
    response_length = responses.size(1)
    response_attention_mask = data.batch["attention_mask"][:, -response_length:]
    teacher_attention_mask = data.batch["teacher_attention_mask"]

    teacher_data.batch["input_ids"] = torch.cat([data.batch["teacher_input_ids"], responses], dim=-1)
    teacher_data.batch["attention_mask"] = torch.cat(
        [teacher_attention_mask, response_attention_mask], dim=-1
    )
    teacher_data.batch["position_ids"] = torch.clamp(
        torch.cumsum(teacher_data.batch["attention_mask"], dim=-1) - 1, min=0
    )
    return teacher_data


def need_reward_model(
    role_worker_mapping: dict[Role, WorkerType],
) -> bool:
    """Given a role worker mapping, do we need reward model."""
    return Role.RewardModel in role_worker_mapping


def need_critic(config: DictConfig) -> bool:
    """Given a config, do we need critic."""
    from verl.trainer.ppo.core_algos import AdvantageEstimator

    if config.critic.enable is not None:
        return bool(config.critic.enable)
    elif config.algorithm.adv_estimator == AdvantageEstimator.GAE:
        return True
    else:
        warnings.warn(
            "Disabled critic as algorithm.adv_estimator != gae. If it is not intended, please set critic.enable=True",
            stacklevel=2,
        )
        return False

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

import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.utils import build_teacher_context_batch, use_actor_as_reference


def test_lora_actor_is_reference_without_external_model():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"lora_rank": 32},
                "ref": {"model": {"path": None}},
            }
        }
    )

    assert use_actor_as_reference(config)


def test_external_teacher_overrides_lora_actor_reference():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"lora_rank": 32},
                "ref": {"model": {"path": "teacher-checkpoint"}},
            }
        }
    )

    assert not use_actor_as_reference(config)


def test_full_finetune_uses_separate_reference_worker():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"lora_rank": 0},
                "ref": {"model": None},
            }
        }
    )

    assert not use_actor_as_reference(config)


def test_build_teacher_context_batch_preserves_student_response():
    data = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[0, 11, 12, 21, 22]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 0]]),
            "position_ids": torch.tensor([[0, 0, 1, 2, 0]]),
            "responses": torch.tensor([[21, 22]]),
            "teacher_input_ids": torch.tensor([[0, 31, 32]]),
            "teacher_attention_mask": torch.tensor([[0, 1, 1]]),
        }
    )

    teacher_data = build_teacher_context_batch(data)

    torch.testing.assert_close(teacher_data.batch["input_ids"], torch.tensor([[0, 31, 32, 21, 22]]))
    torch.testing.assert_close(teacher_data.batch["attention_mask"], torch.tensor([[0, 1, 1, 1, 0]]))
    torch.testing.assert_close(teacher_data.batch["responses"], data.batch["responses"])
    torch.testing.assert_close(data.batch["input_ids"], torch.tensor([[0, 11, 12, 21, 22]]))

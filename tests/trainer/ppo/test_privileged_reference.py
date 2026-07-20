import numpy as np
import torch

from verl.protocol import DataProto
from verl.trainer.ppo.privileged_reference import build_privileged_reference_batch


def test_build_privileged_reference_batch_uses_teacher_prompt_and_sampled_response():
    responses = torch.tensor([[7, 8, 0], [9, 0, 0]])
    response_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    privileged_input_ids = torch.tensor([[0, 0, 31, 32], [0, 41, 42, 43]])
    privileged_attention_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    privileged_position_ids = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 2]])
    privileged_multimodal_inputs = np.array([{"teacher": "one"}, {"teacher": "two"}], dtype=object)
    data = DataProto.from_dict(
        tensors={
            "privileged_input_ids": privileged_input_ids,
            "privileged_attention_mask": privileged_attention_mask,
            "privileged_position_ids": privileged_position_ids,
            "responses": responses,
            "response_mask": response_mask,
        },
        non_tensors={"privileged_multi_modal_inputs": privileged_multimodal_inputs},
        meta_info={"temperature": 1.0},
    )

    teacher_batch = build_privileged_reference_batch(data)

    assert torch.equal(teacher_batch.batch["input_ids"], torch.cat((privileged_input_ids, responses), dim=-1))
    assert torch.equal(
        teacher_batch.batch["attention_mask"], torch.cat((privileged_attention_mask, response_mask), dim=-1)
    )
    assert torch.equal(
        teacher_batch.batch["position_ids"],
        torch.tensor([[0, 0, 0, 1, 2, 3, 4], [0, 0, 1, 2, 3, 4, 5]]),
    )
    assert torch.equal(teacher_batch.batch["responses"], responses)
    assert teacher_batch.non_tensor_batch["multi_modal_inputs"] is privileged_multimodal_inputs
    assert teacher_batch.meta_info == {"temperature": 1.0}
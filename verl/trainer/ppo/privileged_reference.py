import torch

from verl.protocol import DataProto


def build_privileged_reference_batch(data: DataProto) -> DataProto:
    """Replace the student prompt with a privileged prompt for reference scoring."""
    required_batch_keys = {
        "privileged_input_ids",
        "privileged_attention_mask",
        "privileged_position_ids",
        "responses",
    }
    missing_keys = required_batch_keys - set(data.batch.keys())
    if missing_keys:
        raise KeyError(f"Privileged reference scoring requires batch keys: {sorted(missing_keys)}")

    responses = data.batch["responses"]
    prompt_position_ids = data.batch["privileged_position_ids"]
    response_length = responses.size(-1)
    position_offset = torch.arange(1, response_length + 1, device=prompt_position_ids.device)
    position_offset = position_offset.view(*([1] * (prompt_position_ids.ndim - 1)), response_length)

    tensors = {
        "responses": responses,
        "input_ids": torch.cat((data.batch["privileged_input_ids"], responses), dim=-1),
        "attention_mask": torch.cat((data.batch["privileged_attention_mask"], data.batch["response_mask"]), dim=-1),
        "position_ids": torch.cat((prompt_position_ids, prompt_position_ids[..., -1:] + position_offset), dim=-1),
    }
    non_tensors = {}
    if "privileged_multi_modal_inputs" in data.non_tensor_batch:
        non_tensors["multi_modal_inputs"] = data.non_tensor_batch["privileged_multi_modal_inputs"]

    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=data.meta_info.copy())
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace  # Or use a mock object library

import pytest
import torch

from verl.utils.model import extract_multi_modal_inputs, update_model_config


# Parametrize with different override scenarios
@pytest.mark.parametrize(
    "override_kwargs",
    [
        {"param_a": 5, "new_param": "plain_added"},
        {"param_a": 2, "nested_params": {"sub_param_x": "updated_x", "sub_param_z": True}},
    ],
)
def test_update_model_config(override_kwargs):
    """
    Tests that update_model_config correctly updates attributes,
    handling both plain and nested overrides via parametrization.
    """
    # Create a fresh mock config object for each test case
    mock_config = SimpleNamespace(
        param_a=1, nested_params=SimpleNamespace(sub_param_x="original_x", sub_param_y=100), other_param="keep_me"
    )
    # Apply the updates using the parametrized override_kwargs
    update_model_config(mock_config, override_kwargs)

    # Assertions to check if the config was updated correctly
    if "nested_params" in override_kwargs:  # Case 2: Nested override
        override_nested = override_kwargs["nested_params"]
        assert mock_config.nested_params.sub_param_x == override_nested["sub_param_x"], "Nested sub_param_x mismatch"
        assert mock_config.nested_params.sub_param_y == 100, "Nested sub_param_y should be unchanged"
        assert hasattr(mock_config.nested_params, "sub_param_z"), "Expected nested sub_param_z to be added"
        assert mock_config.nested_params.sub_param_z == override_nested["sub_param_z"], "Value of sub_param_z mismatch"
    else:  # Case 1: Plain override (nested params untouched)
        assert mock_config.nested_params.sub_param_x == "original_x", "Nested sub_param_x should be unchanged"
        assert mock_config.nested_params.sub_param_y == 100, "Nested sub_param_y should be unchanged"
        assert not hasattr(mock_config.nested_params, "sub_param_z"), "Nested sub_param_z should not exist"


def test_extract_ragged_audio_preserves_frames_masks_and_sample_order():
    short = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    long = torch.arange(40, dtype=torch.float32).reshape(1, 5, 8) + 100
    result = extract_multi_modal_inputs(
        [
            {
                "input_audio_embeds": short,
                "audio_attention_mask": torch.ones(1, 3),
                "audio_embed_sizes": torch.tensor([1]),
                "audio_frames": torch.tensor([3]),
            },
            {
                "input_audio_embeds": long,
                "audio_attention_mask": torch.ones(1, 5),
                "audio_embed_sizes": torch.tensor([2]),
                "audio_frames": torch.tensor([5]),
            },
        ]
    )
    assert result["input_audio_embeds"].shape == (2, 5, 8)
    assert result["input_audio_embeds"].is_contiguous()
    torch.testing.assert_close(result["input_audio_embeds"][0, :3], short[0])
    torch.testing.assert_close(result["input_audio_embeds"][1], long[0])
    assert torch.count_nonzero(result["input_audio_embeds"][0, 3:]) == 0
    torch.testing.assert_close(
        result["audio_attention_mask"], torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    )
    assert result["audio_embed_sizes"].tolist() == [1, 2]
    assert result["audio_frames"].tolist() == [3, 5]

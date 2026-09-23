import pytest
import torch.nn as nn

from verl.utils.fsdp_utils import get_fsdp_wrap_policy


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)


class Model(nn.Module):
    _no_split_modules = {"DecoderLayer", "OptionalVisionBlock"}

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer()])


def test_default_wrap_policy_ignores_absent_optional_layer():
    model = Model()
    policy = get_fsdp_wrap_policy(model)

    assert policy(model.layers[0], recurse=False, nonwrapped_numel=4)


def test_configured_wrap_policy_rejects_absent_layer():
    with pytest.raises(ValueError, match="OptionalVisionBlock"):
        get_fsdp_wrap_policy(
            Model(),
            config={"transformer_layer_cls_to_wrap": ["OptionalVisionBlock"]},
        )


def test_default_wrap_policy_requires_matching_layer():
    model = Model()
    model._no_split_modules = {"OptionalVisionBlock"}

    with pytest.raises(ValueError, match="Could not find any transformer layer"):
        get_fsdp_wrap_policy(model)

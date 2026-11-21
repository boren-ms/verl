from types import SimpleNamespace

import torch
from torch import nn
from vibevoice_asr.compat import _init_asr_weights


def test_missing_bias_initialization_preserves_loaded_linear_weight():
    layer = nn.Linear(3, 2)
    with torch.no_grad():
        layer.weight.fill_(7)
        layer.bias.fill_(9)
    layer.weight._is_hf_initialized = True
    model = SimpleNamespace(config=SimpleNamespace(decoder_config=SimpleNamespace(initializer_range=0.02)))

    _init_asr_weights(model, layer)

    assert torch.all(layer.weight == 7)
    assert torch.all(layer.bias == 0)


def test_missing_layer_norm_weight_preserves_loaded_bias():
    layer = nn.LayerNorm(3)
    with torch.no_grad():
        layer.weight.fill_(7)
        layer.bias.fill_(9)
    layer.bias._is_hf_initialized = True

    _init_asr_weights(SimpleNamespace(config=SimpleNamespace()), layer)

    assert torch.all(layer.weight == 1)
    assert torch.all(layer.bias == 9)

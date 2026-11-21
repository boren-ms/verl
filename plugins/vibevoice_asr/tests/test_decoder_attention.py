import importlib.util

import pytest
import torch

if importlib.util.find_spec("vibevoice") is None:
    pytest.skip("Pinned VibeVoice package is required", allow_module_level=True)

from hf_vibevoice_asr.rl import VibeVoiceASRForRL  # noqa: E402
from vibevoice.modular.configuration_vibevoice import VibeVoiceASRConfig  # noqa: E402


def test_decoder_attention_does_not_propagate_to_convolutional_tokenizers():
    tokenizer = {"vae_dim": 8, "encoder_n_filters": 4, "encoder_ratios": [2], "encoder_depths": "1-1"}
    config = VibeVoiceASRConfig(
        acoustic_tokenizer_config={**tokenizer, "decoder_n_filters": 4},
        semantic_tokenizer_config=tokenizer.copy(),
        decoder_config={
            "model_type": "qwen2",
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "tie_word_embeddings": False,
        },
        tie_word_embeddings=False,
        attn_implementation="eager",
        verl_decoder_attn_implementation="sdpa",
    )
    model = VibeVoiceASRForRL(config).train()
    assert model.model.language_model.config._attn_implementation == "sdpa"
    assert model.model.acoustic_tokenizer.config._attn_implementation == "eager"
    assert model.model.semantic_tokenizer.config._attn_implementation == "eager"
    assert all(not frontend.training for frontend in model._frontends())
    model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False).logits.square().mean().backward()
    gradient = model.model.language_model.layers[0].self_attn.q_proj.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.count_nonzero()

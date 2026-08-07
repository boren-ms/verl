# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Copyright (c) Microsoft Corporation Speech Team.
"""Qwen3.5 Audio model configuration — self-contained for trust_remote_code."""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Qwen3_5AudioConfig(PretrainedConfig):
    """Configuration class for Qwen3.5 Audio CausalLM."""

    model_type = "qwen3_5_audio"
    keys_to_ignore_at_inference = ["past_key_values"]
    ignore_keys_at_rope_validation = {"mrope_interleaved", "mrope_section"}

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        layer_types=None,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        embd_layer="default",
        embd_pdrop=0.0,
        audio_processor=None,
        partial_rotary_factor=0.25,
        full_attention_interval=4,
        **kwargs,
    ):
        # Pop keys not relevant to PretrainedConfig
        kwargs.pop("lora", None)
        kwargs.pop("dtype", None)
        kwargs.pop("mamba_ssm_dtype", None)
        kwargs.pop("mlp_only_layers", None)
        kwargs.pop("mtp_num_hidden_layers", None)
        kwargs.pop("mtp_use_dedicated_embeddings", None)
        kwargs.pop("attn_output_gate", None)
        kwargs.pop("transformers_version", None)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_parameters = rope_parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.embd_layer = embd_layer
        self.embd_pdrop = embd_pdrop
        self.audio_processor = audio_processor
        self.partial_rotary_factor = partial_rotary_factor

        if layer_types is None:
            self.layer_types = [
                "linear_attention" if bool((i + 1) % full_attention_interval) else "full_attention"
                for i in range(num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

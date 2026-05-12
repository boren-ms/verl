# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Copyright (c) Microsoft Corporation Speech Team (2025.06.29).
# Bundled from phyagi — adapted for standalone use without phyagi package.
"""Qwen3.5 text model configuration (bundled copy, no @strict/@auto_docstring)."""

from typing import Any

from transformers.configuration_utils import PreTrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Qwen3_5TextConfig(PreTrainedConfig):
    r"""
    Configuration class for the Qwen3.5 text backbone used in Qwen3.5-Audio.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Hidden dimension.
        intermediate_size: FFN intermediate size.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of GQA key/value heads.
        head_dim: Head dimension (may differ from hidden_size / num_attention_heads).
        full_attention_interval: Every N layers use full attention; the rest use linear attention.
        layer_types: Explicit list of "full_attention" / "linear_attention" per layer.
            If None, derived from full_attention_interval.
        embd_layer: Embedding type — "default" or a dict with "embedding_cls" and audio kwargs.
        audio_processor: Optional audio processor config dict.
    """

    model_type = "qwen3_5_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_config_key = "text_config"
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    def __init__(
        self,
        vocab_size: int = 248320,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_parameters: Any = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        head_dim: int = 256,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        layer_types: "list[str] | None" = None,
        pad_token_id: "int | None" = None,
        bos_token_id: "int | None" = None,
        eos_token_id: "int | list[int] | None" = None,
        embd_layer: "str | dict" = "default",
        embd_pdrop: float = 0.0,
        audio_processor: "dict | None" = None,
        full_attention_interval: int = 4,
        partial_rotary_factor: float = 0.25,
        # phyagi extra fields
        attn_output_gate: bool = False,
        mtp_num_hidden_layers: int = 0,
        **kwargs,
    ):
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
        self.attn_output_gate = attn_output_gate
        self.mtp_num_hidden_layers = mtp_num_hidden_layers
        self.lora = None
        # dtype used by some norm layers (keep None = use model default)
        self.dtype = None

        if layer_types is None:
            self.layer_types = [
                "linear_attention" if bool((i + 1) % full_attention_interval) else "full_attention"
                for i in range(num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types

        # partial_rotary_factor fed as a rope_kwargs entry by HF convention
        kwargs.setdefault("partial_rotary_factor", partial_rotary_factor)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Qwen3_5TextConfig"]

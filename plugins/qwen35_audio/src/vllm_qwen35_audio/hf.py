# SPDX-License-Identifier: Apache-2.0
"""HuggingFace registration for converted Qwen3.5-Audio checkpoints."""

from typing import Any

from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig


class Qwen3_5AudioForCausalLM(PreTrainedModel, GenerationMixin):
    """HF wrapper matching the converted Qwen3.5-Audio checkpoint key layout.

    The audio encoder weights live under ``embed_tokens_extend`` and are consumed
    by vLLM. Verl only needs the HF language model so weight synchronization uses
    the same ``language_model.*`` names as the checkpoint and vLLM model.
    """

    config_class = Qwen3_5TextConfig
    base_model_prefix = "language_model"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _no_split_modules = ["Qwen3_5DecoderLayer"]
    _tied_weights_keys = {"language_model.lm_head.weight": "language_model.model.embed_tokens.weight"}

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        self.language_model = Qwen3_5ForCausalLM(config)

    def forward(self, *args: Any, **kwargs: Any):
        return self.language_model(*args, **kwargs)

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def prepare_inputs_for_generation(self, *args: Any, **kwargs: Any):
        return self.language_model.prepare_inputs_for_generation(*args, **kwargs)


def register_hf_qwen35_audio_model() -> None:
    AutoModelForCausalLM.register(Qwen3_5TextConfig, Qwen3_5AudioForCausalLM, exist_ok=True)


register_hf_qwen35_audio_model()

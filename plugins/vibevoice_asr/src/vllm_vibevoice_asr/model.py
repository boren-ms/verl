"""Version adapters for the pinned upstream multimodal implementation."""

import inspect

from vibevoice_asr.compat import prepare_upstream

prepare_upstream()

from vllm.model_executor.models.utils import WeightsMapper  # noqa: E402
from vllm.multimodal.processing import BaseDummyInputsBuilder, ProcessorInputs  # noqa: E402
from vllm_plugin.model import (  # noqa: E402
    VibeVoiceDummyInputsBuilder,
    VibeVoiceForCausalLM,  # noqa: F401
    VibeVoiceMultiModalProcessor,
    VibeVoiceProcessingInfo,
)

# The upstream wrapper contains a LoRA-capable Qwen2 decoder but does not
# advertise its capability or its HF names to vLLM's LoRA loader.
VibeVoiceForCausalLM.supports_lora = True
# New vLLM checks the instance protocol before attaching its actual manager.
VibeVoiceForCausalLM.lora_manager = None
VibeVoiceForCausalLM.is_3d_moe_weight = False
VibeVoiceForCausalLM.is_non_gated_moe = False
VibeVoiceForCausalLM.lora_skip_prefixes = []
VibeVoiceForCausalLM.packed_modules_mapping = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
VibeVoiceForCausalLM.embedding_modules = {
    "embed_tokens": "input_embeddings",
    "lm_head": "output_embeddings",
}
VibeVoiceForCausalLM.hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_substr={"base_layer.": ""},
    orig_to_new_prefix={
        "model.language_model.": "language_model.model.",
        "lm_head.": "language_model.lm_head.",
    },
)

# Modern vLLM constructs ProcessorInputs from parsed multimodal items. Use its
# base builder, which still calls upstream's text/audio dummy-data methods.
if "mm_data" not in inspect.signature(ProcessorInputs).parameters:
    VibeVoiceDummyInputsBuilder.get_dummy_processor_inputs = BaseDummyInputsBuilder.get_dummy_processor_inputs

# The parser moved from the processor to ProcessingInfo in recent vLLM.
if hasattr(VibeVoiceProcessingInfo, "get_data_parser"):
    VibeVoiceProcessingInfo.get_data_parser = VibeVoiceMultiModalProcessor._get_data_parser

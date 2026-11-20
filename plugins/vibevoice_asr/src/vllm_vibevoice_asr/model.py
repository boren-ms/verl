"""Version adapters for the pinned upstream multimodal implementation."""

import inspect

from vibevoice_asr.compat import prepare_upstream

prepare_upstream()

from vllm.multimodal.processing import BaseDummyInputsBuilder, ProcessorInputs  # noqa: E402
from vllm_plugin.model import (  # noqa: E402
    VibeVoiceDummyInputsBuilder,
    VibeVoiceForCausalLM,  # noqa: F401
    VibeVoiceMultiModalProcessor,
    VibeVoiceProcessingInfo,
)

# Modern vLLM constructs ProcessorInputs from parsed multimodal items. Use its
# base builder, which still calls upstream's text/audio dummy-data methods.
if "mm_data" not in inspect.signature(ProcessorInputs).parameters:
    VibeVoiceDummyInputsBuilder.get_dummy_processor_inputs = BaseDummyInputsBuilder.get_dummy_processor_inputs

# The parser moved from the processor to ProcessingInfo in recent vLLM.
if hasattr(VibeVoiceProcessingInfo, "get_data_parser"):
    VibeVoiceProcessingInfo.get_data_parser = VibeVoiceMultiModalProcessor._get_data_parser

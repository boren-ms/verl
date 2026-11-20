"""Opt-in registration of the official VibeVoice multimodal vLLM model."""


def register():
    from vibevoice_asr.compat import prepare_upstream

    prepare_upstream()
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Qwen2AudioProcessor, Qwen2Tokenizer
    from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceASRTextTokenizerFast
    from vllm.model_executor.models import ModelRegistry

    from .config import VibeVoiceConfig

    AutoConfig.register("vibevoice", VibeVoiceConfig, exist_ok=True)
    AutoTokenizer.register(
        VibeVoiceConfig,
        slow_tokenizer_class=Qwen2Tokenizer,
        fast_tokenizer_class=VibeVoiceASRTextTokenizerFast,
        exist_ok=True,
    )
    AutoProcessor.register(VibeVoiceConfig, Qwen2AudioProcessor, exist_ok=True)
    ModelRegistry.register_model("VibeVoice", "vllm_vibevoice_asr.model:VibeVoiceForCausalLM")
    ModelRegistry.register_model("VibeVoiceForASRTraining", "vllm_vibevoice_asr.model:VibeVoiceForCausalLM")

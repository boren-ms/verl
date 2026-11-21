"""Register the pinned official offline model with Hugging Face Auto classes."""


def register_hf_audio_model():
    from vibevoice_asr.compat import prepare_upstream

    prepare_upstream()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    from vibevoice.modular.configuration_vibevoice import VibeVoiceASRConfig
    from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceASRTextTokenizerFast

    from .rl import VibeVoiceASRForRL, VibeVoiceASRProcessor

    AutoConfig.register("vibevoice", VibeVoiceASRConfig, exist_ok=True)
    AutoModelForCausalLM.register(VibeVoiceASRConfig, VibeVoiceASRForRL, exist_ok=True)
    AutoTokenizer.register(
        VibeVoiceASRConfig,
        fast_tokenizer_class=VibeVoiceASRTextTokenizerFast,
        exist_ok=True,
    )
    AutoProcessor.register(VibeVoiceASRConfig, VibeVoiceASRProcessor, exist_ok=True)


register_hf_audio_model()

"""Register the pinned official offline model with Hugging Face Auto classes."""


def register_hf_audio_model():
    from vibevoice_asr.compat import prepare_upstream

    prepare_upstream()
    from transformers import AutoConfig, AutoModelForCausalLM
    from vibevoice.modular.configuration_vibevoice import VibeVoiceASRConfig
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration

    AutoConfig.register("vibevoice", VibeVoiceASRConfig, exist_ok=True)
    AutoModelForCausalLM.register(VibeVoiceASRConfig, VibeVoiceASRForConditionalGeneration, exist_ok=True)


register_hf_audio_model()

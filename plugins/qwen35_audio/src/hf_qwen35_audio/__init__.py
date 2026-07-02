# Self-contained Qwen3.5 Audio model for HF trust_remote_code deployment.
from .configuration_qwen3_5_audio import Qwen3_5AudioConfig
from .modeling_qwen3_5_audio import Qwen3_5AudioForCausalLM

__all__ = ["Qwen3_5AudioConfig", "Qwen3_5AudioForCausalLM", "register_hf_audio_model"]


def register_hf_audio_model() -> None:
    """Register Qwen3.5-Audio with the Transformers Auto* registries.

    Once registered, ``AutoConfig`` / ``AutoModelForCausalLM`` / ``AutoProcessor``
    resolve ``model_type == "qwen3_5_audio"`` checkpoints to the implementation in
    this installed package, so loads no longer depend on the per-checkpoint
    ``trust_remote_code`` ``*.py`` copies (which can be stale or missing). Load the
    checkpoint with ``trust_remote_code=False`` (or strip ``auto_map`` from
    ``config.json``) so the registered package class takes precedence over any
    bundled remote code.

    Idempotent and best-effort: each registration is guarded so importing this
    package never fails if a class is already registered or Transformers internals
    differ across versions.
    """
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except Exception:
        return

    try:
        AutoConfig.register("qwen3_5_audio", Qwen3_5AudioConfig, exist_ok=True)
    except Exception:
        pass
    try:
        AutoModelForCausalLM.register(Qwen3_5AudioConfig, Qwen3_5AudioForCausalLM, exist_ok=True)
    except Exception:
        pass

    # The processor (and its audio feature extractor) are optional extras; register
    # them when available so ``AutoProcessor.from_pretrained`` also resolves locally.
    try:
        from transformers import AutoProcessor

        from .processing_qwen3_5_audio import Qwen3_5AudioProcessor

        AutoProcessor.register(Qwen3_5AudioConfig, Qwen3_5AudioProcessor, exist_ok=True)
    except Exception:
        pass


# Register on import so that simply importing this package (or the parent
# ``vllm_qwen35_audio`` plugin) wires the Auto* registries.
register_hf_audio_model()

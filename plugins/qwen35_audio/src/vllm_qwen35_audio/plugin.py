# SPDX-License-Identifier: Apache-2.0
"""vLLM plugin registration for Qwen3.5-Audio."""

import importlib
import inspect
import os
import sys
import types
import warnings
from contextlib import nullcontext
from typing import Any

ARCHITECTURE = "Qwen3_5AudioForCausalLM"
OFFICIAL_MODEL_CLASS = (
    "vllm.model_executor.models.qwen3_5_audio:Qwen3_5AudioForCausalLM"
)


def _maybe_register_qwen35_config() -> None:
    """Best-effort config registration for official vLLM installs.

    Some vLLM versions already know the qwen3_5 config class. When they do, we
    register it with vLLM's config registry early so converted Qwen3.5-Audio
    checkpoints can be loaded without remote code. For non-vLLM code paths,
    prefer Transformers' native qwen3_5 module so AutoModelForCausalLM can build
    the HF actor model.
    """
    vllm_configs = _maybe_register_qwen35_config_for_vllm()
    if _maybe_register_hf_qwen35_config():
        return
    if vllm_configs is not None:
        _maybe_register_vllm_qwen35_config_for_transformers(*vllm_configs)


def _maybe_register_qwen35_config_for_vllm() -> tuple[type[Any], type[Any]] | None:
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm.transformers_utils.configs.qwen3_5 import (
            Qwen3_5Config,
            Qwen3_5TextConfig,
        )
    except Exception:
        return None

    _CONFIG_REGISTRY.setdefault("qwen3_5", Qwen3_5Config)
    _CONFIG_REGISTRY.setdefault("qwen3_5_text", Qwen3_5TextConfig)
    return Qwen3_5Config, Qwen3_5TextConfig


def _maybe_register_hf_qwen35_config() -> bool:
    try:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
        from transformers.models.qwen3_5.configuration_qwen3_5 import (
            Qwen3_5Config,
            Qwen3_5TextConfig,
        )
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5ForCausalLM,
            Qwen3_5Model,
            Qwen3_5TextModel,
        )
    except Exception:
        return False

    AutoConfig.register("qwen3_5", Qwen3_5Config, exist_ok=True)
    AutoConfig.register("qwen3_5_text", Qwen3_5TextConfig, exist_ok=True)
    AutoModel.register(Qwen3_5Config, Qwen3_5Model, exist_ok=True)
    AutoModel.register(Qwen3_5TextConfig, Qwen3_5TextModel, exist_ok=True)
    AutoModelForCausalLM.register(Qwen3_5TextConfig, Qwen3_5ForCausalLM, exist_ok=True)
    return True


def _maybe_register_vllm_qwen35_config_for_transformers(
    qwen35_config: type[Any],
    qwen35_text_config: type[Any],
) -> None:
    try:
        from transformers import AutoConfig
    except Exception:
        return

    AutoConfig.register("qwen3_5", qwen35_config, exist_ok=True)
    AutoConfig.register("qwen3_5_text", qwen35_text_config, exist_ok=True)


def register() -> None:
    """Register Qwen3.5-Audio with vLLM's out-of-tree model registry."""
    from vllm import ModelRegistry

    _maybe_disable_cudnn()
    _maybe_register_qwen35_config()
    ModelRegistry.register_model(ARCHITECTURE, _resolve_model_class())


def register_transformers_config() -> None:
    """Register Qwen3.5 config aliases for non-vLLM code paths."""
    _maybe_register_qwen35_config()


def _maybe_disable_cudnn() -> None:
    if os.getenv("QWEN35_AUDIO_DISABLE_CUDNN") != "1":
        return

    import torch

    torch.backends.cudnn.enabled = False


def _resolve_model_class() -> str | type[Any]:
    """Resolve Qwen3.5-Audio model class from official vLLM or bundled plugin."""
    try:
        _install_multimodal_profiling_compat()
        _install_multimodal_inputs_compat()
        _install_multimodal_processing_compat()
        _install_flash_attn_rotary_compat()
        _install_qwen35_text_model_compat()
        try:
            module = importlib.import_module("vllm.model_executor.models.qwen3_5_audio")
        except ModuleNotFoundError as e:
            if e.name != "vllm.model_executor.models.qwen3_5_audio":
                raise
            module = importlib.import_module("vllm_qwen35_audio.qwen3_5_audio")
        _install_qwen35_audio_model_compat(module)
        model_cls = getattr(module, ARCHITECTURE)
    except Exception as e:
        raise ImportError(
            f"Failed to load Qwen3.5-Audio model from vLLM or bundled plugin: {e}"
        ) from e

    # Patch in marker methods if missing (for older vLLM versions)
    if not hasattr(model_cls, "_mark_tower_model"):
        model_cls._mark_tower_model = _noop_model_marker
    if not hasattr(model_cls, "_mark_language_model"):
        model_cls._mark_language_model = _noop_model_marker

    return model_cls


def _install_multimodal_profiling_compat() -> None:
    try:
        importlib.import_module("vllm.multimodal.profiling")
        return
    except ModuleNotFoundError as e:
        if e.name != "vllm.multimodal.profiling":
            raise

    from vllm.multimodal.processing import BaseDummyInputsBuilder

    module = types.ModuleType("vllm.multimodal.profiling")
    module.BaseDummyInputsBuilder = BaseDummyInputsBuilder
    sys.modules["vllm.multimodal.profiling"] = module


def _install_multimodal_inputs_compat() -> None:
    import vllm.inputs as inputs
    import vllm.multimodal.inputs as multimodal_inputs

    try:
        from vllm.inputs import MultiModalDataDict
    except ImportError:
        MultiModalDataDict = dict

    if not hasattr(inputs, "MultiModalDataDict"):
        inputs.MultiModalDataDict = MultiModalDataDict
    if not hasattr(multimodal_inputs, "MultiModalDataDict"):
        multimodal_inputs.MultiModalDataDict = MultiModalDataDict


def _install_multimodal_processing_compat() -> None:
    import vllm.multimodal.processing as processing
    from vllm.multimodal.processing import processor

    for name in (
        "BaseMultiModalProcessor",
        "BaseProcessingInfo",
        "PromptReplacement",
        "PromptUpdate",
        "ResolvedPromptUpdate",
    ):
        if not hasattr(processing, name) and hasattr(processor, name):
            setattr(processing, name, getattr(processor, name))


def _install_qwen35_text_model_compat() -> None:
    module = importlib.import_module("vllm.model_executor.models.qwen3_5")
    target_cls = module.Qwen3_5ForCausalLM
    source_cls = module.Qwen3_5ForConditionalGeneration

    for name in (
        "get_mamba_state_dtype_from_config",
        "get_mamba_state_shape_from_config",
        "get_mamba_state_copy_func",
    ):
        if hasattr(target_cls, name):
            continue
        method = getattr(source_cls, name)
        method_func = getattr(method, "__func__", method)
        setattr(target_cls, name, classmethod(method_func))


def _install_flash_attn_rotary_compat() -> None:
    """Let vLLM fall back when an installed flash_attn wheel is ABI-incompatible."""
    try:
        common = importlib.import_module(
            "vllm.model_executor.layers.rotary_embedding.common"
        )
    except ModuleNotFoundError as e:
        if e.name != "vllm.model_executor.layers.rotary_embedding.common":
            raise
        return

    apply_rotary_cls = getattr(common, "ApplyRotaryEmb", None)
    if apply_rotary_cls is None:
        return

    original_init = apply_rotary_cls.__init__
    if getattr(original_init, "_qwen35_audio_flash_compat", False):
        return

    def __init__(
        self,
        enforce_enable: bool = False,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
    ) -> None:
        try:
            original_init(
                self,
                enforce_enable=enforce_enable,
                is_neox_style=is_neox_style,
                enable_fp32_compute=enable_fp32_compute,
            )
        except ImportError as e:
            name = getattr(e, "name", "") or ""
            if "flash_attn" not in name and "flash_attn" not in str(e):
                raise
            if not getattr(__init__, "_warned", False):
                warnings.warn(
                    "Ignoring unavailable flash_attn rotary kernel; vLLM will "
                    "use its CUDA/native rotary fallback.",
                    stacklevel=2,
                )
                __init__._warned = True
            self.is_neox_style = is_neox_style
            self.enable_fp32_compute = enable_fp32_compute
            self.apply_rotary_emb_flash_attn = None

    __init__._qwen35_audio_flash_compat = True
    apply_rotary_cls.__init__ = __init__


def _install_qwen35_audio_model_compat(module: Any) -> None:
    dummy_builder_cls = module.Qwen3_5AudioDummyInputsBuilder
    original = dummy_builder_cls.get_dummy_mm_data
    if getattr(original, "_qwen35_audio_compat", False):
        return
    supports_mm_options = "mm_options" in inspect.signature(original).parameters

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options=None):
        if supports_mm_options:
            return original(self, seq_len, mm_counts, mm_options or {})
        return original(self, seq_len, mm_counts)

    get_dummy_mm_data._qwen35_audio_compat = True
    dummy_builder_cls.get_dummy_mm_data = get_dummy_mm_data


def _noop_model_marker(*_args: object, **_kwargs: object):
    return nullcontext()

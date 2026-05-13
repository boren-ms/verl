# SPDX-License-Identifier: Apache-2.0
"""vLLM plugin registration for Qwen3.5-Audio."""

import importlib
import inspect
import os
import sys
import types
from contextlib import nullcontext
from typing import Any

ARCHITECTURE = "Qwen3_5AudioForCausalLM"
OFFICIAL_MODEL_CLASS = (
    "vllm.model_executor.models.qwen3_5_audio:Qwen3_5AudioForCausalLM"
)


def _maybe_register_qwen35_config() -> None:
    """Best-effort config registration for official vLLM installs.

    Some vLLM versions already know the qwen3_5 config class. When they do, we
    register it with Transformers AutoConfig early so tokenizer/config loading
    can resolve converted Qwen3.5-Audio checkpoints without remote code.
    """
    native_hf_qwen35_text = _native_hf_qwen35_text_available()

    try:
        from transformers import AutoConfig
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm.transformers_utils.configs.qwen3_5 import (
            Qwen3_5Config,
            Qwen3_5TextConfig,
        )
    except Exception:
        return

    _CONFIG_REGISTRY.setdefault("qwen3_5", Qwen3_5Config)
    _CONFIG_REGISTRY.setdefault("qwen3_5_text", Qwen3_5TextConfig)
    if not native_hf_qwen35_text:
        AutoConfig.register("qwen3_5", Qwen3_5Config, exist_ok=True)
        AutoConfig.register("qwen3_5_text", Qwen3_5TextConfig, exist_ok=True)


def _native_hf_qwen35_text_available() -> bool:
    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.for_model("qwen3_5_text")
    except Exception:
        return False
    return type(config) in AutoModelForCausalLM._model_mapping.keys()


def register() -> None:
    """Register Qwen3.5-Audio with vLLM's out-of-tree model registry."""
    from vllm import ModelRegistry

    _maybe_disable_cudnn()
    _maybe_register_qwen35_config()
    ModelRegistry.register_model(ARCHITECTURE, _resolve_model_class())


def _maybe_disable_cudnn() -> None:
    if os.getenv("QWEN35_AUDIO_DISABLE_CUDNN") != "1":
        return

    import torch

    torch.backends.cudnn.enabled = False


def _resolve_model_class() -> str | type[Any]:
    """Resolve Qwen3.5-Audio model class from official vLLM or bundled shim."""
    try:
        _install_multimodal_profiling_compat()
        _install_multimodal_inputs_compat()
        _install_multimodal_processing_compat()
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
            f"Failed to load official vLLM Qwen3.5-Audio model: {e}\n"
            f"Please ensure you are using a vLLM version that includes "
            f"Qwen3.5-Audio support (0.11.0+)."
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
    try:
        from vllm.multimodal.processing import processor
    except ImportError:
        processor = processing

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

"""Compatibility for the pinned upstream implementation on Transformers 5."""

import importlib
import sys
from threading import RLock

_lock = RLock()
_ready = False


def _init_asr_weights(self, module):
    """Preserve loaded parameters when Transformers initializes missing ones.

    Upstream writes through ``param.data``, bypassing Transformers 5.7's
    per-parameter initialization guards and overwriting the loaded head and
    connectors. Check each parameter so a missing bias can still be initialized.
    """
    from torch import nn

    config = getattr(self.config, "language_model_config", None)
    if not hasattr(config, "initializer_range"):
        config = getattr(self.config, "decoder_config", None)
    std = getattr(config, "initializer_range", 0.02)

    def initialize(parameter, function, **kwargs):
        if parameter is not None and not getattr(parameter, "_is_hf_initialized", False):
            function(parameter, **kwargs)

    if isinstance(module, nn.Linear):
        initialize(module.weight, nn.init.normal_, mean=0.0, std=std)
        initialize(module.bias, nn.init.zeros_)
    elif isinstance(module, nn.LayerNorm):
        initialize(module.weight, nn.init.ones_)
        initialize(module.bias, nn.init.zeros_)


def prepare_upstream():
    """Import upstream while allowing its own AutoModel registrations.

    Transformers 5 ships classes with the same names as the original release.
    Its Auto mapping checks class names, so upstream's distinct config classes
    require exist_ok=True. Limit the override to upstream classes and restore
    the registration method immediately after these imports.
    """
    global _ready
    with _lock:
        if _ready:
            return
        from transformers.models.auto.auto_factory import _LazyAutoMapping

        # Transformers 5 folded the fast tokenizer into tokenization_qwen2.
        try:
            importlib.import_module("transformers.models.qwen2.tokenization_qwen2_fast")
        except ModuleNotFoundError as exc:
            name = "transformers.models.qwen2.tokenization_qwen2_fast"
            if exc.name != name:
                raise
            module = importlib.import_module("transformers.models.qwen2.tokenization_qwen2")
            from transformers import Qwen2TokenizerFast

            module.Qwen2TokenizerFast = Qwen2TokenizerFast
            sys.modules[name] = module

        original = _LazyAutoMapping.register

        def register(mapping, key, value, exist_ok=False):
            if key.__module__.startswith("vibevoice."):
                exist_ok = True
            return original(mapping, key, value, exist_ok=exist_ok)

        _LazyAutoMapping.register = register
        try:
            importlib.import_module("vibevoice.modular.modeling_vibevoice_asr")
            importlib.import_module("vibevoice.modular.modeling_vibevoice")
            importlib.import_module("vibevoice.processor.vibevoice_asr_processor")
        finally:
            _LazyAutoMapping.register = original

        from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRPreTrainedModel
        from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceASRTextTokenizerFast

        VibeVoiceASRPreTrainedModel._init_weights = _init_asr_weights
        apply_template = VibeVoiceASRTextTokenizerFast.apply_chat_template

        def apply_chat_template(self, *args, **kwargs):
            # Upstream concatenates this result with a token-ID list. In v5
            # apply_chat_template defaults to returning a BatchEncoding.
            kwargs.setdefault("return_dict", False)
            return apply_template(self, *args, **kwargs)

        VibeVoiceASRTextTokenizerFast.apply_chat_template = apply_chat_template
        _ready = True

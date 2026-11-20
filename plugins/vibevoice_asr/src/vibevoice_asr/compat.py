"""Compatibility for the pinned upstream implementation on Transformers 5."""

import importlib
import sys
from threading import RLock

_lock = RLock()
_ready = False


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

        from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceASRTextTokenizerFast

        apply_template = VibeVoiceASRTextTokenizerFast.apply_chat_template

        def apply_chat_template(self, *args, **kwargs):
            # Upstream concatenates this result with a token-ID list. In v5
            # apply_chat_template defaults to returning a BatchEncoding.
            kwargs.setdefault("return_dict", False)
            return apply_template(self, *args, **kwargs)

        VibeVoiceASRTextTokenizerFast.apply_chat_template = apply_chat_template
        _ready = True

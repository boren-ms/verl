import types
from contextlib import contextmanager
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer


def get_speech_peft_model(model, lora_name):
    config = model.config

    lora_config = LoraConfig(
        r=config.speech_lora["r"],
        lora_alpha=config.speech_lora["lora_alpha"],
        target_modules=config.speech_lora["layer"],
        lora_dropout=config.speech_lora["dp"],
        task_type="CAUSAL_LM",
    )
    get_peft_model(model.model, lora_config, adapter_name=lora_name)
    return model


def init_model(model_id=None, new_lora=None):
    """Initialize the model and processor."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation="flash_attention_2",
    )
    model.set_lora_adapter("speech")
    model = add_adapter_func(model)
    if new_lora:
        model.merge_and_unload()  # merge lora and back to normal Linear
        model = get_speech_peft_model(model, lora_name=new_lora)  # revert peft model
    return model


def can_merge_adapter(model):
    return hasattr(model, "merge_adapter") and hasattr(model, "unmerge_adapter")


@contextmanager
def merge_adapter_if_possible(model, merge=True):
    """Context manager to merge and unmerge adapter layers if the model supports it."""
    if can_merge_adapter(model) and merge:
        model.merge_adapter()
        try:
            yield model
        finally:
            model.unmerge_adapter()
    else:
        yield model


def has_lora_adapter(cls):
    for module in cls.modules():
        if isinstance(module, LoraLayer):
            return True
    return False


def merge_adapter(cls, merge=True, adapter="speech"):
    if isinstance(adapter, str):
        adapter = [adapter]
    for name, module in cls.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        if merge:
            module.merge(adapter_names=adapter)
        else:
            module.unmerge()


def set_lora_adapter(self, adapter_name) -> None:
    for module in self.modules():
        if isinstance(module, LoraLayer):
            if module.merged:
                module.unmerge()
            module.set_adapter(adapter_name)
            module._disable_adapters = False


def unset_lora_adapter(self) -> None:
    for module in self.modules():
        if isinstance(module, LoraLayer):
            for layer_name in module.adapter_layer_names:
                layer = getattr(module, layer_name)
                layer.requires_grad_(False)
            module._disable_adapters = True


def unmerge_adapter(cls):
    return merge_adapter(cls, False)


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def merge_and_unload(model, adapter="speech"):
    if isinstance(adapter, str):
        adapter = [adapter]
    key_list = [key for key, _ in model.named_modules() if "lora" not in key]
    for key in key_list:
        try:
            parent, target, target_name = _get_submodules(model, key)
        except AttributeError:
            continue
        if hasattr(target, "base_layer"):
            target.merge(adapter_names=adapter)
            setattr(parent, target_name, target.get_base_layer())
    return model


def add_adapter_func(obj):
    obj.merge_adapter = types.MethodType(merge_adapter, obj)
    obj.unmerge_adapter = types.MethodType(unmerge_adapter, obj)
    obj.merge_and_unload = types.MethodType(merge_and_unload, obj)
    obj.set_lora_adapter = types.MethodType(set_lora_adapter, obj)
    obj.unset_lora_adapter = types.MethodType(unset_lora_adapter, obj)
    return obj


def merge_model_adapter(model, lora_name="speech"):
    model.set_lora_adapter(lora_name)
    model = add_adapter_func(model)
    model.merge_and_unload()  # merge lora and back to normal Linear
    return model

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from packaging import version as vs

from verl.third_party.vllm import get_version

try:
    from msgspec import field
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.request import LoRARequest
    from vllm.lora.utils import get_adapter_absolute_path
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

    class TensorLoRARequest(LoRARequest):
        peft_config: dict = field(default=None)
        lora_tensors: dict = field(default=None)

except (ImportError, ModuleNotFoundError):
    LoRAModel = None
    LoRARequest = None
    get_adapter_absolute_path = None
    LRUCacheWorkerLoRAManager = None
    TensorLoRARequest = None


STABLE_LORA_ID = 1


def replace_lora_adapter(llm_engine, lora_request: TensorLoRARequest) -> bool:
    """Replace the single in-memory LoRA without changing its request ID."""
    for lora_id in llm_engine.list_loras():
        if not llm_engine.remove_lora(lora_id):
            raise RuntimeError(f"Failed to remove vLLM LoRA adapter {lora_id}.")
    return llm_engine.add_lora(lora_request)


class VLLMHijack:
    @staticmethod
    def hijack():
        def hijack__load_adapter(self, lora_request: TensorLoRARequest) -> LoRAModel:
            """
            based on vllm.lora.worker_manager.WorkerLoRAManager._load_adapter, support load adapter with lora tensors

            Reason:
            VLLM does not support adding LoRA from tensors directly. It only supports adding LoRA via file paths.
            To synchronize the LoRA tensors of the actor model, we need to find a workaround to enable VLLM to
            load memory-based LoRA tensors.
            """
            try:
                supported_lora_modules = self._adapter_manager.supported_lora_modules
                packed_modules_mapping = self._adapter_manager.packed_modules_mapping
                expected_lora_modules: list[str] = []
                for module in supported_lora_modules:
                    if module in packed_modules_mapping:
                        expected_lora_modules.extend(packed_modules_mapping[module])
                    else:
                        expected_lora_modules.append(module)

                expected_lora_modules = list(set(expected_lora_modules))

                lora_tensors = None
                from vllm.lora.peft_helper import PEFTHelper

                if isinstance(lora_request, TensorLoRARequest):
                    peft_config = lora_request.peft_config
                    lora_tensors = lora_request.lora_tensors
                    peft_helper = PEFTHelper.from_dict(peft_config)
                else:
                    lora_path = get_adapter_absolute_path(lora_request.lora_path)

                    peft_helper = PEFTHelper.from_local_dir(
                        lora_path,
                        self.max_position_embeddings,
                        lora_request.tensorizer_config_dict,
                    )

                # Validates the LoRA configuration against requirements before
                # loading weights, throwing an exception if validation fails.
                peft_helper.validate_legal(self.lora_config)

                # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
                # to ensure correct loading of lora weights.
                model = self._adapter_manager.model
                hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)

                # Get model-defined prefixes to skip during LoRA loading.
                lora_skip_prefixes = getattr(model, "lora_skip_prefixes", None)

                if isinstance(lora_request, TensorLoRARequest):
                    lora = self._lora_model_cls.from_lora_tensors(
                        lora_model_id=lora_request.lora_int_id,
                        tensors=lora_tensors,
                        peft_helper=peft_helper,
                        device="cpu",
                        dtype=self.lora_config.lora_dtype,
                        model_vocab_size=self.vocab_size,
                        weights_mapper=hf_to_vllm_mapper,
                        skip_prefixes=lora_skip_prefixes,
                    )
                else:
                    lora = self._lora_model_cls.from_local_checkpoint(
                        lora_path,
                        expected_lora_modules,
                        peft_helper=peft_helper,
                        lora_model_id=lora_request.lora_int_id,
                        device="cpu",
                        dtype=self.lora_config.lora_dtype,
                        model_vocab_size=self.vocab_size,
                        weights_mapper=hf_to_vllm_mapper,
                        skip_prefixes=lora_skip_prefixes,
                    )
            except Exception as e:
                raise e

            return lora

        def do_hijack(target_cls, target_method_name, hooking_method):
            setattr(target_cls, target_method_name, hooking_method)

        if LRUCacheWorkerLoRAManager is None:
            return
        do_hijack(LRUCacheWorkerLoRAManager, "_load_adapter", hijack__load_adapter)


def is_version_ge(pkg: str = "vllm", minver: str = "0.7.3"):
    """check if the package version is greater than or equal to the minimum version"""
    return vs.parse(get_version(pkg)) >= vs.parse(minver)

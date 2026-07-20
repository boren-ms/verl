# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from verl.utils.vllm.utils import STABLE_LORA_ID, replace_lora_adapter


class _FakeLoRAEngine:
    def __init__(self):
        self.adapters = {17, 23}
        self.calls = []

    def list_loras(self):
        return self.adapters.copy()

    def remove_lora(self, lora_id):
        self.calls.append(("remove", lora_id))
        self.adapters.remove(lora_id)
        return True

    def add_lora(self, lora_request):
        self.calls.append(("add", lora_request))
        self.adapters.add(lora_request)
        return True


def test_replace_lora_adapter_reuses_stable_id():
    engine = _FakeLoRAEngine()

    assert replace_lora_adapter(engine, STABLE_LORA_ID)

    assert engine.adapters == {STABLE_LORA_ID}
    assert {lora_id for method, lora_id in engine.calls if method == "remove"} == {17, 23}

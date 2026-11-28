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
"""Import shim for Qwen alignment in explicitly English-only environments.

Qwen imports Nagisa even for English. Put this directory on PYTHONPATH only
for English alignment when Nagisa/DyNet is unavailable; Japanese is disabled.
This does not provide or claim the Nagisa package's functionality.
"""


def tagging(*args, **kwargs):
    raise RuntimeError("Japanese tokenization is disabled in this English-only alignment environment")

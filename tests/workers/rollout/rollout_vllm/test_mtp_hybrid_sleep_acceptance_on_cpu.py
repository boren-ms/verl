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

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("ray")
pytest.importorskip("vllm")

from verl.trainer.ppo.ray_trainer import compute_spec_decode_metrics
from verl.workers.rollout.vllm_rollout import utils as vllm_utils
from verl.workers.rollout.vllm_rollout import vllm_async_server


class _FakeMtpEngine:
    """Minimal model of the MTP failure mode seen in hybrid sleep."""

    def __init__(self):
        self.mtp_drafter_available = True
        self.sleep_levels_that_discard_mtp_drafter = {2}

    async def sleep(self, level: int):
        if level in self.sleep_levels_that_discard_mtp_drafter:
            self.mtp_drafter_available = False

    async def reset_encoder_cache(self):
        pass

    def sync_actor_weights(self):
        pass

    def generate_spec_decode_stats(self):
        num_draft_tokens = 3
        num_accepted_tokens = num_draft_tokens if self.mtp_drafter_available else 0
        num_verify_steps = 1
        return num_draft_tokens, num_accepted_tokens, num_verify_steps


def test_mtp_hybrid_sleep_keeps_drafter_available_for_nonzero_acceptance(monkeypatch):
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)

    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = SimpleNamespace(mtp=SimpleNamespace(enable=True, enable_rollout=True))
    server.model_config = SimpleNamespace(lora_rank=0, lora={})
    server.engine = _FakeMtpEngine()

    asyncio.run(server._sleep_hybrid())
    server.engine.sync_actor_weights()
    drafts, accepts, verifies = server.engine.generate_spec_decode_stats()
    metrics = compute_spec_decode_metrics(
        spec_drafts=np.array([drafts]),
        spec_accepts=np.array([accepts]),
        spec_verifies=np.array([verifies]),
    )

    assert metrics["rollout/spec_accept_rate"] > 0.0
    assert metrics["rollout/spec_accept_length"] > 1.0


def test_refresh_kv_zero_meta_rebuilds_runner_addresses():
    calls = []
    worker = SimpleNamespace(model_runner=SimpleNamespace(_init_kv_zero_meta=lambda: calls.append(True)))

    vllm_utils.vLLMColocateWorkerExtension.refresh_kv_zero_meta(worker)

    assert calls == [True]


def test_torch_kv_block_zeroer_zeros_logical_blocks(monkeypatch):
    class FullAttentionSpec:
        block_size = 4
        num_kv_heads = 1
        head_size = 2
        cache_dtype = "float32"

    monkeypatch.setattr("vllm.v1.kv_cache_interface.FullAttentionSpec", FullAttentionSpec)
    kv_cache = np.ones((2, 6, 2, 1, 2), dtype=np.float32)
    kv_tensor = torch.from_numpy(kv_cache)
    group = SimpleNamespace(
        kv_cache_spec=FullAttentionSpec(),
        kv_cache_group_id=0,
        backend=SimpleNamespace(get_kv_cache_block_dim=lambda *args, **kwargs: 1),
        layer_names=["layer"],
    )
    class Zeroer:
        pass

    zeroer = Zeroer()

    def init_runner_meta():
        zeroer.init_meta(
            [group],
            [2],
            "float32",
            set(),
            {"layer": SimpleNamespace(kv_cache=[kv_tensor])},
        )

    worker = SimpleNamespace(model_runner=SimpleNamespace(_kv_block_zeroer=zeroer, _init_kv_zero_meta=init_runner_meta))
    vllm_utils.vLLMColocateWorkerExtension.use_torch_kv_block_zeroer(worker)
    zeroer.zero_block_ids([1])

    assert np.all(kv_cache[:, 2:4] == 0)
    assert np.all(kv_cache[:, :2] == 1)
    assert np.all(kv_cache[:, 4:] == 1)

    replacement = Zeroer()
    replacement._verl_cache_slices = [(kv_tensor, 1, 2)]
    replacement.zero_block_ids([0])
    assert np.all(kv_cache[:, :2] == 0)

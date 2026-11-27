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
import gc
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl.utils import transferqueue_utils as tqu


async def _noop():
    return None


def _open_fd_count() -> int:
    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("requires Linux /proc file descriptor accounting")
    return len(os.listdir("/proc/self/fd"))


def test_temp_event_loop_releases_file_descriptors():
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    tqu._shutdown_async_bridge_runtime()
    try:
        tqu._run_async_in_temp_loop(_noop)
        before = _open_fd_count()
        for _ in range(32):
            tqu._run_async_in_temp_loop(_noop)
        leaked = _open_fd_count() - before
    finally:
        tqu._shutdown_async_bridge_runtime()
        if gc_was_enabled:
            gc.enable()
        gc.collect()

    assert leaked == 0


def test_async_bridge_loop_reused_between_calls():
    tqu._shutdown_async_bridge_runtime()
    try:
        tqu._run_async_in_temp_loop(_noop)
        first_thread = tqu._ASYNC_BRIDGE_THREAD
        first_loop = tqu._ASYNC_BRIDGE_LOOP

        assert first_thread is not None
        assert first_loop is not None
        assert first_thread.is_alive()
        assert not first_loop.is_closed()

        tqu._run_async_in_temp_loop(_noop)
        assert tqu._ASYNC_BRIDGE_THREAD is first_thread
        assert tqu._ASYNC_BRIDGE_LOOP is first_loop
    finally:
        tqu._shutdown_async_bridge_runtime()


def _metadata(size, fields, ready=True):
    return SimpleNamespace(
        size=size,
        field_names=fields,
        production_status=np.ones(size, dtype=np.int8) if ready else np.zeros(size, dtype=np.int8),
    )


@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize(
    "first",
    [
        _metadata(2, []),
        _metadata(1, ["loss_mask", "input_ids"]),
        _metadata(2, ["input_ids"]),
        _metadata(2, ["loss_mask", "input_ids"], ready=False),
    ],
)
def test_verified_write_retries_unconfirmed_data_before_publishing_tags(monkeypatch, first, synchronous):
    calls = []

    async def put(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(tags=None)

    client = SimpleNamespace(
        async_kv_retrieve_meta=AsyncMock(
            side_effect=[
                first,
                _metadata(2, ["loss_mask", "input_ids"]),
            ]
        )
    )
    monkeypatch.setattr(tqu, "tq", SimpleNamespace(async_kv_batch_put=put, get_client=lambda: client))
    fields = TensorDict({"loss_mask": torch.ones(2, 3), "input_ids": torch.ones(2, 3)}, batch_size=[2])
    tags = [{"status": "success"}] * 2
    kwargs = dict(keys=["a", "b"], partition_id="train", fields=fields, tags=tags)
    if synchronous:
        result = tqu.kv_batch_put_verified(**kwargs)
    else:
        result = asyncio.run(tqu.async_kv_batch_put_verified(**kwargs))
    assert len(calls) == 3
    assert calls[0]["fields"] is calls[1]["fields"] is fields
    assert "tags" not in calls[0] and "tags" not in calls[1]
    assert "fields" not in calls[2] and calls[2]["tags"] is tags
    assert result.tags is tags


def test_verified_write_does_not_publish_success_after_retry_exhaustion(monkeypatch):
    put = AsyncMock(return_value=SimpleNamespace(tags=None))
    client = SimpleNamespace(async_kv_retrieve_meta=AsyncMock(return_value=_metadata(2, [])))
    monkeypatch.setattr(tqu, "tq", SimpleNamespace(async_kv_batch_put=put, get_client=lambda: client))
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        asyncio.run(
            tqu.async_kv_batch_put_verified(
                keys=["a", "b"],
                partition_id="train",
                fields=TensorDict({"loss_mask": torch.ones(2, 3)}, batch_size=[2]),
                tags=[{"status": "success"}] * 2,
            )
        )
    assert put.await_count == 3
    assert all("tags" not in call.kwargs for call in put.await_args_list)

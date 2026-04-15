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
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import AsyncGenerator, Generator

import ray
import torch
import torch.distributed as dist
import zmq

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry, TensorMeta
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _set_cuda_device():
    """Set CUDA device based on Ray GPU assignment when CUDA_VISIBLE_DEVICES is not set."""
    gpu_ids = ray.get_gpu_ids()
    if gpu_ids:
        gpu_id = int(gpu_ids[0])
        torch.cuda.set_device(gpu_id)
        print(f"[NCCLCkptEngine] _set_cuda_device: ray_gpu_ids={gpu_ids}, set device={gpu_id}", flush=True)
        return gpu_id
    cur = torch.cuda.current_device()
    print(f"[NCCLCkptEngine] _set_cuda_device: no ray_gpu_ids, current_device={cur}", flush=True)
    return cur


@dataclass
class MasterMetadata:
    zmq_ip: str
    zmq_port: int
    nccl_store_port: int = 0


class BroadcastOperation:
    """Async broadcast operation with NCCL in separate thread.

    Args:
        rank (int): The rank of the current process.
        pg (dist.ProcessGroup): The torch.distributed process group for NCCL.
        bucket (torch.Tensor): The tensor to broadcast.
        metadata (dict[str, TensorMeta]): The metadata of the tensor.
        socket (zmq.Socket): The zeromq socket to communicate with master.
        topic (str): The topic to subscribe.
    """

    def __init__(
        self,
        rank: int,
        pg: dist.ProcessGroup,
        bucket: torch.Tensor,
        metadata: dict[str, TensorMeta],
        socket: zmq.Socket,
        topic: str,
    ) -> None:
        self.rank = rank
        self.pg = pg
        self.bucket = bucket
        self.metadata = metadata
        self.socket = socket
        self.topic = topic

        loop = asyncio.get_running_loop()
        self._task = loop.run_in_executor(None, self._run)

    def _run(self):
        # Set CUDA device for this thread (thread pool threads default to device 0)
        torch.cuda.set_device(self.bucket.device)

        # broadcast tensor meta via zeromq PUB/SUB
        if self.rank == 0:
            self.socket.send_string(self.topic, flags=zmq.SNDMORE)
            self.socket.send_pyobj(self.metadata)
            print(f"[BroadcastOp] _run: rank=0, sent metadata via zmq, bucket device={self.bucket.device}", flush=True)
        else:
            self.socket.recv_string()
            self.metadata = self.socket.recv_pyobj()
            print(f"[BroadcastOp] _run: rank={self.rank}, received metadata via zmq, bucket device={self.bucket.device}", flush=True)

        # broadcast tensor via torch.distributed NCCL (use low-level PG API)
        print(f"[BroadcastOp] _run: rank={self.rank}, starting NCCL broadcast, current_device={torch.cuda.current_device()}", flush=True)
        opts = dist.BroadcastOptions()
        opts.rootRank = 0
        opts.rootTensor = 0
        work = self.pg.broadcast([self.bucket], opts)
        work.wait()
        print(f"[BroadcastOp] _run: rank={self.rank}, NCCL broadcast DONE", flush=True)

    async def wait_for_complete(self) -> dict[str, TensorMeta]:
        """Wait for the broadcast operation to complete.

        Returns:
            dict[str, TensorMeta]: The bucket meta after broadcast.
        """
        await self._task
        return self.metadata


@CheckpointEngineRegistry.register("nccl")
class NCCLCheckpointEngine(CheckpointEngine):
    """NCCL checkpoint engine using torch.distributed for collective communication.

    Args:
        bucket_size (int): Bucket size in bytes to transfer multiple weights at one time. Note that we use
            two buffer to send and recv weights at same time, so the device memory overhead is 2 * bucket_size.
        group_name (str): The name of the NCCL process group. Defaults to "default".
        rebuild_group (bool): Whether to rebuild the NCCL process group in each update. Defaults to False.
        is_master (bool): Whether the current process is the master process. Defaults to False.
        rollout_dtype (torch.dtype): The dtype of the weights received from rollout workers. Defaults to torch.bfloat16.
    """

    def __init__(
        self,
        bucket_size: int,
        group_name: str = "default",
        rebuild_group: bool = False,
        is_master: bool = False,
        rollout_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.bucket_size = bucket_size
        self.group_name = group_name
        self.rebuild_group = rebuild_group
        self.rollout_dtype = rollout_dtype
        self.pg = None
        self._store = None

        # start zeromq server for broadcasting bucket tensor metadata
        self.is_master = is_master
        self.topic = "bucket_metadata"
        if self.is_master:
            self._start_zmq_server()

    def prepare(self) -> MasterMetadata:
        # Only reset CUDA device for master; non-master FSDP workers already have correct device set
        if self.is_master:
            dev = _set_cuda_device()
            print(f"[NCCLCkptEngine] prepare: is_master=True, device={dev}", flush=True)
        else:
            dev = torch.cuda.current_device()
            print(f"[NCCLCkptEngine] prepare: is_master=False, current_device={dev}", flush=True)
        self.send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda")
        self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda")

        if self.is_master:
            self.nccl_store_port, _ = get_free_port(self.ip)
            print(f"[NCCLCkptEngine] prepare: master ip={self.ip}, zmq_port={self.listen_port}, nccl_store_port={self.nccl_store_port}", flush=True)
            return MasterMetadata(zmq_ip=self.ip, zmq_port=self.listen_port, nccl_store_port=self.nccl_store_port)
        return None

    def finalize(self):
        """Destroy the NCCL process group if rebuild_group is True."""
        if self.rebuild_group:
            if self.rank >= 0 and self.pg is not None:
                dist.destroy_process_group(self.pg)
                self.pg = None
                self._store = None
            self.rank = None
            self.world_size = None

        self.send_buf = None
        self.recv_buf = None

        torch.cuda.empty_cache()

    @classmethod
    def build_topology(cls, trainer_world_size: int, rollout_world_size: int, metadata: list[dict]):
        trainer_kwargs = {
            "rank": [0] + [-1] * (trainer_world_size - 1),
            "world_size": [rollout_world_size + 1] * trainer_world_size,
            "master_metadata": [metadata[0]] * trainer_world_size,
        }
        rollout_kwargs = {
            "rank": list(range(1, rollout_world_size + 1)),
            "world_size": [rollout_world_size + 1] * rollout_world_size,
            "master_metadata": [metadata[0]] * rollout_world_size,
        }
        return trainer_kwargs, rollout_kwargs

    def _start_zmq_server(self):
        self.ip = ray.util.get_node_ip_address().strip("[]")
        self.listen_port, _ = get_free_port(self.ip)

        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        if is_valid_ipv6_address(self.ip):
            address = f"tcp://[{self.ip}]:{self.listen_port}"
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{self.ip}:{self.listen_port}"

        self.socket.bind(address)

    def _connect_zmq_client(self, metadata: MasterMetadata):
        assert not self.is_master, "Master process should not connect to other processes."
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        if is_valid_ipv6_address(metadata.zmq_ip):
            address = f"tcp://[{metadata.zmq_ip}]:{metadata.zmq_port}"
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{metadata.zmq_ip}:{metadata.zmq_port}"

        self.socket.connect(address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata):
        """Initialize the NCCL process group using torch.distributed.

        Args:
            rank (int): The rank of the current process.
            world_size (int): The total number of processes.
            master_metadata (MasterMetadata): Metadata from the master process.
        """
        # For trainer workers other than rank 0, their rank should be -1.
        if rank < 0:
            self.rank = rank
            self.world_size = world_size
            print(f"[NCCLCkptEngine] init_process_group: rank={rank}, skipping (non-participating)", flush=True)
            return

        dev = _set_cuda_device()
        print(f"[NCCLCkptEngine] init_process_group: rank={rank}, world_size={world_size}, device={dev}, master_metadata={master_metadata}", flush=True)

        if self.rebuild_group or self.pg is None:
            # Create a TCPStore for NCCL process group rendezvous
            store_host = master_metadata.zmq_ip
            store_port = master_metadata.nccl_store_port
            is_master_store = rank == 0
            print(f"[NCCLCkptEngine] init_process_group: creating TCPStore host={store_host}:{store_port}, is_master={is_master_store}", flush=True)
            self._store = dist.TCPStore(
                host_name=store_host,
                port=store_port,
                world_size=world_size,
                is_master=is_master_store,
                timeout=timedelta(seconds=120),
            )
            print(f"[NCCLCkptEngine] init_process_group: TCPStore created, creating ProcessGroupNCCL...", flush=True)
            prefix_store = dist.PrefixStore(f"ckpt_{self.group_name}", self._store)
            pg_opts = dist.ProcessGroupNCCL.Options()
            pg_opts._timeout = timedelta(seconds=120)
            self.pg = dist.ProcessGroupNCCL(prefix_store, rank, world_size, pg_opts)
            print(f"[NCCLCkptEngine] init_process_group: ProcessGroupNCCL created for rank={rank}", flush=True)
            self.rank = rank
            self.world_size = world_size
        else:
            assert self.rank == rank, f"rank {rank} is not equal to self.rank {self.rank}"
            assert self.world_size == world_size, (
                f"world_size {world_size} is not equal to self.world_size {self.world_size}"
            )

        if self.rank > 0:
            self._connect_zmq_client(master_metadata)

        # barrier via broadcast on the same device as send/recv buffers
        # This ensures the NCCL comm is created for the buffer device,
        # so subsequent broadcasts in thread pool threads reuse the same comm.
        print(f"[NCCLCkptEngine] init_process_group: rank={self.rank} starting barrier (broadcast on {self.send_buf.device})...", flush=True)
        dummy = torch.zeros(1, dtype=torch.uint8, device=self.send_buf.device)
        barrier_bcast_opts = dist.BroadcastOptions()
        barrier_bcast_opts.rootRank = 0
        barrier_bcast_opts.rootTensor = 0
        work = self.pg.broadcast([dummy], barrier_bcast_opts)
        work.wait()
        print(f"[NCCLCkptEngine] init_process_group: rank={self.rank} barrier DONE", flush=True)

        logger.info(f"init_process_group rank: {self.rank}, world_size: {self.world_size}")

    @torch.no_grad()
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        assert self.rank <= 0, "Trainer workers other than rank 0 should not send weights."
        print(f"[NCCLCkptEngine] send_weights: rank={self.rank}, is_master={self.is_master}, device={torch.cuda.current_device()}", flush=True)

        # For trainer rank other than 0, consume weights without sending.
        if self.rank < 0:
            for name, weight in weights:
                pass
            return

        send_buf, recv_buf = self.send_buf, self.recv_buf
        broadcast_op = None

        start_time = time.time()
        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0
        for name, weight in weights:
            # fill the tensor bucket
            if offset + weight.nbytes > self.bucket_size:
                torch.cuda.synchronize()

                # wait previous broadcast op finish
                if broadcast_op is not None:
                    await broadcast_op.wait_for_complete()

                broadcast_op = BroadcastOperation(
                    rank=self.rank,
                    pg=self.pg,
                    bucket=send_buf,
                    metadata={"bucket_meta": bucket_meta, "is_last": False},
                    socket=self.socket,
                    topic=self.topic,
                )

                # swap send_buf and recv_buf
                send_buf, recv_buf = recv_buf, send_buf
                bucket_meta = {}
                offset = 0

            assert offset + weight.nbytes <= self.bucket_size, (
                f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
            )

            bucket_meta[name] = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": offset,
            }
            send_buf[offset : offset + weight.nbytes].copy_(weight.view(-1).view(torch.uint8))
            offset += weight.nbytes

        # broadcast last bucket
        torch.cuda.synchronize()
        if broadcast_op is not None:
            await broadcast_op.wait_for_complete()

        broadcast_op = BroadcastOperation(
            rank=self.rank,
            pg=self.pg,
            bucket=send_buf,
            metadata={"bucket_meta": bucket_meta, "is_last": True},
            socket=self.socket,
            topic=self.topic,
        )
        await broadcast_op.wait_for_complete()
        logger.info(f"Rank {self.rank} send weights done, time cost: {time.time() - start_time:.2f}s")

    @torch.no_grad()
    async def receive_weights(self) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive the weights of the model.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        assert self.rank > 0, "Rank 0 should not receive weights."
        print(f"[NCCLCkptEngine] receive_weights: rank={self.rank}, device={torch.cuda.current_device()}, recv_buf device={self.recv_buf.device}", flush=True)
        send_buf, recv_buf = self.send_buf, self.recv_buf
        total_bytes, total_params = 0, 0

        # receive first bucket
        start_time = time.time()
        broadcast_op = BroadcastOperation(
            rank=self.rank,
            pg=self.pg,
            bucket=recv_buf,
            metadata=None,
            socket=self.socket,
            topic=self.topic,
        )
        metadata = await broadcast_op.wait_for_complete()
        total_bytes += self.bucket_size
        total_params += len(metadata["bucket_meta"])

        # swap send_buf and recv_buf
        send_buf, recv_buf = recv_buf, send_buf
        while not metadata["is_last"]:
            # 1. receive next bucket
            broadcast_op = BroadcastOperation(
                rank=self.rank,
                pg=self.pg,
                bucket=recv_buf,
                metadata=None,
                socket=self.socket,
                topic=self.topic,
            )

            # 2. yield tensor from send_buf
            for name, meta in metadata["bucket_meta"].items():
                dtype, shape = meta["dtype"], meta["shape"]
                size = dtype.itemsize * shape.numel()
                tensor = send_buf[meta["offset"] : meta["offset"] + size].view(dtype=dtype).view(shape)
                yield name, tensor

            # 3. wait for next bucket broadcast finish
            metadata = await broadcast_op.wait_for_complete()
            total_bytes += self.bucket_size
            total_params += len(metadata["bucket_meta"])

            # 4. swap send_buf and recv_buf
            torch.cuda.synchronize()  # sync non-blocking copy
            send_buf, recv_buf = recv_buf, send_buf

        # yield tensor from send_buf
        for name, meta in metadata["bucket_meta"].items():
            dtype, shape = meta["dtype"], meta["shape"]
            size = dtype.itemsize * shape.numel()
            tensor = send_buf[meta["offset"] : meta["offset"] + size].view(dtype=dtype).view(shape)
            yield name, tensor

        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        logger.info(
            f"Rank {self.rank} receive weights done, total_params: {total_params}, "
            f"time cost: {time_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s"
        )

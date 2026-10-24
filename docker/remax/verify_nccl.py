"""torchrun --standalone --nproc-per-node=8 docker/remax/verify_nccl.py"""
import os

import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))
value = torch.tensor([dist.get_rank() + 1.0], device=f"cuda:{rank}")
dist.all_reduce(value)
expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
assert value.item() == expected, (rank, value, expected)
if dist.get_rank() == 0:
    print(f"NCCL all-reduce passed across {dist.get_world_size()} GPUs; sum={expected}", flush=True)
dist.destroy_process_group()

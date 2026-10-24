"""Queued diagnostic: run only after verl-n2-i10 is verified free.

This uses tiny FSDP weights, not the ASR model. A pass narrows the failure
context; it does not prove full vLLM initialization or training recovery.
Run separate fresh processes with and without --all-visible and --hijack.
"""

import argparse
import json
import os
import socket
from datetime import timedelta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-visible", action="store_true")
    parser.add_argument("--hijack", action="store_true")
    args = parser.parse_args()
    import ray

    ray.init(address="auto")
    nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0)]
    if len(nodes) != 2 or any(n["Resources"]["GPU"] != 8 for n in nodes):
        raise RuntimeError("Expected exactly two eight-GPU nodes; do not use another pool")
    if not socket.gethostname().startswith("verl-n2-i10-"):
        raise RuntimeError("Probe is restricted to verl-n2-i10")
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    @ray.remote(num_gpus=1, num_cpus=1)
    class Probe:
        def address(self):
            if not socket.gethostname().startswith("verl-n2-i10-"):
                raise RuntimeError("Actor scheduled outside verl-n2-i10")
            return ray.util.get_node_ip_address()

        def run(self, rank, world, master):
            import torch
            import torch.distributed as dist
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

            device = int(ray.get_gpu_ids()[0]) if args.all_visible else 0
            torch.cuda.set_device(device)
            if args.hijack:
                from verl.utils.vllm import VLLMHijack

                VLLMHijack.hijack()
            dist.init_process_group(
                "nccl", init_method=f"tcp://{master}:29647", rank=rank,
                world_size=world, timeout=timedelta(minutes=5),
            )

            def sample(phase):
                stream = torch.cuda.Stream(device=device)
                with torch.cuda.stream(stream):
                    logits = torch.randn(8, 151936, device="cuda")
                    top_k = torch.full((8,), 151935, device="cuda", dtype=torch.int32)
                    top_p = torch.full((8,), 0.9, device="cuda")
                    tokens, _ = TopKTopPSampler()(logits, {}, top_k, top_p)
                stream.synchronize()
                assert tokens.shape == (8,)
                print(json.dumps(dict(phase=phase, rank=rank, device=device,
                                      host=socket.gethostname(), stream=stream.cuda_stream,
                                      visible=os.getenv("CUDA_VISIBLE_DEVICES"))), flush=True)

            try:
                sample("ray_nccl")
                model = FSDP(torch.nn.Linear(128, 128).cuda(), device_id=device)
                model(torch.randn(8, 128, device="cuda")).square().mean().backward()
                torch.cuda.synchronize()
                sample("after_fsdp_backward")
                # Match the repaired runtime's TileLang/FLA import ordering.
                import tilelang  # noqa: F401
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401

                sample("after_tilelang_fla_import")
                dist.barrier()
                return dict(rank=rank, status="passed")
            finally:
                dist.destroy_process_group()

    env = {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"} if args.all_visible else {}
    actors = [Probe.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False),
        runtime_env={"env_vars": env},
    ).remote() for n in nodes for _ in range(8)]
    try:
        addresses = ray.get([actor.address.remote() for actor in actors])
        master = addresses[0]
        print(ray.get([a.run.remote(i, 16, master) for i, a in enumerate(actors)]))
    finally:
        for actor in actors:
            ray.kill(actor)
        ray.shutdown()


if __name__ == "__main__":
    main()

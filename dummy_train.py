# %%
import subprocess
import fire
import time
import os
from mpi4py import MPI
import socket
import ray

RAY_PORT = 6379


def get_head_address(comm, rank, port=RAY_PORT):
    """Get the Ray head address. Rank 0 creates it, others receive via broadcast."""
    if rank == 0:
        hostname = socket.gethostname()
        host_ip = socket.gethostbyname(hostname)
        head_addr = f"{host_ip}:{port}"
    else:
        head_addr = None

    # Everyone learns the head address
    head_addr = comm.bcast(head_addr, root=0)
    return head_addr


def init_ray(port=RAY_PORT):
    """Initialize Ray with MPI. Rank 0 starts as head, others connect as workers."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    print(f"Initializing Ray for rank {rank} out of {world_size}...")

    head_addr = get_head_address(comm, rank, port)
    print(f"Rank {rank} got head address: {head_addr}")

    # Rank 0: head
    if rank == 0:
        print(f"Starting Ray head at rank {rank} with address {head_addr}...")
        ray.init()

    print(f"Rank {rank} waiting for head ray...")
    comm.barrier()
    if rank != 0 and not ray.is_initialized():
        print(f"Connecting rank[{rank}] to Ray head at {head_addr}...")
        ray.init(address=f"ray://{head_addr}")

    print(f"Rank {rank} connected to Ray head at {head_addr}")

    if ray.is_initialized():
        info = ray.cluster_resources()
        print("Ray Cluster Resources:", info)
        nodes = [node for node in ray.nodes() if node["Alive"]]
        nodes = sorted(nodes, key=lambda x: x["NodeManagerHostname"])
        print("Found nodes:")
        for i, node in enumerate(nodes):
            print(f" - {i}: {node['NodeManagerHostname']}[{node['NodeManagerAddress']}]")
    else:
        print(f"Ray is not initialized on rank[{rank}].")


def print_env():
    """Print environment variables."""
    print("Environment Variables:")
    for key, value in os.environ.items():
        print(f"{key}={value}")


def print_pip():
    """Print installed pip packages."""
    print("Installed pip packages:")
    try:
        import pkg_resources

        installed_packages = pkg_resources.working_set
        for dist in installed_packages:
            print(f"{dist.project_name}=={dist.version}")
    except ImportError:
        print("using pip list command")
        subprocess.run(["pip", "list"])


def dummy_training_loop(n=10):
    """A dummy training loop that runs indefinitely."""
    print("Starting dummy training with endless loop...")
    print_env()
    print_pip()
    init_ray()
    print("Starting training loop...")
    step = 0
    while step < n:
        step += 1
        print(f"Training step {step}")
        time.sleep(1)


if __name__ == "__main__":
    fire.Fire(dummy_training_loop)


# %%

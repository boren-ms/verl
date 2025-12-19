# %%
import subprocess
import fire
import time
import os
from mpi4py import MPI
import socket
import ray

def get_host_ip():
    """Get the host IP address."""
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    return host_ip


def init_ray(port=6379):
    """Initialize Ray with MPI. Rank 0 starts as head, others connect as workers."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    print(f"Initializing Ray for rank {rank} out of {world_size}...")

    head_ip = None
    # Rank 0: head
    if rank == 0:
        head_ip = get_host_ip()
        print(f"[{rank}] Starting Ray head with address {head_ip}:{port}...")
        cmd = ["ray", "start", "--head", f"--node-ip-address={head_ip}", f"--port={port}"]
        print("cmd:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    comm.bcast(head_ip, root=0)
    print(f"[{rank}] obtained head IP: {head_ip}")
    if not ray.is_initialized() and rank != 0:
        print(f"[{rank}] Connecting to Ray head at {head_ip}:{port}...")
        cmd = ["ray", "start", f"--address={head_ip}:{port}"]
        print("cmd:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    if ray.is_initialized():
        info = ray.cluster_resources()
        print(f"[{rank}] Ray is initialized.")
        print(f"[{rank}] Cluster Info:", info)
        nodes = [node for node in ray.nodes() if node["Alive"]]
        nodes = sorted(nodes, key=lambda x: x["NodeManagerHostname"])
        print(f"[{rank}] Found nodes:")
        for i, node in enumerate(nodes):
            print(f" - {i}: {node['NodeManagerHostname']}[{node['NodeManagerAddress']}]")
    else:
        print(f"[{rank}] Ray is not initialized .")


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

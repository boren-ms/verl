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


def run_cmd(cmd):
    """Run a shell command and return its output."""
    print(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    print(result.stdout.strip())
    return result.returncode, result.stdout.strip()


def init_ray(port=6379):
    """Initialize Ray with MPI. Rank 0 starts as head, others connect as workers."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    hostname = socket.gethostname()

    print(f"[{rank}] Host name: {hostname}")
    print(f"Initializing Ray for rank {rank} out of {world_size}...")
    if rank == 0:
        print(f"[{rank}] Starting Ray head on {hostname}:{port}...")
        run_cmd(f"ray start --head --port={port}")
        run_cmd("ray status")
    else:
        head_host = "node-0"
        print(f"[{rank}] Connecting to Ray at {head_host}:{port}...")
        run_cmd(f"ray start --address={head_host}:{port}")
        run_cmd("ray status")

    comm.barrier()  # Ensure all ranks have started Ray
    ray.init(address="auto")
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

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


def print_ray_info(rank_str):
    ray.init(address="auto")
    info = ray.cluster_resources()
    print(f"[{rank_str}] Cluster Info:", info)
    nodes = [node for node in ray.nodes() if node["Alive"]]
    nodes = sorted(nodes, key=lambda x: x["NodeManagerHostname"])
    print(f"[{rank_str}] Found nodes:")
    for i, node in enumerate(nodes):
        print(f" - {i}: {node['NodeManagerHostname']}[{node['NodeManagerAddress']}]")


def init_ray(port=6379):
    """Initialize Ray with MPI. Rank 0 starts as head, others connect as workers."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    rank_str = f"{rank}/{world_size}"
    hostname = socket.gethostname()

    print(f"[{rank_str}] Host name: {hostname}")
    if rank == 0:
        print(f"[{rank_str}] Starting Ray head on {hostname}:{port}...")
        run_cmd(f"ray start --head --port={port} --disable-usage-stats")
    comm.barrier()  # wait for head to start
    if rank != 0:
        head_host = "node-0"
        print(f"[{rank_str}] Connecting to Ray at {head_host}:{port}...")
        run_cmd(f"ray start --address={head_host}:{port}")
    comm.barrier()  # Ensure all ranks have started Ray
    run_cmd("ray status")
    print_ray_info(rank_str)


def print_accelerate_info():
    """Print information about the Accelerate library and devices."""
    from accelerate import PartialState

    state = PartialState()
    print("Accelerate State:", state)


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


def dummy_train(n=10):
    """A dummy training function."""
    print("Dummy training function called.")
    import torch

    # Consume GPU memory and compute matrix multiply
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create large matrices to consume GPU memory
    size = 8192
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Perform matrix multiplication
    print(f"Computing matrix multiplication of size {size}x{size}...")
    for _i in range(n):
        c = torch.matmul(a, b)
    print(f"Result shape: {c.shape}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

    return c


def dummy_training_loop(n=-1):
    """A dummy training loop that runs indefinitely."""
    print("Starting dummy training with endless loop...")
    print_env()
    print_pip()
    init_ray()
    print_accelerate_info()
    print("Starting training loop...")
    step = 0
    while step < n or n < 0:
        step += 1
        print(f"Training step {step}")
        dummy_train(n=5)
        time.sleep(600)  # Sleep for 10 minutes


if __name__ == "__main__":
    fire.Fire(dummy_training_loop)


# %%

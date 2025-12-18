import fire
from mpi4py import MPI
import socket
import ray

RAY_PORT = 6379


def get_head_address(comm, rank, port=RAY_PORT):
    """Get the Ray head address. Rank 0 creates it, others receive via broadcast."""
    if rank == 0:
        host = socket.gethostname()
        head_addr = f"{host}:{port}"
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

    # Rank 0: head
    if rank == 0:
        print("Starting Ray head...")
        ray.init()
    elif not ray.is_initialized():  # same node
        print(f"Connecting rank[{rank}] to Ray head at {head_addr}...")
        ray.init(address=f"ray://{head_addr}")
    else:
        print(f"Ray already initialized for rank[{rank}]")

    print(f"Rank {rank} connected to Ray head at {head_addr}")


if __name__ == "__main__":
    fire.Fire(init_ray)

#! /usr/bin/env python3
import os
import ray
import fire
from recipe.phimm.utils.shared import run_cmd, to_list, is_package_version


@ray.remote
def prepare_env(forced=False):
    """Prepare the environment on each node by installing necessary packages."""
    hostname = os.uname().nodename
    print(f"[{hostname}] Preparing environment...")
    required = [
        "torch==2.10.0",
        "ray==2.46.0",
        "transformers==4.55.4",
        "vllm==0.11.0",
        "flash-attn==2.8.3",
    ]
    if all(is_package_version(*pkg.split("==")) for pkg in required) and not forced:
        print(f"Required packages already installed on {hostname}, skipping installation.")
        return
    run_cmd("bash quick_install.sh")
    print(f"[{hostname}] Environment preparation completed.")


@ray.remote
def prepare_data(forced=False):
    """Prepare the data on each node by downloading and processing it."""
    hostname = os.uname().nodename
    print(f"[{hostname}] Preparing data...")
    print("Doing nothing for data preparation.")
    print(f"[{hostname}] Data preparation completed.")


class RayNode:
    def __init__(self, indexs=None):
        """Initialize the RayHelper with the specified nodes."""
        print("Connecting to Ray cluster...")
        ray.init(address="auto")  # Connect to the running cluster
        print("Connected to Ray cluster.")
        nodes = [node for node in ray.nodes() if node["Alive"]]
        nodes = sorted(nodes, key=lambda x: x["NodeManagerHostname"])
        print("Found nodes:")
        for i, node in enumerate(nodes):
            print(f" - {i}: {node['NodeManagerHostname']}[{node['NodeManagerAddress']}]")

        self.indexs = to_list(indexs) if indexs is not None else list(range(len(nodes)))
        self.nodes = [nodes[i] for i in self.indexs]
        print(
            f"Initialized RayHelper with {len(self.nodes)} nodes: {[node['NodeManagerHostname'] for node in self.nodes]}"
        )

    def async_run(self, func, *args, waiting=True, **kwargs):
        # Launch one task per node, each pinned to a specific node
        results = []
        for node in self.nodes:
            node_ip = node["NodeManagerAddress"]
            # Use custom resource label to ensure the function runs on this node
            # Each node has a resource label 'node:<ip>'
            node_label = f"node:{node_ip}"
            result = func.options(resources={node_label: 0.01}).remote(*args, **kwargs)
            results.append(result)
        return results

    def prepare(self, forced=False):
        """Prepare the environment, data, and output on all Ray nodes."""
        results = []
        results += self.async_run(prepare_env, forced=forced)
        results += self.async_run(prepare_data, forced=forced)
        results = ray.get(results)
        print("Preparation completed.")

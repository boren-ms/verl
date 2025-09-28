#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import ray
import os
from pathlib import Path
import fire
from ray_tool import (
    RayNode,
    update_envs,
    prepare_local_checkpoint,
    get_local_path,
    get_remote_path,
    search_models,
)

DEFAULT_EVAL_CONFIG = "eval_conf/eval_vllm_dev_tiny.yaml"


@ray.remote
def launch_evaluation(model_path, config_file=None, ray_node=None):
    """Launch evaluation using the specified YAML config file."""
    cur_dir = Path(__file__).parent
    os.chdir(cur_dir)
    print(f"Working Dir: {os.getcwd()}")
    config_file = config_file or DEFAULT_EVAL_CONFIG
    config_file = Path(config_file).absolute()
    update_envs(config_file)

    model_path = Path(model_path).absolute()

    print(f"Config file: {config_file}")
    print(f"Model path: {model_path}")
    assert model_path.exists(), f"Model {model_path} does not exist."
    if not isinstance(ray_node, RayNode):
        ray_node = RayNode(ray_node)

    head = ray_node.indexs[0]
    rank = int(os.environ.get("RCALL_INSTANCE_INDEX", "0")) - head
    rank_size = ray_node.num_nodes
    num_gpu = int(os.environ.get("RCALL_NUM_GPU", "8"))
    main_process_ip = ray_node.hostname()
    main_process_port = 12345
    script_path = cur_dir / "trl/scripts/eval_bias.py"
    cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        str(num_gpu * rank_size),
        "--num_machines",
        str(rank_size),
        "--machine_rank",
        str(rank),
        "--main_process_ip",
        str(main_process_ip),
        "--main_process_port",
        str(main_process_port),
        str(script_path),
        "--config",
        str(config_file),
        "--model_path",
        str(model_path),
    ]

    rcall_logdir = os.environ.get("RCALL_LOGDIR", os.path.expanduser("~/logs"))
    os.makedirs(rcall_logdir, exist_ok=True)
    rank_log_file = os.path.join(rcall_logdir, f"{config_file.stem}_{model_path.stem}_{rank}.log")
    print(f"Logging to {rank_log_file}")
    with open(rank_log_file, "w") as logf:
        logf.write(f"Running {' '.join(cmd)}\n")
    # Optionally, printenv could be logged here

    with open(rank_log_file, "a") as logf:
        process = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)


def evaluate_model(remote_model_dir=None, local_model_dir=None, config=None, ray_node=None):
    """Evaluate the model using the specified configuration."""

    if remote_model_dir:
        print(f"Evaluating {remote_model_dir}")
        local_model_dir = get_local_path(remote_model_dir)
        # print("Preparing models from ", remote_model_dir)
        # ray_node.run(prepare_local_checkpoint, local_dir=local_model_dir, remote_dir=remote_model_dir)
    elif local_model_dir:
        print(f"Evaluating {local_model_dir}")
        remote_model_dir = get_remote_path(local_model_dir)
    else:
        raise ValueError("Either remote_model_dir or local_model_dir must be provided.")

    print("Syncing outputs from head to other nodes...")
    # ray_node.broadcast_local_dir(str(local_model_dir))
    ray_node.prepare_local_checkpoint(local_dir=local_model_dir, remote_dir=remote_model_dir)

    print("Watching on ", local_model_dir)
    watcher = ray_node.run_output_watcher(local_model_dir, remote_model_dir, interval=120, sync_all=True)

    print(f"Evaluating {local_model_dir} with config {config}")
    ray_node.run(launch_evaluation, local_model_dir, config, ray_node)
    watcher.flush.remote()
    print("Evaluation completed on ", local_model_dir)


def main(model_path="", config=None, forced=False, nodes=None):
    """Launch the job on all nodes by preparing the environment and data."""
    ray_node = RayNode(nodes)
    print(f"Search models: {model_path if model_path else 'default'}")
    model_paths = search_models(model_path)
    if not model_paths:
        print(f"No models found for {model_path}, existing evaluation.")
        return
    else:
        print(f"Found {len(model_paths)} models")
        for i, model_path in enumerate(model_paths):
            print(f"[{i}] Model: {model_path}")
    ray_node.prepare_all(forced=forced)
    for model_path in model_paths:
        evaluate_model(remote_model_dir=model_path, config=config, ray_node=ray_node)


if __name__ == "__main__":
    """Main entry point for launching the job on a Ray cluster."""
    fire.Fire(main)
    # Example usage: python launch_job.py --config_file="path/to/config.yaml"

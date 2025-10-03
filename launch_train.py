#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import shutil
import subprocess
import os
from pathlib import Path
import ray
import fire
from ray_tool import (
    RayNode,
    get_output_dirs,
    ORNG_USER,
)
from launch_eval import evaluate_model


def dup_config_file(config_file, new_stem):
    """Duplicate the config file with stem suffix."""
    if not new_stem:
        return config_file
    new_config_file = config_file.with_stem(new_stem)
    if new_config_file.exists():
        new_config_file.unlink()
    shutil.copy(config_file, new_config_file)
    return new_config_file


def get_job_name(config_file, acc_config=None, n_node=1):
    """Get the new config file name with suffixes."""
    config_file = Path(config_file).absolute()
    n_gpu = int(os.environ.get("RCALL_NUM_GPU", "8"))
    parts = [config_file.stem, f"G{n_node}x{n_gpu}"]
    if acc_config:
        parts.append(Path(acc_config).stem)
    return "_".join(parts)


def launch_training(module, config_file, output_dir):
    """Launch training using the specified YAML config file."""
    config_file = Path(config_file).absolute()
    config_name = config_file.stem
    config_path = config_file.parent
    data_path = ORNG_USER.data_path
    os.chdir(str(Path(__file__).parent))
    print(f"Working Dir: {os.getcwd()}")
    print(f"Config file: {config_file}")
    print(f"Data Dir: {data_path}")
    print(f"Output Dir: {output_dir}")
    cmd = [
        "/root/.pyenv/versions/3.12.9/bin/python3",
        "-m",
        module,
        "--config-name",
        config_name,
        "--config-path",
        str(config_path),
        f"trainer.experiment_name={config_name}",
        f"trainer.default_local_dir={output_dir}",
    ]

    rcall_logdir = os.environ.get("RCALL_LOGDIR", os.path.expanduser("~/logs"))
    os.makedirs(rcall_logdir, exist_ok=True)
    log_file = os.path.join(rcall_logdir, f"{config_name}.log")
    print(f"Logging to {log_file}")
    with open(log_file, "w") as logf:
        logf.write(f"Running {' '.join(cmd)}\n")
        logf.write(f"Working Dir: {os.getcwd()}\n")
        logf.write(f"Output Dir: {output_dir}\n")
        logf.write(f"Data Dir: {data_path}\n")
    # Optionally, printenv could be logged here
    env = os.environ.copy()
    env["DATA_PATH"] = data_path
    with open(log_file, "a") as logf:
        process = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)


def get_task_module(task=None, config_file=None):
    """Return the script path for the given task."""
    assert task or config_file, "Either task or config_file must be provided"
    tasks = {"dapo": "recipe.phimm.main_dapo"}

    if not task and config_file:
        name_parts = Path(config_file).parent.name.split("_")
        name_parts += Path(config_file).stem.split("_")
        task = next((t for t in tasks if t in name_parts), None)
    assert task, "Task must be specified or inferred from config_file"
    return tasks[task]


def main(config_file, task=None, forced=False, seed_name=None, nodes=None):
    """Launch the job on all nodes by preparing the environment and data."""
    task_module = get_task_module(task, config_file)
    print(f"Using script: {task_module}")
    ray_node = RayNode(nodes)

    config_file = Path(config_file).absolute()
    job_name = config_file.stem

    print(f"Training config: {config_file}")
    print(f"Job name: {job_name}")
    output_dir, remote_output_dir = get_output_dirs(job_name)
    remote_seed_dir = remote_output_dir.replace(job_name, seed_name) if seed_name else remote_output_dir
    print("Preparing output on all nodes from seed: ", remote_seed_dir)
    ray_node.prepare_all(local_dir=output_dir, remote_dir=remote_seed_dir, forced=forced)

    print("Starting output watcher on head node...")
    watcher = ray_node.run_output_watcher(output_dir, remote_output_dir, 600)

    print(f"Launching training with {config_file}...")
    launch_training(str(task_module), str(config_file), str(output_dir))
    print("Training completed on all nodes.")

    print("Launching evaluation on all nodes")
    evaluate_model(local_model_dir=output_dir, ray_node=ray_node)
    print("Evaluation completed on all nodes.")

    watcher.flush.remote()
    print("All tasks completed.")


if __name__ == "__main__":
    """Main entry point for launching the job on a Ray cluster."""
    fire.Fire(main)
    # Example usage: python launch_job.py  --config_file="path/to/config.yaml"

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


@ray.remote
def launch_training(script_path, config_file, output_dir):
    """Launch training using the specified YAML config file."""
    config_file = Path(config_file).absolute()
    config_name = config_file.stem
    config_path = config_file.parent
    print(f"Using config file: {config_file}")
    print(f"Output directory: {output_dir}")

    cmd = [
        "python3",
        script_path,
        "--config-name",
        config_name,
        "--config-path",
        str(config_path),
        f"trainer.experiment_name=${config_name}",
        f"trainer.default_local_dir=${output_dir}",
    ]
    env = {"DATA_PATH": ORNG_USER.data_path()}

    rcall_logdir = os.environ.get("RCALL_LOGDIR", os.path.expanduser("~/logs"))
    os.makedirs(rcall_logdir, exist_ok=True)
    log_file = os.path.join(rcall_logdir, f"{config_name}.log")
    print(f"Logging to {log_file}")
    with open(log_file, "w") as logf:
        logf.write(f"Running {' '.join(cmd)}\n")
        logf.write(f"Environment Variables:\n {env}")
    # Optionally, printenv could be logged here
    with open(log_file, "a") as logf:
        process = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)


def get_task_script(task=None, config_file=None):
    """Return the script path for the given task."""
    assert task or config_file, "Either task or config_file must be provided"
    cur_dir = Path(__file__).parent
    tasks = {"dapo": cur_dir / "recipe/dapo/main_dapo.py"}

    if not task and config_file:
        name_parts = Path(config_file).parent.name.split("_")
        name_parts += Path(config_file).stem.split("_")
        task = next((t for t in tasks if t in name_parts), None)
    assert task, "Task must be specified or inferred from config_file"
    script_path = tasks[task]
    assert script_path.exists(), f"Script {script_path} does not exist."
    return script_path


def get_acc_config(name=None):
    """Return the accelerate config file path for the given name."""
    cwd = Path(__file__).parent
    name_dict = {
        "zero1": cwd / "trl/accelerate_configs/zero1.yaml",
        "zero2": cwd / "trl/accelerate_configs/zero2.yaml",
        "zero3": cwd / "trl/accelerate_configs/zero3.yaml",
        "fsdp2": cwd / "trl/accelerate_configs/fsdp2.yaml",
        "multi": cwd / "trl/accelerate_configs/multi_gpu.yaml",
    }
    return name_dict.get(name, None)


def main(config_file, task=None, forced=False, seed_name=None, nodes=None):
    """Launch the job on all nodes by preparing the environment and data."""
    script_path = get_task_script(task, config_file)
    print(f"Using script: {script_path}")
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
    ray_node.run(launch_training, str(script_path), str(config_file), str(output_dir))
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

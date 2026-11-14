"""Prepare every GPU node and submit the installed ASR recipe to Ray."""

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", "--submission-id", dest="name", required=True)
    parser.add_argument("--config", "--config-name", dest="config")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without connecting to Ray")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
        parser.error("--name must contain only letters, numbers, underscores, dots or hyphens")
    return args


def command_for(args):
    module = "recipe.phimm.main_asr_eval" if args.evaluate else "recipe.phimm.main_asr_remax"
    config = args.config or ("eval_openfull_zip" if args.evaluate else "remax_main_base3a_2611r2")
    if "/" in config or "\\" in config:
        raise ValueError("Use an installed config name, not a YAML path")
    command = ["python", "-u", "-m", module, "--config-name", config.removesuffix(".yaml")]
    if not any(item.startswith("trainer.experiment_name=") for item in args.overrides):
        command.append(f"trainer.experiment_name={args.name}")
    return command + args.overrides


def prepare_node(repository, expected_hash):
    """Runs once on each live GPU Ray node, before any training is submitted."""
    import socket

    script = Path(repository) / "plugins/phimm/submit.py"
    if hashlib.sha256(script.read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError(f"Sync this checkout to all nodes before submitting: {socket.gethostname()}")
    result = subprocess.run(
        ["bash", str(Path(repository) / "quick_install.sh")],
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode:
        raise RuntimeError(f"Preparation failed on {socket.gethostname()}:\n{result.stdout}\n{result.stderr}")
    return {"node": socket.gethostname(), "setup": result.stdout}


def main(argv=None):
    args = parse_args(argv)
    command = command_for(args)
    entrypoint = f"cd {shlex.quote(str(REPOSITORY))} && exec {shlex.join(command)}"
    if args.dry_run:
        print(json.dumps({"submission_id": args.name, "entrypoint": entrypoint}, indent=2))
        return

    import ray
    from ray.job_submission import JobSubmissionClient
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(address="auto", log_to_driver=False)
    try:
        address = ray._private.worker.global_worker.node.address_info["webui_url"]
        client = JobSubmissionClient(address="http://" + address)
        if any(job.submission_id == args.name for job in client.list_jobs()):
            raise ValueError(f"Submission ID {args.name!r} already exists; inspect it before retrying")
        nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0) > 0]
        if not nodes:
            raise RuntimeError("No live GPU Ray nodes")
        expected_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        prepare = ray.remote(num_cpus=1, num_gpus=1)(prepare_node)
        prepared = ray.get(
            [
                prepare.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)).remote(
                    str(REPOSITORY), expected_hash
                )
                for node in nodes
            ],
            timeout=360,
        )
        from omegaconf import OmegaConf

        source_paths = os.pathsep.join(
            str(REPOSITORY / path) for path in ("plugins/phimm/src", "plugins/qwen35_audio/src", ".")
        )
        if os.environ.get("PYTHONPATH"):
            source_paths += os.pathsep + os.environ["PYTHONPATH"]
        composition = subprocess.run(
            [*command, "--cfg", "job", "--resolve"],
            cwd=REPOSITORY,
            env={**os.environ, "PYTHONPATH": source_paths},
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        config = OmegaConf.create(composition.stdout)
        expected_nodes = config.trainer.nnodes
        expected_gpus = config.trainer.n_gpus_per_node
        if len(nodes) != expected_nodes or any(node["Resources"]["GPU"] != expected_gpus for node in nodes):
            raise RuntimeError(
                f"Config requires {expected_nodes} x {expected_gpus} GPUs; "
                f"live nodes have {[node['Resources']['GPU'] for node in nodes]}"
            )
        receipt_dir = REPOSITORY / "tmp/phimm-submissions" / args.name
        receipt_dir.mkdir(parents=True, exist_ok=True)
        (receipt_dir / "effective_config.yaml").write_text(composition.stdout)
        runtime_env = {
            "env_vars": {
                # Prefer the synced source even in workers started before editable setup.
                "PYTHONPATH": source_paths,
                "LD_LIBRARY_PATH": "/usr/local/cuda-13.0/compat:"
                "/root/.pyenv/versions/3.12.9/lib/python3.12/site-packages/nvidia/cu13/lib:"
                + os.environ.get("LD_LIBRARY_PATH", ""),
                "HYDRA_FULL_ERROR": "1",
            }
        }
        submission = client.submit_job(submission_id=args.name, entrypoint=entrypoint, runtime_env=runtime_env)
        receipt = {
            "submission_id": submission,
            "entrypoint": command,
            "dashboard": address,
            "prepared": prepared,
            "nodes": [{"id": node["NodeID"], "gpus": node["Resources"]["GPU"]} for node in nodes],
            "config": str(receipt_dir / "effective_config.yaml"),
        }
        (receipt_dir / "submission.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

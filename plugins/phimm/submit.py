"""Submit an already-synced ASR recipe to the existing Ray cluster."""

import argparse
import json
import os
import shlex
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-id", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--sft", action="store_true", help="Run the Qwen audio supervised trainer")
    mode.add_argument("--sft-evaluate", action="store_true", help="Decode native SFT checkpoints without training")
    parser.add_argument("--config-name")
    parser.add_argument("--config-path")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_intermixed_args(argv)
    import ray
    from ray.job_submission import JobSubmissionClient

    repository = Path(__file__).resolve().parents[2]
    module = "recipe.phimm.main_asr_eval" if args.evaluate else "recipe.phimm.main_asr_remax"
    if args.sft:
        module = "recipe.phimm.sft.main_ray"
    if args.sft_evaluate:
        module = "recipe.phimm.sft.evaluate"
    command = ["python", "-u", "-m", module]
    for flag, value in [("--config-name", args.config_name), ("--config-path", args.config_path)]:
        if value is not None:
            command.extend([flag, value])
    if not any(item.startswith("trainer.experiment_name=") for item in args.overrides):
        command.append(f"trainer.experiment_name={args.submission_id}")
    command.extend(args.overrides)
    source_paths = os.pathsep.join(
        str(repository / path) for path in ("plugins/phimm/src", "plugins/qwen35_audio/src", ".")
    )
    if os.environ.get("PYTHONPATH"):
        source_paths += os.pathsep + os.environ["PYTHONPATH"]
    runtime_env = {
        "env_vars": {
            "PYTHONPATH": source_paths,
            "LD_LIBRARY_PATH": "/usr/local/cuda-13.0/compat:"
            "/root/.pyenv/versions/3.12.9/lib/python3.12/site-packages/nvidia/cu13/lib:"
            + os.environ.get("LD_LIBRARY_PATH", ""),
            "HYDRA_FULL_ERROR": "1",
        }
    }
    ray.init(address="auto", log_to_driver=False)
    try:
        address = ray._private.worker.global_worker.node.address_info["webui_url"]
        client = JobSubmissionClient(address="http://" + address)
        submission = client.submit_job(
            submission_id=args.submission_id,
            entrypoint=f"cd {shlex.quote(str(repository))} && exec {shlex.join(command)}",
            runtime_env=runtime_env,
        )
        print(json.dumps({"submission_id": submission, "entrypoint": command, "dashboard": address}))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

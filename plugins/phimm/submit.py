"""Submit an already-synced ASR recipe to the existing Ray cluster."""

import argparse
import json
import os
import shlex
from pathlib import Path

import ray
from ray.job_submission import JobSubmissionClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-id", required=True)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    ray.init(address="auto")
    address = ray._private.worker.global_worker.node.address_info["webui_url"]
    client = JobSubmissionClient(address="http://" + address)
    module = "recipe.phimm.main_asr_eval" if args.evaluate else "recipe.phimm.main_asr_remax"
    command = ["python", "-u", "-m", module, *args.overrides]
    runtime_env = {
        "env_vars": {
            "LD_LIBRARY_PATH": "/usr/local/cuda-13.0/compat:"
            "/root/.pyenv/versions/3.12.9/lib/python3.12/site-packages/nvidia/cu13/lib:"
            + os.environ.get("LD_LIBRARY_PATH", ""),
            "HYDRA_FULL_ERROR": "1",
        }
    }
    submission = client.submit_job(
        submission_id=args.submission_id,
        entrypoint=f"cd {shlex.quote(str(repository))} && exec {shlex.join(command)}",
        runtime_env=runtime_env,
    )
    print(json.dumps({"submission_id": submission, "entrypoint": command, "dashboard": address}))


if __name__ == "__main__":
    main()

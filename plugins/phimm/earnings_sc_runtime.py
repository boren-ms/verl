"""Execution helper for the isolated earnings-short SC/OpenLong experiment."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import ray
from ray.job_submission import JobSubmissionClient
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

ROOT = Path(__file__).resolve().parents[2]


@ray.remote(num_cpus=0)
def inspect_node():
    import importlib.metadata as md
    import socket

    def command(args):
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()

    return {
        "host": socket.gethostname(),
        "revision": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "versions": {name: md.version(name) for name in ("ray", "torch", "vllm", "transformers", "numpy")},
        "gpus": command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ),
        "gpu_processes": command(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
        ),
        "processes": command(["ps", "-eo", "pid,ppid,etime,args"]),
    }


@ray.remote(num_cpus=1, num_gpus=1)
def prepare_node(revision):
    import torch

    actual = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    assert actual == revision, (actual, revision)
    assert torch.cuda.is_available()
    x = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16)
    assert (x @ x).float().mean().item() == 64
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--system",
            "--no-deps",
            "--no-build-isolation",
            "-e",
            ".",
            "-e",
            "plugins/phimm",
            "-e",
            "plugins/qwen35_audio",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ["python", "-c", "import recipe.phimm.main_asr_remax,hf_qwen35_audio,verl; print(verl.__file__)"],
        cwd=ROOT,
        check=True,
    )
    return {"revision": actual, "host": os.uname().nodename, "gpu": torch.cuda.get_device_name(0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["health", "bootstrap"])
    parser.add_argument("--revision")
    args = parser.parse_args()
    ray.init(address="auto", logging_level="ERROR")
    try:
        dashboard = "http://" + ray._private.worker.global_worker.node.address_info["webui_url"]
        client = JobSubmissionClient(dashboard)
        nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0]
        jobs = []
        for job in client.list_jobs():
            if not job.submission_id:
                continue
            item = {
                key: getattr(job, key, None)
                for key in (
                    "submission_id",
                    "status",
                    "entrypoint",
                    "message",
                    "start_time",
                    "end_time",
                    "driver_exit_code",
                )
            }
            log = client.get_job_logs(job.submission_id)
            clean = re.sub(r"\x1b\[[0-9;]*m", "", log)
            steps = re.findall(r"(?:step:|training/global_step:)(\d+)", clean)
            item.update(
                last_step=max(map(int, steps), default=0), log_bytes=len(log.encode()), tail=clean.splitlines()[-12:]
            )
            if "Traceback (most recent call last)" in clean:
                item["traceback_tail"] = clean[clean.rfind("Traceback (most recent call last)") :][-8000:]
            jobs.append(item)
        node_checks = ray.get(
            [
                inspect_node.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)
                ).remote()
                for n in nodes
            ]
        )
        result = {
            "dashboard": dashboard,
            "jobs": jobs,
            "available": ray.available_resources(),
            "nodes": [{"id": n["NodeID"], "ip": n["NodeManagerAddress"], "gpus": n["Resources"]["GPU"]} for n in nodes],
            "node_checks": node_checks,
            "placement_groups": ray.util.placement_group_table(),
        }
        if args.action == "bootstrap":
            assert args.revision
            assert len(nodes) == 2 and all(n["Resources"]["GPU"] == 8 for n in nodes)
            assert ray.available_resources().get("GPU", 0) == 16
            assert not [j for j in jobs if str(j["status"]) not in ("SUCCEEDED", "FAILED", "STOPPED")]
            assert not [g for g in result["placement_groups"].values() if g["state"] != "REMOVED"]
            assert not [n for n in node_checks if n["gpu_processes"]]
            result["bootstrap"] = ray.get(
                [
                    prepare_node.options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)
                    ).remote(args.revision)
                    for n in nodes
                ]
            )
        print("SC_PIPELINE_RESULT=" + json.dumps(result, default=str), flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# %%
from ray.job_submission import JobSubmissionClient
import fire
import ray
import socket


def ray_url(address="auto"):
    """Get the Ray dashboard URL."""
    if not ray.is_initialized():
        ray.init(address=address)
    url = ray._private.worker.global_worker.node.address_info["webui_url"]
    return f"http://{url}"


class RayJob:
    def __init__(self, address="auto"):
        self.client = JobSubmissionClient(address=address)
        self.url = ray_url(address)
        self.host = socket.gethostname()

    def _jobs(self, status=None, submit=False):
        jobs = self.client.list_jobs()
        for job in jobs:
            if status and job.status != status:
                continue
            if submit and job.submission_id is None:
                continue
            yield job

    def list(self, running=True, submit=False):
        """List running Ray jobs."""
        status = "RUNNING" if running else None
        jobs = self._jobs(status=status, submit=submit)
        print("\nHost:", self.host, "Ray:", self.url)
        for i, job in enumerate(jobs):
            if "ray_job.py" in job.entrypoint:
                continue
            print(f"[{i}]", job.entrypoint)

    def cleanup(self, command=None):
        """Clean up all stopped Ray jobs."""
        jobs = self._jobs(status="RUNNING", submit=True)
        for job in jobs:
            if command and job.entrypoint.find(command) < 0:
                continue
            print(f"Cancel job: {job.entrypoint}")
            self.client.stop_job(job.job_id)


if __name__ == "__main__":
    fire.Fire(RayJob)
# %%

#!/usr/bin/env python3
from ray.job_submission import JobSubmissionClient
import fire


class RayJob:
    def __init__(self, address="auto"):
        self.client = JobSubmissionClient(address=address)

    def list(self):
        """List running Ray jobs."""
        jobs = self.client.list_jobs()
        for job in jobs:
            if job.status != "RUNNING":
                continue
            if job.submission_id is None:
                continue
            print(job.entrypoint)


if __name__ == "__main__":
    fire.Fire(RayJob)

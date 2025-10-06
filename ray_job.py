#!/usr/bin/env python3
from ray.job_submission import JobSubmissionClient
import fire


class RayJob:
    def __init__(self, address="auto"):
        self.client = JobSubmissionClient(address=address)

    def _jobs(self, status=None, submit=False):
        jobs = self.client.list_jobs()
        for job in jobs:
            if status and job.status != status:
                continue
            if submit and job.submission_id is None:
                continue
            yield job

    def list(self):
        """List running Ray jobs."""
        jobs = self._jobs(status="RUNNING", submit=True)
        for job in jobs:
            print(job.entrypoint)

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

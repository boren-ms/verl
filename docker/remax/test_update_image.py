"""Local orchestration tests; never queue a pipeline or access a GPU."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("update_image", Path(__file__).with_name("update_image.py"))
update = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(update)
DIGEST = "sha256:" + "a" * 64
TASK = {"type": "Task", "name": update.DOCKER_TASK, "result": "succeeded", "log": {"id": 5}}
LOG = f"2026-10-24T00:00:00Z torch211-vllm024-42: digest: {DIGEST} size: 857\n"


class ImageUpdateTests(unittest.TestCase):
    def test_partial_pipeline_requires_successful_docker_task_and_digest(self):
        self.assertEqual(update.publication([TASK], LOG)["image_digest"], f"{update.IMAGE}@{DIGEST}")
        for task, log in ((dict(TASK, result="failed"), LOG), (TASK, "no push receipt")):
            with self.assertRaises(RuntimeError):
                update.publication([task], log)

    def test_status_records_digest_without_submitting(self):
        calls = []

        class API:
            def request(self, path, payload=None, raw=False):
                calls.append((path, payload))
                if "/timeline?" in path:
                    return {"records": [TASK]}
                if "/logs/" in path:
                    return LOG
                return {"status": "completed", "result": "partiallySucceeded", "sourceVersion": "abc"}

        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            update.status(API(), 42, Path(folder))
            receipt = json.loads((Path(folder) / "42.json").read_text())
            self.assertEqual(receipt["image_digest"], f"{update.IMAGE}@{DIGEST}")
            self.assertTrue((Path(folder) / "42-docker.log").exists())
        self.assertTrue(all(payload is None for _, payload in calls))

    def test_failed_build_cannot_report_a_previous_success(self):
        class API:
            def request(self, path, **kwargs):
                if "/timeline?" in path:
                    return {"records": [dict(TASK, result="failed")]}
                if "/logs/" in path:
                    return "build failed"
                return {"status": "completed", "result": "failed"}

        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            update.save(Path(folder) / "42.json", {"image_digest": "stale"})
            with self.assertRaises(RuntimeError):
                update.status(API(), 42, Path(folder))
            self.assertNotIn("image_digest", json.loads((Path(folder) / "42.json").read_text()))

    def test_build_queues_exactly_once_and_records_id(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder)
            (state / "bootstrapping/images/verl-remax").mkdir(parents=True)
            args = SimpleNamespace(
                dry_run=False, state_dir=state, branch=update.BRANCH, reuse_context_build="", no_wait=True, timeout=60
            )

            def command(*parts, **kwargs):
                result = ""
                if parts[:4] == ("git", "remote", "get-url", "origin"):
                    result = update.REMOTE
                elif parts[:3] == ("git", "branch", "--show-current"):
                    result = update.BRANCH
                elif parts[:3] == ("git", "rev-parse", "HEAD"):
                    result = "a" * 40
                return subprocess.CompletedProcess(parts, 0, result)

            with (
                patch.object(update, "run", side_effect=command),
                patch.object(update.shutil, "copy2"),
                patch.object(update.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)),
                patch.object(update, "AzureDevOps") as api,
                patch.object(update, "status") as status,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                api.return_value.request.return_value = {"id": 42}
                update.build(args)
                self.assertEqual(api.return_value.request.call_count, 1)
                queued = api.return_value.request.call_args.args[1]
                self.assertEqual(queued["sourceVersion"], "a" * 40)
                self.assertEqual(queued["templateParameters"], {"reuseContextBuildId": ""})
                self.assertFalse(status.call_args.kwargs["wait"])
            self.assertEqual(json.loads((state / "42.json").read_text())["build_id"], 42)

    def test_build_dry_run_never_uses_network_or_git(self):
        for extra in ([], ["--fresh-packages"]):
            with (
                patch.object(update, "run", side_effect=AssertionError("Git/network executed")),
                patch.object(update, "AzureDevOps", side_effect=AssertionError("API executed")),
                patch.object(update.sys, "argv", ["update_image.py", "build", "--dry-run", *extra]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                update.main()

    def test_subprocess_output_can_be_redirected_to_receipt(self):
        with tempfile.TemporaryFile(mode="w+") as output:
            update.run(update.sys.executable, "-c", "print('saved')", stdout=output)
            output.seek(0)
            self.assertEqual(output.read(), "saved\n")

    def test_verify_rejects_image_mismatch_before_bpush(self):
        pod = {
            "spec": {"containers": [{"name": "main", "image": "different"}]},
            "status": {"containerStatuses": [{"name": "main", "ready": True, "restartCount": 0}]},
        }
        with (
            patch.object(update, "run", return_value=subprocess.CompletedProcess([], 0, json.dumps(pod))) as run,
            patch.object(
                update.sys,
                "argv",
                ["update_image.py", "verify", "--pool", "test-pool", "--image", f"{update.IMAGE}@{DIGEST}"],
            ),
        ):
            with self.assertRaises(RuntimeError):
                update.main()
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0], "kubectl")

    def test_verify_uses_expected_commands_and_saves_result(self):
        image = f"{update.IMAGE}@{DIGEST}"
        pod = {
            "metadata": {"uid": "fresh-pod"},
            "spec": {"nodeName": "gpu-node", "containers": [{"name": "main", "image": image}]},
            "status": {"containerStatuses": [{"name": "main", "imageID": image, "ready": True, "restartCount": 0}]},
        }
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(
                image=image,
                pool="image-test",
                cluster="cluster",
                namespace="user",
                kubeconfig=None,
                dry_run=False,
                state_dir=Path(folder),
                timeout=60,
            )
            with (
                patch.object(update, "run", return_value=subprocess.CompletedProcess([], 0, json.dumps(pod))) as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                update.verify(args)
                commands = [call.args for call in run.call_args_list]
                self.assertEqual([cmd[0] for cmd in commands], ["kubectl", "zsh", "brix", "kubectl"])
                remote = commands[2][-1]
                self.assertLess(remote.index("cd /opt/verl-source"), remote.index("bash quick_install.sh"))
                self.assertIn("--nproc-per-node=8", remote)
            self.assertTrue(
                json.loads((Path(folder) / "verify-image-test-fresh-pod/result.json").read_text())["passed"]
            )


if __name__ == "__main__":
    unittest.main()

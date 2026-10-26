#!/usr/bin/env python3
"""Build/publish the Orange image, resume monitoring, or verify a dedicated pool."""

import argparse
import base64
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[2]
REMOTE = "https://dev.azure.com/project-argos/Mimco/_git/bootstrapping"
API_ROOT = "https://dev.azure.com/project-argos/Mimco/_apis/"
BRANCH = "codex/verl-remax-image-oct23"
DOCKER_TASK = "Build and publish dedicated verl image"
IMAGE = "iridiumsdc.azurecr.io/boren/verl-remax"


def run(*args, cwd=REPO, capture=False, **kwargs):
    kwargs.setdefault("stdout", subprocess.PIPE if capture else None)
    return subprocess.run(args, cwd=cwd, check=True, text=True, **kwargs)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


class AzureDevOps:
    def __init__(self):
        token = os.environ.get("AZURE_DEVOPS_EXT_PAT")
        username = ""
        if not token:
            result = run(
                "git",
                "credential",
                "fill",
                capture=True,
                input="protocol=https\nhost=dev.azure.com\npath=project-argos/Mimco/_git/bootstrapping\n\n",
                stderr=subprocess.PIPE,
            )
            credential = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
            token = credential.get("password")
            username = credential.get("username", "")
        if not token:
            raise RuntimeError("Configure the Azure DevOps Git credential helper or AZURE_DEVOPS_EXT_PAT")
        self.authorization = "Basic " + base64.b64encode(f"{username}:{token}".encode()).decode()

    def request(self, path, payload=None, raw=False):
        request = Request(
            API_ROOT + path,
            method="POST" if payload is not None else "GET",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Authorization": self.authorization, "Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=45) as response:
                body = response.read().decode()
        except HTTPError as error:
            # Do not print authentication headers, credentials, or response bodies.
            raise RuntimeError(f"Azure DevOps returned HTTP {error.code} for {request.method}") from None
        return body if raw else json.loads(body)


def build_url(build_id):
    return f"https://dev.azure.com/project-argos/Mimco/_build/results?buildId={build_id}"


def publication(records, logs):
    task = next((r for r in records if r.get("name") == DOCKER_TASK and r.get("type") == "Task"), None)
    if not task or task.get("result") != "succeeded":
        raise RuntimeError("Docker build/push task did not succeed; inspect the pipeline logs")
    matches = re.findall(r"(torch211-vllm024-\d+): digest: (sha256:[0-9a-f]{64})", logs)
    if not matches:
        raise RuntimeError("No published image digest found in the successful Docker task log")
    tag, digest = matches[-1]
    return {"image_tag": f"{IMAGE}:{tag}", "image_digest": f"{IMAGE}@{digest}"}


def status(api, build_id, state_dir, wait=False, timeout=10800):
    receipt_path = state_dir / f"{build_id}.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"build_id": build_id}
    previous = None
    deadline = time.monotonic() + timeout
    print(build_url(build_id), flush=True)
    while True:
        build = api.request(f"build/builds/{build_id}?api-version=7.1")
        timeline = (
            api.request(f"build/builds/{build_id}/timeline?api-version=7.1")
            if build["status"] != "notStarted"
            else {"records": []}
        )
        records = timeline.get("records", [])
        active = [r["name"] for r in records if r.get("state") == "inProgress"]
        current = (build["status"], build.get("result"), active)
        if current != previous:
            print(f"{build['status']} / {build.get('result') or '-'}: {', '.join(active)}", flush=True)
            previous = current
        receipt.update(
            status=build["status"],
            result=build.get("result"),
            url=build_url(build_id),
            source_revision=build.get("sourceVersion"),
        )
        if build["status"] == "completed":
            receipt.pop("image_tag", None)
            receipt.pop("image_digest", None)
            save(state_dir / f"{build_id}-timeline.json", timeline)
            for record in records:
                if record.get("name") == DOCKER_TASK and record.get("log"):
                    logs = api.request(f"build/builds/{build_id}/logs/{record['log']['id']}?api-version=7.1", raw=True)
                    (state_dir / f"{build_id}-docker.log").write_text(logs)
                    if build.get("result") in ("succeeded", "partiallySucceeded"):
                        receipt.update(publication(records, logs))
            save(receipt_path, receipt)
            if not receipt.get("image_digest") or build.get("result") not in ("succeeded", "partiallySucceeded"):
                raise RuntimeError(f"Build did not publish a verified digest; see {receipt_path}")
            print(f"Tag:    {receipt['image_tag']}\nDigest: {receipt['image_digest']}\nReceipt: {receipt_path}")
            if build.get("result") == "partiallySucceeded":
                print("Pipeline completed with issues; Docker task succeeded. Review the saved timeline warnings.")
            return
        save(receipt_path, receipt)
        if not wait:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Monitoring timed out; build continues remotely. Resume with: status {build_id} --wait")
        time.sleep(30)


def build(args):
    if args.dry_run:
        print(
            f"Source: {REPO}\nBuild checkout: {args.state_dir / 'bootstrapping'}\n"
            f"Remote: {REMOTE}\nBranch: {args.branch}\nPipeline: 2910\n"
            f"Reuse wheel artifact: {args.reuse_context_build or 'download fresh'}\n"
            "Would stage source, commit/push the build branch, then queue exactly one build.\n"
            "No Git changes, network calls, builds, or GPU allocations were performed."
        )
        return
    api = AzureDevOps()
    checkout = args.state_dir / "bootstrapping"
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--filter=blob:none", "--sparse", "--branch", args.branch, REMOTE, str(checkout))
        run("git", "sparse-checkout", "set", "images/verl-remax", ".azuredevops", cwd=checkout)
    if run("git", "remote", "get-url", "origin", cwd=checkout, capture=True).stdout.strip() != REMOTE:
        raise RuntimeError(f"Unexpected origin in {checkout}")
    if run("git", "branch", "--show-current", cwd=checkout, capture=True).stdout.strip() != args.branch:
        raise RuntimeError(f"Unexpected branch in {checkout}; use a separate --state-dir")
    if run("git", "status", "--porcelain", cwd=checkout, capture=True).stdout:
        raise RuntimeError(f"Build checkout has local changes: {checkout}; inspect them before retrying")
    run("git", "pull", "--ff-only", cwd=checkout)
    image_dir = checkout / "images/verl-remax"
    # Compare with the artifact-producing revision, not the latest branch state.
    reuse = args.reuse_context_build
    if reuse:
        previous = api.request(f"build/builds/{int(reuse)}?api-version=7.1")
        artifacts = api.request(f"build/builds/{int(reuse)}/artifacts?api-version=7.1")
        if previous.get("definition", {}).get("id") != 2910 or not any(
            item["name"] == "verl-context" for item in artifacts.get("value", [])
        ):
            raise RuntimeError("The reuse build must provide pipeline 2910's verl-context artifact")
        revision = previous["sourceVersion"]
        run("git", "fetch", "origin", revision, cwd=checkout)
        previous_list = run(
            "git", "show", f"{revision}:images/verl-remax/source/docker/remax/packages.list", cwd=checkout, capture=True
        ).stdout
        if previous_list != (REPO / "docker/remax/packages.list").read_text():
            print("Package list changed; downloading fresh wheel artifacts instead of reusing the old bundle.")
            reuse = ""
    source = image_dir / "source"
    if source.exists():
        shutil.rmtree(source)
    run(sys.executable, str(REPO / "docker/remax/stage_source.py"), str(source))
    shutil.copy2(REPO / "docker/remax/Dockerfile", image_dir / "Dockerfile")
    shutil.copy2(REPO / "docker/remax/orange-pipeline.yaml", checkout / ".azuredevops/kube-orange-box.yaml")
    run("git", "add", "--sparse", "images/verl-remax", ".azuredevops/kube-orange-box.yaml", cwd=checkout)
    run("git", "diff", "--cached", "--check", cwd=checkout)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=checkout).returncode
    if changed not in (0, 1):
        raise RuntimeError("Cannot inspect staged build source")
    if changed:
        run("git", "commit", "-m", "Update Orange verl ReMax image source", cwd=checkout)
    run("git", "push", "origin", "HEAD", cwd=checkout)
    revision = run("git", "rev-parse", "HEAD", cwd=checkout, capture=True).stdout.strip()
    payload = {
        "definition": {"id": 2910},
        "sourceBranch": "refs/heads/" + args.branch,
        "sourceVersion": revision,
        "templateParameters": {"reuseContextBuildId": reuse},
    }
    intent = args.state_dir / f"queue-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
    save(intent, {"request": payload, "state": "submitting"})
    # Never retry a POST automatically: the server may have accepted it even if the response was lost.
    try:
        result = api.request("build/builds?api-version=7.1", payload)
    except Exception:
        print(
            f"Queue response uncertain. Check pipeline 2910 for commit {revision} before retrying. Intent: {intent}",
            file=sys.stderr,
        )
        raise
    build_id = result["id"]
    save(intent, {"request": payload, "state": "submitted", "build_id": build_id})
    save(
        args.state_dir / f"{build_id}.json",
        {
            "build_id": build_id,
            "source_revision": revision,
            "local_revision": run("git", "rev-parse", "HEAD", capture=True).stdout.strip(),
            "local_dirty": bool(run("git", "status", "--porcelain", capture=True).stdout),
            "reuse_context_build": reuse,
        },
    )
    print(f"Queued build {build_id}", flush=True)
    status(api, build_id, args.state_dir, wait=not args.no_wait, timeout=args.timeout)


def verify(args):
    if not re.fullmatch(re.escape(IMAGE) + r"@sha256:[0-9a-f]{64}", args.image):
        raise RuntimeError("--image must be the published verl-remax image with an immutable sha256 digest")
    kubectl = ["kubectl"]
    if args.kubeconfig:
        kubectl += ["--kubeconfig", str(args.kubeconfig)]
    else:
        kubectl += ["--context", args.cluster]
    kubectl += ["-n", args.namespace, "get", "pod", args.pool + "-0", "-o", "json"]
    if args.dry_run:
        print(shlex.join(kubectl))
        print(
            f"Would verify image identity, bpush current checkout to {args.pool}, run baked-image checks, "
            "NCCL, quick_install.sh, and checkout checks. The pool must already exist."
        )
        return
    pod = json.loads(run(*kubectl, capture=True).stdout)
    main = next(c for c in pod["spec"]["containers"] if c["name"] == "main")
    runtime = next(c for c in pod["status"].get("containerStatuses", []) if c["name"] == "main")
    if main["image"] != args.image or runtime.get("imageID", "").split("@")[-1] != args.image.split("@")[-1]:
        raise RuntimeError("Requested or actual pod image differs from --image")
    if not runtime.get("ready") or runtime.get("restartCount"):
        raise RuntimeError("Verification requires a Ready dedicated pod with zero restarts")
    report_dir = args.state_dir / f"verify-{args.pool}-{pod['metadata']['uid']}"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "result.json").unlink(missing_ok=True)
    save(
        report_dir / "pod.json",
        {
            "uid": pod["metadata"]["uid"],
            "pool": args.pool,
            "node": pod["spec"]["nodeName"],
            "image": args.image,
            "image_id": runtime["imageID"],
        },
    )
    env = dict(os.environ, BRIX_AVAILABLE_CLUSTERS=args.cluster)
    # bpush is the user's zsh function; quote both path and pool as shell arguments.
    run("zsh", "-ic", f"cd {shlex.quote(str(REPO))} && bpush {shlex.quote(args.pool)}", env=env)
    remote = (
        "cd /opt/verl-source && python /root/code/verl/docker/remax/verify.py && "
        "cd /root/code/verl && python -m torch.distributed.run --standalone --nproc-per-node=8 "
        "docker/remax/verify_nccl.py && bash quick_install.sh && python docker/remax/verify.py"
    )
    log = report_dir / "checks.log"
    with log.open("w") as output:
        print(f"Running GPU verification; follow the log with: tail -f {shlex.quote(str(log))}", flush=True)
        run(
            "brix",
            "ssh",
            args.pool,
            "--",
            "bash -lc " + shlex.quote(remote),
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=args.timeout,
        )
    after = json.loads(run(*kubectl, capture=True).stdout)
    current = next(c for c in after["status"]["containerStatuses"] if c["name"] == "main")
    if after["metadata"]["uid"] != pod["metadata"]["uid"] or current.get("restartCount") or not current.get("ready"):
        raise RuntimeError("Pod changed/restarted during verification; inspect the saved log")
    save(
        report_dir / "result.json",
        {"passed": True, "image": args.image, "pool": args.pool, "pod_uid": pod["metadata"]["uid"], "log": str(log)},
    )
    print(
        f"Verification passed. Logs: {log}\nPause this dedicated pool when finished: "
        f"BRIX_AVAILABLE_CLUSTERS={args.cluster} brix pause {args.pool}"
    )


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=REPO / "tmp/remax-image-update",
        help="Private build checkout, receipts, and logs (default: repo/tmp/remax-image-update)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    update = commands.add_parser("build", help="Stage, commit/push build source, queue once, and monitor")
    update.add_argument("--branch", default=BRANCH)
    wheels = update.add_mutually_exclusive_group()
    wheels.add_argument("--reuse-context-build", default="2200290", help="Wheel artifact build ID (default: 2200290)")
    wheels.add_argument("--fresh-packages", action="store_const", const="", dest="reuse_context_build")
    update.add_argument("--no-wait", action="store_true")
    update.add_argument("--dry-run", action="store_true")
    update.add_argument("--timeout", type=int, default=10800)
    check = commands.add_parser("status", help="Read a build result; optionally wait without queueing again")
    check.add_argument("build_id", type=int)
    check.add_argument("--wait", action="store_true")
    check.add_argument("--timeout", type=int, default=10800)
    gpu = commands.add_parser("verify", help="Check an existing dedicated, fresh 8-GPU image-test pool")
    gpu.add_argument("--pool", required=True)
    gpu.add_argument("--image", required=True)
    gpu.add_argument("--cluster", default="prod-westus2-cw-6")
    gpu.add_argument("--namespace", default="boren")
    gpu.add_argument("--kubeconfig", type=Path)
    gpu.add_argument("--timeout", type=int, default=1800)
    gpu.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.state_dir = args.state_dir.expanduser().resolve()
    if args.command == "build":
        if args.dry_run:
            build(args)
        else:
            args.state_dir.mkdir(parents=True, exist_ok=True)
            with (args.state_dir / "build.lock").open("w") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError("Another build command is using this --state-dir") from None
                build(args)
    elif args.command == "status":
        args.state_dir.mkdir(parents=True, exist_ok=True)
        status(AzureDevOps(), args.build_id, args.state_dir, args.wait, args.timeout)
    else:
        verify(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "Interrupted locally. A queued pipeline continues remotely; use status <build-ID> --wait.", file=sys.stderr
        )
        sys.exit(130)
    except (RuntimeError, subprocess.SubprocessError, OSError, ValueError, StopIteration) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

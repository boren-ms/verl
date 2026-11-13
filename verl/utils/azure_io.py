"""Optional Azure Blob IO; blobfile and bbb are needed only for az:// paths."""

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from filelock import FileLock


def copy_to_local(source, cache_dir=None, always_recopy=False):
    import blobfile as bf

    # As in the ASR source cache, the namespace identifies immutable content.
    namespace = os.environ.get("VERL_REMOTE_CACHE_NAMESPACE", "")
    identity = hashlib.sha256(f"{namespace}\n{source}".encode()).hexdigest()
    root = Path(cache_dir or Path.home() / ".blobfile") / "verl-azure" / identity
    root.mkdir(parents=True, exist_ok=True)
    destination = root / source.rstrip("/").rsplit("/", 1)[-1]
    with FileLock(str(root / "download.lock"), timeout=1800):
        if destination.exists() and not always_recopy:
            return str(destination)
        staging = Path(tempfile.mkdtemp(dir=root))
        payload = staging / destination.name
        try:
            if not bf.exists(source):
                raise FileNotFoundError(source)
            if bf.isdir(source):
                subprocess.run(
                    ["bbb", "sync", "--concurrency", "64", source.rstrip("/") + "/", str(payload) + "/"], check=True
                )
            else:
                bf.copy(source, str(payload), overwrite=True)
            if destination.is_dir():
                shutil.rmtree(destination)
            os.replace(payload, destination)
        finally:
            shutil.rmtree(staging)
    return str(destination)


def upload_path(local_path, remote_path):
    """Publish a file or directory synchronously; propagate transfer failures."""
    import blobfile as bf

    if os.path.isdir(local_path):
        subprocess.run(
            ["bbb", "sync", "--concurrency", "16", str(local_path).rstrip("/") + "/", remote_path.rstrip("/") + "/"],
            check=True,
        )
    else:
        bf.copy(str(local_path), remote_path, overwrite=True)

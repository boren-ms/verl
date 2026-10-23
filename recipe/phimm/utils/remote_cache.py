"""Opt-in node-shared cache for remote resources with an immutable generation.

VERL_REMOTE_CACHE_NAMESPACE must identify immutable remote content, e.g. its
manifest digest. Change it when content changes. Explicit local destinations
retain their existing semantics. No existing caches are deleted or migrated.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from filelock import FileLock


def shared_remote_cache(remote_path, cache_dir, namespace, populate):
    """Publish one complete download atomically; serialize concurrent processes.

    ``populate(destination)`` must either finish successfully or raise. Cache
    failures cannot publish a success marker. The final data path is separate
    from marker metadata so model loaders see only the original payload.
    """
    basename = remote_path.rstrip("/").rsplit("/", 1)[-1]
    if basename in ("", ".", ".."):
        raise ValueError("Remote resource needs a valid basename")
    identity = {"schema": 1, "remote_path": remote_path, "namespace": namespace}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(cache_dir) / "shared-v1"
    root.mkdir(parents=True, exist_ok=True)
    entry = root / key
    with FileLock(str(root / (key + ".lock")), timeout=1800):
        marker = entry / "complete.json"
        payload = entry / "payload" / basename
        if marker.is_file() and payload.exists():
            if json.loads(marker.read_text()) == identity:
                return str(payload)
        # Never overwrite a suspicious/incomplete published entry or remove
        # another process's files. Failure requires explicit operator review.
        if entry.exists():
            raise RuntimeError(f"Incomplete shared cache entry requires review: {entry}")
        staging = Path(tempfile.mkdtemp(prefix=key + ".staging-", dir=root))
        try:
            (staging / "payload").mkdir()
            populate(str(staging / "payload" / basename))
            if not (staging / "payload" / basename).exists():
                raise RuntimeError("Cache download returned without a payload")
            (staging / "complete.json").write_text(json.dumps(identity, sort_keys=True))
            os.rename(staging, entry)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return str(payload)

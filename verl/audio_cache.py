import fcntl
import hashlib
import logging
import os
from pathlib import Path
import subprocess
import uuid


logger = logging.getLogger(__name__)

ORANGE_DATA_PREFIX = "az://orngwus2cresco/data/"
LOCAL_DATA_ROOT = Path("~/data").expanduser()
FALLBACK_CACHE_ROOT = Path("/tmp/verl-audio-cache")
LOCK_ROOT = Path("/tmp/verl-audio-locks")
BLOB_READ_TIMEOUT_SECONDS = 45
BLOB_READ_RETRY_LIMIT = 3


def _split_audio_source(source: str) -> tuple[str, str]:
    if "#" in source:
        file_path, separator, time_range = source.rpartition("#")
        if ":" in time_range:
            return file_path, f"{separator}{time_range}"

    parts = source.rsplit(":", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0], f":{parts[1]}:{parts[2]}"
    return source, ""


def local_audio_source(source: str) -> str:
    """Map an Orange audio reference to its persistent local equivalent."""
    file_path, suffix = _split_audio_source(source)
    if not file_path.startswith(ORANGE_DATA_PREFIX):
        return source
    relative_path = file_path.removeprefix(ORANGE_DATA_PREFIX)
    return f"{LOCAL_DATA_ROOT / relative_path}{suffix}"


def resolve_audio_source(source: str) -> str:
    """Prefer an existing local copy of an Orange audio reference."""
    local_source = local_audio_source(source)
    local_file, _ = _split_audio_source(local_source)
    return local_source if local_source != source and Path(local_file).is_file() else source


def _copy_remote_file(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = local_path.with_name(f"{local_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    for attempt in range(1, BLOB_READ_RETRY_LIMIT + 1):
        try:
            subprocess.run(
                ["bbb", "cp", remote_path, str(temp_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=BLOB_READ_TIMEOUT_SECONDS,
            )
            os.replace(temp_path, local_path)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            temp_path.unlink(missing_ok=True)
            if attempt == BLOB_READ_RETRY_LIMIT:
                raise RuntimeError(
                    f"Failed to cache remote audio after {BLOB_READ_RETRY_LIMIT} attempts: {remote_path}"
                ) from exc
            logger.warning(
                "Failed to cache remote audio %s (attempt %d/%d); retrying.",
                remote_path,
                attempt,
                BLOB_READ_RETRY_LIMIT,
            )


def _ensure_cached_remote_file(remote_path: str, local_path: Path) -> None:
    if local_path.is_file():
        return

    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(str(local_path).encode()).hexdigest()
    with (LOCK_ROOT / f"{lock_name}.lock").open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if not local_path.is_file():
            _copy_remote_file(remote_path, local_path)


def cache_audio_source(source: str) -> str:
    """Cache an Orange audio reference under ``~/data`` and return its local reference."""
    local_source = local_audio_source(source)
    if local_source == source:
        return source
    remote_file, _ = _split_audio_source(source)
    local_file, _ = _split_audio_source(local_source)
    local_path = Path(local_file)
    _ensure_cached_remote_file(remote_file, local_path)
    return local_source


def localize_audio_source(source: str) -> str:
    """Return a readable local file, persistently caching Orange paths."""
    resolved_source = resolve_audio_source(source)
    if resolved_source != source:
        return resolved_source
    if source.startswith(ORANGE_DATA_PREFIX):
        return cache_audio_source(source)
    if "://" not in source:
        return source

    remote_file, suffix = _split_audio_source(source)
    digest = hashlib.sha256(remote_file.encode()).hexdigest()
    extension = Path(remote_file).suffix or ".audio"
    local_path = FALLBACK_CACHE_ROOT / f"{digest}{extension}"
    _ensure_cached_remote_file(remote_file, local_path)
    return f"{local_path}{suffix}"
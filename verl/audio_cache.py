import fcntl
import hashlib
import logging
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import threading
import uuid


logger = logging.getLogger(__name__)

ORANGE_DATA_PREFIX = "az://orngwus2cresco/data/"
LOCAL_DATA_ROOT = Path("~/data").expanduser()
LOCK_ROOT = Path("/tmp/verl-audio-locks")
BLOB_READ_TIMEOUT_SECONDS = int(os.getenv("VERL_BLOB_READ_TIMEOUT_SECONDS", "120"))
BLOB_READ_RETRY_LIMIT = int(os.getenv("VERL_BLOB_READ_RETRY_LIMIT", "10"))

_CACHE_SERVER_PROCESS = None
_CACHE_SERVER_QUEUE = None
_CACHE_SERVER_OWNER_PID = None
_CACHE_SERVER_LOCK = threading.Lock()


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


def _run_bbb_transfer(remote_path: str, local_path: Path) -> None:
    command = ["bbb", "cp", remote_path, str(local_path)]
    for attempt in range(1, BLOB_READ_RETRY_LIMIT + 1):
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=BLOB_READ_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
                raise subprocess.TimeoutExpired(
                    command,
                    BLOB_READ_TIMEOUT_SECONDS,
                    output=stdout,
                    stderr=stderr,
                )
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            local_path.unlink(missing_ok=True)
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


def _copy_remote_file(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = local_path.with_name(f"{local_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    _run_bbb_transfer(remote_path, temp_path)
    os.replace(temp_path, local_path)


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


def _audio_cache_server(request_queue) -> None:
    """Cache submitted source batches in a dedicated process."""
    from concurrent.futures import ThreadPoolExecutor

    while True:
        request_type, payload, max_workers = request_queue.get()
        if request_type == "dataset":
            ds, fields = payload
            sources = tuple(
                dict.fromkeys(
                    source
                    for field in fields
                    for source in ds[field]
                    if isinstance(source, str) and source
                )
            )
        else:
            sources = payload
        if not sources:
            continue
        with ThreadPoolExecutor(max_workers=min(max_workers, len(sources))) as executor:
            for source, error in zip(sources, executor.map(_cache_audio_source_safely, sources), strict=True):
                if error is not None:
                    logger.warning("Failed to cache audio %s: %s", source, error)


def _cache_audio_source_safely(source: str):
    try:
        cache_audio_source(source)
    except Exception as exc:
        return exc
    return None


def _submit_audio_cache_request(request) -> None:
    global _CACHE_SERVER_PROCESS, _CACHE_SERVER_QUEUE, _CACHE_SERVER_OWNER_PID
    owner_pid = os.getpid()
    with _CACHE_SERVER_LOCK:
        if (
            _CACHE_SERVER_PROCESS is None
            or _CACHE_SERVER_OWNER_PID != owner_pid
            or not _CACHE_SERVER_PROCESS.is_alive()
        ):
            _CACHE_SERVER_QUEUE = multiprocessing.Queue()
            _CACHE_SERVER_PROCESS = multiprocessing.Process(
                target=_audio_cache_server,
                args=(_CACHE_SERVER_QUEUE,),
                name="verl-audio-cache",
                daemon=True,
            )
            _CACHE_SERVER_PROCESS.start()
            _CACHE_SERVER_OWNER_PID = owner_pid
        _CACHE_SERVER_QUEUE.put_nowait(request)


def submit_audio_cache(sources, max_workers: int = 16) -> None:
    """Submit audio references for background caching without waiting for them."""
    sources = tuple(dict.fromkeys(source for source in sources if isinstance(source, str) and source))
    if sources:
        _submit_audio_cache_request(("sources", sources, max(1, int(max_workers))))


def submit_audio_cache_dataset(ds, fields, max_workers: int = 16) -> None:
    """Submit dataset audio fields without reading them in the caller process."""
    fields = tuple(field for field in fields if field in ds.column_names)
    if fields:
        _submit_audio_cache_request(("dataset", (ds, fields), max(1, int(max_workers))))


def localize_audio_source(source: str) -> str:
    """Prefer or cache a local copy of an Orange audio reference."""
    resolved_source = resolve_audio_source(source)
    if resolved_source != source:
        return resolved_source
    if source.startswith(ORANGE_DATA_PREFIX):
        return cache_audio_source(source)
    return source
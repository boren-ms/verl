#!/usr/bin/env python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
"""File-system agnostic IO APIs"""

import atexit
import hashlib
import os
import shutil
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor

try:
    from hdfs_io import copy, exists, makedirs  # for internal use only
except ImportError:
    from .hdfs_io import copy, exists, makedirs

__all__ = ["copy", "exists", "makedirs"]

_HDFS_PREFIX = "hdfs://"

# Background executor for non-blocking copy_to_remote uploads.
_UPLOAD_EXECUTOR = None
_UPLOAD_FUTURES: list[Future] = []
_UPLOAD_LOCK = threading.Lock()


def _get_upload_executor() -> ThreadPoolExecutor:
    """Lazily create the shared thread pool used for async ``copy_to_remote`` uploads."""
    global _UPLOAD_EXECUTOR
    with _UPLOAD_LOCK:
        if _UPLOAD_EXECUTOR is None:
            max_workers = int(os.environ.get("VERL_COPY_TO_REMOTE_WORKERS", "4"))
            _UPLOAD_EXECUTOR = ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="copy_to_remote"
            )
            atexit.register(_UPLOAD_EXECUTOR.shutdown, wait=True)
    return _UPLOAD_EXECUTOR


def wait_for_remote_uploads(timeout=None):
    """Block until all in-flight async ``copy_to_remote`` uploads have finished.

    Re-raises the first exception encountered by any background upload. Call this
    before any operation that depends on the uploads being complete (e.g. checkpoint
    rotation that may delete the local files, or process shutdown).

    Args:
        timeout (float, optional): Per-future timeout in seconds. ``None`` waits
            indefinitely.
    """
    with _UPLOAD_LOCK:
        pending = _UPLOAD_FUTURES.copy()
        _UPLOAD_FUTURES.clear()
    first_error = None
    for future in pending:
        try:
            future.result(timeout=timeout)
        except Exception as e:  # noqa: BLE001 - surface the first failure after draining all
            if first_error is None:
                first_error = e
    if first_error is not None:
        raise first_error


def is_non_local(path):
    """Check if a path is a non-local (HDFS) path.

    Args:
        path (str): The path to check.

    Returns:
        bool: True if the path is an HDFS path, False otherwise.
    """
    return path.startswith(_HDFS_PREFIX)


def md5_encode(path: str) -> str:
    """Generate an MD5 hash of a path string.

    This function is used to create unique identifiers for paths, typically
    for creating cache directories or lock files.

    Args:
        path (str): The path to encode.

    Returns:
        str: The hexadecimal MD5 hash of the path.
    """
    return hashlib.md5(path.encode()).hexdigest()


def get_local_temp_path(hdfs_path: str, cache_dir: str) -> str:
    """Generate a unique local cache path for an HDFS resource.
    Creates a MD5-hashed subdirectory in cache_dir to avoid name conflicts,
    then returns path combining this subdirectory with the HDFS basename.

    Args:
        hdfs_path (str): Source HDFS path to be cached
        cache_dir (str): Local directory for storing cached files

    Returns:
        str: Absolute local filesystem path in format:
            {cache_dir}/{md5(hdfs_path)}/{basename(hdfs_path)}
    """
    # make a base64 encoding of hdfs_path to avoid directory conflict
    encoded_hdfs_path = md5_encode(hdfs_path)
    temp_dir = os.path.join(cache_dir, encoded_hdfs_path)
    os.makedirs(temp_dir, exist_ok=True)
    dst = os.path.join(temp_dir, os.path.basename(hdfs_path))
    return dst


def verify_copy(src: str, dest: str) -> bool:
    """
    verify the copy of src to dest by comparing their sizes and file structures.

    return:
        bool: True if the copy is verified, False otherwise.
    """
    if not os.path.exists(src):
        return False
    if not os.path.exists(dest):
        return False

    if os.path.isfile(src) != os.path.isfile(dest):
        return False

    if os.path.isfile(src):
        src_size = os.path.getsize(src)
        dest_size = os.path.getsize(dest)
        if src_size != dest_size:
            return False
        return True

    src_files = set()
    dest_files = set()

    for root, dirs, files in os.walk(src):
        rel_path = os.path.relpath(root, src)
        dest_root = os.path.join(dest, rel_path) if rel_path != "." else dest

        if not os.path.exists(dest_root):
            return False

        for entry in os.listdir(root):
            src_entry = os.path.join(root, entry)
            src_files.add(os.path.relpath(src_entry, src))

        for entry in os.listdir(dest_root):
            dest_entry = os.path.join(dest_root, entry)
            dest_files.add(os.path.relpath(dest_entry, dest))

    if src_files != dest_files:
        return False

    for rel_path in src_files:
        src_entry = os.path.join(src, rel_path)
        dest_entry = os.path.join(dest, rel_path)

        if os.path.isdir(src_entry) != os.path.isdir(dest_entry):
            return False

        if os.path.isfile(src_entry):
            src_size = os.path.getsize(src_entry)
            dest_size = os.path.getsize(dest_entry)
            if src_size != dest_size:
                return False

    return True


def copy_to_shm(src: str):
    """
    Load the model into   /dev/shm   to make the process of loading the model multiple times more efficient.
    """
    shm_model_root = "/dev/shm/verl-cache/"
    src_abs = os.path.abspath(os.path.normpath(src))
    dest = os.path.join(shm_model_root, hashlib.md5(src_abs.encode("utf-8")).hexdigest())
    os.makedirs(dest, exist_ok=True)
    dest = os.path.join(dest, os.path.basename(src_abs))
    if os.path.exists(dest) and verify_copy(src, dest):
        # inform user and depends on him
        print(
            f"[WARNING]: The memory model path {dest} already exists. If it is not you want, please clear it and "
            f"restart the task."
        )
    else:
        if os.path.isdir(src):
            shutil.copytree(src, dest, symlinks=False, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest)
    return dest


def _record_directory_structure(folder_path):
    record_file = os.path.join(folder_path, ".directory_record.txt")
    with open(record_file, "w") as f:
        for root, dirs, files in os.walk(folder_path):
            for dir_name in dirs:
                relative_dir = os.path.relpath(os.path.join(root, dir_name), folder_path)
                f.write(f"dir:{relative_dir}\n")
            for file_name in files:
                if file_name != ".directory_record.txt":
                    relative_file = os.path.relpath(os.path.join(root, file_name), folder_path)
                    f.write(f"file:{relative_file}\n")
    return record_file


def _check_directory_structure(folder_path, record_file):
    if not os.path.exists(record_file):
        return False
    existing_entries = set()
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            relative_dir = os.path.relpath(os.path.join(root, dir_name), folder_path)
            existing_entries.add(f"dir:{relative_dir}")
        for file_name in files:
            if file_name != ".directory_record.txt":
                relative_file = os.path.relpath(os.path.join(root, file_name), folder_path)
                existing_entries.add(f"file:{relative_file}")
    with open(record_file) as f:
        recorded_entries = set(f.read().splitlines())
    return existing_entries == recorded_entries


def to_local(file_name, local_path, hdfs_path=None):
    for remote_dir in [hdfs_path, local_path]:
        if remote_dir is None:
            continue
        remote_file = os.path.join(remote_dir, file_name)
        local_file = copy_to_local(remote_file)
        if local_file is not None:
            return local_file
    return None


def copy_to_remote(local_file, hdfs_path=None, overwrite=False, blocking=True):
    """Upload a local file or directory into the remote (HDFS/blob) directory ``hdfs_path``.

    For a file, it lands at ``hdfs_path/<basename>``. For a directory, its own
    name and internal structure are preserved (files land under
    ``hdfs_path/<basename(local_file)>/...``). No-op when ``hdfs_path`` is ``None``.

    Args:
        local_file (str): Local file or directory to upload.
        hdfs_path (str, optional): Remote destination directory. ``None`` disables upload.
        overwrite (bool): Overwrite existing remote files. Defaults to ``False``.
        blocking (bool): When ``True`` (default), upload synchronously and return the
            remote path. When ``False``, schedule the upload on a background thread and
            return a ``concurrent.futures.Future`` immediately so the caller is not
            blocked. Use :func:`wait_for_remote_uploads` to await pending async uploads.

    Returns:
        When ``hdfs_path`` is ``None``: the original ``local_file``.
        When ``blocking`` is ``True``: the remote path.
        When ``blocking`` is ``False``: a ``Future`` resolving to the remote path.
    """
    if hdfs_path is None:
        return local_file
    from recipe.phimm.utils.shared import upload_file

    remote_path = os.path.join(hdfs_path, os.path.basename(local_file.rstrip("/")))

    def _do_upload():
        if os.path.isdir(local_file):
            for root, _, files in os.walk(local_file):
                rel = os.path.relpath(root, local_file)
                target_dir = remote_path if rel == "." else os.path.join(remote_path, rel)
                for file_name in files:
                    upload_file(
                        os.path.join(root, file_name),
                        os.path.join(target_dir, file_name),
                        overwrite=overwrite,
                    )
        else:
            upload_file(local_file, remote_path, overwrite=overwrite)
        return remote_path

    if blocking:
        return _do_upload()

    future = _get_upload_executor().submit(_do_upload)
    with _UPLOAD_LOCK:
        _UPLOAD_FUTURES.append(future)
    return future


def path_join(*args):
    """Join multiple path components into a single path."""
    if any([x is None for x in args]):
        return None
    return os.path.join(*args)


def copy_to_local(
    src: str, cache_dir=None, filelock=".file.lock", verbose=False, always_recopy=False, use_shm: bool = False
) -> str:
    """Copy files/directories from HDFS to local cache with validation.

    Args:
        src (str): Source path - HDFS path (hdfs://...) or local filesystem path
        cache_dir (str, optional): Local directory for cached files. Uses system tempdir if None
        filelock (str): Base name for file lock. Defaults to ".file.lock"
        verbose (bool): Enable copy operation logging. Defaults to False
        always_recopy (bool): Force fresh copy ignoring cache. Defaults to False
        use_shm (bool): Enable shared memory copy. Defaults to False

    Returns:
        str: Local filesystem path to copied resource
    """
    # Save to a local path for persistence.
    # local_path = copy_local_path_from_hdfs(src, cache_dir, filelock, verbose, always_recopy)
    from recipe.phimm.utils.shared import cache_remote_path

    local_path = cache_remote_path(src, cache_dir=cache_dir)
    if local_path is None:
        return None
    # Load into shm to improve efficiency.
    if use_shm:
        return copy_to_shm(local_path)
    return local_path


def copy_local_path_from_hdfs(
    src: str, cache_dir=None, filelock=".file.lock", verbose=False, always_recopy=False
) -> str:
    """Deprecated. Please use copy_to_local instead."""
    from filelock import FileLock

    assert src[-1] != "/", f"Make sure the last char in src is not / because it will cause error. Got {src}"

    if is_non_local(src):
        # download from hdfs to local
        if cache_dir is None:
            # get a temp folder
            cache_dir = tempfile.gettempdir()
        os.makedirs(cache_dir, exist_ok=True)
        assert os.path.exists(cache_dir)
        local_path = get_local_temp_path(src, cache_dir)
        # get a specific lock
        filelock = md5_encode(src) + ".lock"
        lock_file = os.path.join(cache_dir, filelock)
        with FileLock(lock_file=lock_file):
            if always_recopy and os.path.exists(local_path):
                if os.path.isdir(local_path):
                    shutil.rmtree(local_path, ignore_errors=True)
                else:
                    os.remove(local_path)
            if not os.path.exists(local_path):
                if verbose:
                    print(f"Copy from {src} to {local_path}")
                copy(src, local_path)
                if os.path.isdir(local_path):
                    _record_directory_structure(local_path)
            elif os.path.isdir(local_path):
                # always_recopy=False, local path exists, and it is a folder: check whether there is anything missed
                record_file = os.path.join(local_path, ".directory_record.txt")
                if not _check_directory_structure(local_path, record_file):
                    if verbose:
                        print(f"Recopy from {src} to {local_path} due to missing files or directories.")
                    shutil.rmtree(local_path, ignore_errors=True)
                    copy(src, local_path)
                    _record_directory_structure(local_path)
        return local_path
    else:
        return src


def local_mkdir_safe(path):
    """_summary_
    Thread-safe directory creation function that ensures the directory is created
    even if multiple processes attempt to create it simultaneously.

    Args:
        path (str): The path to create a directory at.
    """

    from filelock import FileLock

    if not os.path.isabs(path):
        working_dir = os.getcwd()
        path = os.path.join(working_dir, path)

    # Using hash value of path as lock file name to avoid long file name
    lock_filename = f"ckpt_{hash(path) & 0xFFFFFFFF:08x}.lock"
    lock_path = os.path.join(tempfile.gettempdir(), lock_filename)

    try:
        with FileLock(lock_path, timeout=60):  # Add timeout
            # make a new dir
            os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Warning: Failed to acquire lock for {path}: {e}")
        # Even if the lock is not acquired, try to create the directory
        os.makedirs(path, exist_ok=True)

    return path

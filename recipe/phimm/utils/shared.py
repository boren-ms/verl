import blobfile as bf
import subprocess
import sys
from pathlib import Path
import functools
import hashlib
import importlib.metadata


def is_package_version(package_name, target_version):
    """Check if the specified package is installed with the target version."""
    try:
        version = importlib.metadata.version(package_name)
        return version == target_version
    except importlib.metadata.PackageNotFoundError:
        return False


def hash_id(d) -> str:
    return hashlib.sha256(str(d).encode("utf-8")).hexdigest()


def to_int(value, default=-1):
    """Convert a value to an integer, if possible."""
    try:
        return int(value)
    except ValueError:
        return default


def to_list(x, default=None):
    """Convert the input to a list."""
    if x is None:
        return default
    return x if is_list(x) else [x]


def is_list(obj):
    return not isinstance(obj, (str, bytes)) and hasattr(obj, "__getitem__")


def chkp_index(name, default=-1):
    """Extract the checkpoint index from a checkpoint directory name."""
    if not name.startswith("checkpoint-"):
        return default
    return to_int(name.split("-")[-1], default)


def find_chkps(model_dir, specified=None):
    """Find all checkpoint directories in the model directory."""
    chkps = [d for d in bf.scandir(model_dir) if d.is_dir and d.name.startswith("checkpoint-")]
    if not chkps:
        return []
    chkps = sorted(chkps, key=lambda d: chkp_index(d.name), reverse=True)
    if specified is None:
        return [chkp.path for chkp in chkps]

    if isinstance(specified, int):
        specified = [specified]

    idxs = [chkp_index(chkp.name) for chkp in reversed(chkps)]  # ascending
    chkp_indices = [i if i >= 0 else idxs[i] for i in map(to_int, specified) if -i <= len(idxs)]
    return [chkp.path for chkp in chkps if chkp_index(chkp.name) in chkp_indices]


def get_config_path(config_path=None):
    """Get the config path from command line arguments or environment variables."""
    if config_path:
        return Path(config_path).resolve()

    print("Find config from sys:", sys.argv)
    if "--config" in sys.argv:
        config_index = sys.argv.index("--config") + 1
        if config_index < len(sys.argv):
            return Path(sys.argv[config_index]).resolve()
    if "--config-name" in sys.argv:
        config_index = sys.argv.index("--config-name") + 1
        if config_index < len(sys.argv):
            return Path(sys.argv[config_index])
    return None


def run_cmd(cmd, cwd=None, check=True):
    """Run a shell command and print it."""
    if is_list(cmd):
        cmd = " ".join(cmd)
    print(f"Running: {cmd}")
    print(f"Working Dir: {cwd}")
    ret = subprocess.run(cmd, shell=True, check=check, cwd=cwd)
    print(f"Cmd: {cmd} returned: {ret.returncode}")
    return ret


@functools.cache
def cache_dir(remote_path, local_path=None):
    """Sync a directory from remote to local."""
    if not remote_path.startswith("az://"):
        return remote_path

    if local_path is None:
        local_path = str(Path.home() / Path(".blobfile", *Path(remote_path).parts[3:]))

    print(f"Syncing {remote_path} to {local_path} ...")

    cmd = [
        "bbb",
        "sync",
        "--concurrency",
        "64",
        f"{remote_path.rstrip('/')}/",
        f"{local_path.rstrip('/')}/",
    ]
    run_cmd(cmd, check=True)
    return local_path


def get_value(d, key, default=None):
    """Get a value from a nested dictionary using dot notation."""
    if key in d:
        return d[key]
    keys = key.split(".")
    for k in keys:
        if k in d:
            d = d[k]
        else:
            return default
    return d


def get_values(lst, key, default=None):
    """Get a list of values from a list of dictionaries using dot notation."""
    return [get_value(d, key, default) for d in lst]


def rank_print(*args, main=True, **kwargs):
    if main and not dist_state().is_main_process:
        return
    rank = dist_state().process_index
    print(f"[{rank}]", *args, **kwargs)


def dist_state():
    from accelerate import PartialState

    return PartialState()


def all_rank_print(*args, **kwargs):
    rank_print(*args, main=False, **kwargs)

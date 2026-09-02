#! /usr/bin/env python3
import subprocess
import ray
import os
from pathlib import Path
import fire
import time
import blobfile as bf


def sorted_nodes(nodes):
    """Sort nodes by their index."""
    if nodes is None:
        return None
    if isinstance(nodes, str):
        nodes = [int(x) for x in nodes.split(",")]
    if not is_list(nodes):
        nodes = [nodes]
    return sorted(nodes)


def to_list(data):
    """Convert data to a list if it is not already."""
    if is_list(data):
        return list(data)
    return [data]


def is_list(obj):
    return not isinstance(obj, (str, bytes)) and hasattr(obj, "__getitem__")


def to_int(value, default=-1):
    """Convert a value to an integer, if possible."""
    try:
        return int(value)
    except ValueError:
        return default


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


def upload_file(local_path, remote_path, overwrite=False):
    """Upload a file from local to remote storage."""
    local_mtime = Path(local_path).stat().st_mtime
    remote_mtime = None
    if bf.exists(remote_path) and not overwrite:
        print(f"Remote file {remote_path} already exists, skipping upload.")
        return
    try:
        if bf.exists(remote_path):
            remote_mtime = bf.stat(remote_path).mtime
    except Exception as e:
        print(f"Could not stat remote file {remote_path}: {e}")
    # Copy if remote does not exist or local is newer
    if remote_mtime is None or local_mtime > remote_mtime:
        print(f"Syncing file {local_path.name} to {remote_path} (local newer or remote missing)")
        bf.copy(local_path, remote_path, overwrite=True)
    else:
        print(f"Skipping {local_path.name}: remote is newer or same.")


@ray.remote
class OutputWatcher:
    def __init__(self, local_dir, remote_dir, interval=600, sync_all=True):
        self.local_dir = local_dir
        self.remote_dir = remote_dir
        self.interval = interval
        self._running = True
        self.sync_all = sync_all

    def sync_output_dir(self):
        """sync checkpoint folder from remote to local."""
        print("Syncing latest checkpoint from local to remote ...")
        if not Path(self.local_dir).exists():
            print(f"Local directory [{self.local_dir}] does not exist, skipping sync.")
            return
        if self.sync_all:
            print(f"Syncing all files from {self.local_dir} to {self.remote_dir}")
            cmd = [
                "bbb",
                "sync",
                "--concurrency",
                "64",
                f"{self.local_dir}/",
                f"{self.remote_dir}/",
            ]
            run_cmd(cmd)
            return
        print(f"Syncing files expecting checkpoints from {self.local_dir} to {self.remote_dir}")
        cmd = [
            "bbb",
            "sync",
            "--concurrency",
            "64",
            f"{self.local_dir}/",
            f"{self.remote_dir}/",
            "--exclude",
            "checkpoint-*",
        ]
        run_cmd(cmd)
        print("Syncing latest checkpoint ...")
        chkp_dirs = [d for d in Path(self.local_dir).iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        chkp_dirs = sorted(chkp_dirs, key=lambda d: chkp_index(d.name), reverse=True)
        ckhps = [chkp_index(d.name) for d in chkp_dirs]
        if not chkp_dirs:
            print(f"No checkpoint found in {self.local_dir}.")
            return
        print(f"Found {len(chkp_dirs)} checkpoints in {self.local_dir}.")
        print("Latest 20 checkpoints: ", ckhps[:20])
        local_chkp_dir = chkp_dirs[0]
        print(f"Latest checkpoint: {local_chkp_dir}")
        remote_chkp_dir = f"{self.remote_dir}/{local_chkp_dir.relative_to(self.local_dir)}"
        cmd = [
            "bbb",
            "sync",
            "--concurrency",
            "64",
            f"{local_chkp_dir}/",
            f"{remote_chkp_dir}/",
        ]
        print(f"Syncing latest checkpoint from {local_chkp_dir} to {remote_chkp_dir}")
        run_cmd(cmd)
        print("Sync completed.")

    def start(self):
        print(f"Watcher started with interval {self.interval / 60} minutes.")
        print(f"Local dir: {self.local_dir}")
        print(f"Remote dir: {self.remote_dir}")
        while self._running:
            print("Watcher tick!")
            self.sync_output_dir()
            time.sleep(self.interval)

    def flush(self):
        """Flush the output directory by syncing it."""
        print("Flushing output directory...")
        self.sync_output_dir()
        print("Flush completed.")

    def stop(self, flush=True):
        """Stop the output watcher."""
        if flush:
            self.flush()
        self._running = False
        print("Watcher stopped.")


def run_output_watcher(local_dir=None, remote_dir=None, interval=600, sync_all=False, head_label=None):
    """Start the output watcher to sync outputs periodically."""
    resources = {}
    if head_label is not None:
        resources = {head_label: 0.01}
    print(f"Watching  @ {head_label} every {interval / 60} minutes")
    print(f"Local directory: {local_dir}")
    print(f"Remote directory: {remote_dir}")
    watcher = OutputWatcher.options(resources=resources).remote(
        local_dir=local_dir, remote_dir=remote_dir, interval=interval, sync_all=sync_all
    )
    watcher.start.remote()
    return watcher


def is_valid_model_path(model_dir):
    """Check if the model path is valid."""
    if not bf.exists(model_dir):
        # print(f"Model path {model_dir} does not exist.")
        return False
    if not bf.isdir(model_dir):
        # print(f"Model path {model_dir} is not a directory.")
        return False
    if any(is_valid_model_path(chkp) for chkp in find_chkps(model_dir)):
        return True
    config_file = f"{model_dir}/config.json"
    if not bf.exists(config_file):
        # print(f"Config file {config_file} does not exist in the model directory.")
        return False
    if not any(bf.glob(f"{model_dir}/*.safetensors")):
        # print(f"No .safetensors files found in {model_dir}.")
        return False
    return True


def scan_models(input_dir):
    """Scan the input directory for valid model paths."""
    if not bf.exists(input_dir) or not bf.isdir(input_dir):
        return []
    if is_valid_model_path(input_dir):
        return [input_dir]
    paths = []
    for p in bf.scandir(input_dir):
        if p.is_file:
            continue
        if is_valid_model_path(p.path):
            paths.append(p.path)
        else:
            paths += scan_models(p.path)
    return paths


def search_models(model_path=None):
    """Search for the model path in the local filesystem."""
    model_path = model_path or ""
    remote_model_dir = f"{ORNG_USER.output_path}/{model_path}"
    model_paths = scan_models(remote_model_dir)

    if not model_paths:
        print(f"Found no models from {remote_model_dir}, switching to data folder")
        model_path = f"{ORNG_USER.home_path}/data/ckp/hf_models/{model_path}"
        if is_valid_model_path(model_path):
            model_paths += scan_models(model_path)
    return model_paths


def get_region():
    """Get the region of the Kubernetes cluster from the environment variable."""
    rcall_kube_cluster = os.environ.get("RCALL_KUBE_CLUSTER", "")
    cluster_region = rcall_kube_cluster.split("-")[1] if "-" in rcall_kube_cluster else None
    return cluster_region


REGION_STORAGES = {
    "southcentralus": "orngscuscresco",
    "westus2": "orngwus2cresco",
    "uksouth": "orngcresco",
}


class UserStorage:
    """Class to manage user storage paths based on the region of the Kubernetes cluster."""

    def __init__(self, region=None):
        """Initialize the UserStorage with the specified region."""
        self.region = region or get_region()
        if not self.region:
            self.region = "westus2"
            print("Warning: RCALL_KUBE_CLUSTER not set, defaulting region to westus2")
        self.region_storage = REGION_STORAGES.get(self.region, "orngscuscresco")
        self.user = os.environ.get("OPENAI_USER", "boren")

    @property
    def home_path(self):
        """Get the storage path based on the region."""
        return f"{self.blob_path}/{self.user}"

    @property
    def blob_path(self):
        """Get the data storage account based on the region."""
        return f"az://{self.region_storage}/data"

    @property
    def data_path(self):
        """Get the user data storage path based on the region."""
        return f"{self.home_path}/data"

    @property
    def output_path(self):
        """Get the user output storage path based on the region."""
        return f"{self.home_path}/outputs"


def local_home():
    """Get the local home path."""
    # redirect to /root/code as home, since it is 20TB
    # return Path.home() / "code"
    return Path.home()


ORNG_USER = UserStorage()


def get_output_dirs(rel_path=None):
    """Get the remote output directory based on the job name."""
    # job_name = job_name or os.environ.get("RCALL_JOB_NAME", None)
    remote_output_dir = f"{ORNG_USER.output_path}"
    local_output_dir = local_home() / "outputs"
    if rel_path:
        local_output_dir = local_output_dir / rel_path
        remote_output_dir = f"{remote_output_dir}/{rel_path}"
    return str(local_output_dir), remote_output_dir


def is_remote_path(file_path):
    """Check if the file path is a remote path."""
    return str(file_path).startswith("az://")


def get_local_path(file_path):
    """Get the local path for the remote path."""
    file_path = str(file_path)
    if not is_remote_path(file_path):
        return file_path
    return file_path.replace(ORNG_USER.home_path, str(local_home()))


def get_remote_path(file_path):
    """Get the remote path for the local path."""
    file_path = str(file_path)
    if is_remote_path(file_path):
        return file_path
    return file_path.replace(str(local_home()), ORNG_USER.home_path)


@ray.remote
def prepare_local_checkpoint(local_dir, remote_dir):
    """Prepare output on each node by syncing from the remote storage."""
    hostname = os.uname().nodename
    print(f"Sync remote output on node: {hostname}")
    print(f"Remote output directory: {remote_dir}")
    print(f"Local output directory: {local_dir}")
    if not bf.exists(remote_dir) or not bf.isdir(remote_dir):
        print(f"Remote directory [{remote_dir}] does not exist.")
        return

    # sync remote files to local directory
    for file_path in bf.scandir(remote_dir):
        if not file_path.is_file:
            continue
        local_file_path = Path(local_dir) / file_path.name
        if local_file_path.exists():
            print(f"File {local_file_path} already exists, skipping.")
            continue
        print(f"Syncing file {file_path.name} to {local_file_path}")
        # local_file_path.parent.mkdir(parents=True, exist_ok=True)
        # bf.copy(file_path, local_file_path)
        cmd = ["bbb", "cp", f"{remote_dir}/{file_path.name}", f"{local_file_path}"]
        run_cmd(cmd)

    # sync remote checkpoints to local directory
    chkps = [(chkp_index(d.name), d.name) for d in bf.scandir(remote_dir) if d.is_dir and chkp_index(d.name) >= 0]
    chkps = sorted(chkps, key=lambda x: x[0], reverse=True)
    if not chkps:
        print(f"No checkpoints found in {remote_dir}.")
        return
    print(f"Found {len(chkps)} checkpoints in {remote_dir}.")
    print("Latest 20 checkpoints: ", [chkp[0] for chkp in chkps[:20]])
    latest_chkp = chkps[0][1]
    print(f"Syncing latest checkpoint ({latest_chkp}) to local directory...")
    cmd = [
        "bbb",
        "sync",
        "--concurrency",
        "64",
        f"{remote_dir}/{latest_chkp}/",
        f"{local_dir}/{latest_chkp}/",
    ]
    run_cmd(cmd)
    print("Data preparation completed.")


@ray.remote
def prepare_env(forced=False):
    """Prepare the environment on each node by installing necessary packages."""
    hostname = os.uname().nodename
    print(f"Preparing environment on node: {hostname}")
    if forced:
        run_cmd("find . -maxdepth 1 -name '.env_done_*' -delete")
    run_cmd("bash quick_install.sh")
    print("Environment preparation completed.")


@ray.remote
def prepare_data(forced=False):
    """Prepare data on each node by syncing from the remote storage."""
    hostname = os.uname().nodename
    print(f"Preparing data on node: {hostname}")
    local_dir = local_home() / "data"
    done_tag = local_dir / "data_preparation_done"
    if done_tag.exists() and not forced:
        print(f"Data preparation already done on {hostname}, skipping.")
        return
    remote_dir = ORNG_USER.data_path
    print(f"Remote directory: {remote_dir}")

    rel_dirs = [
        # "gsm8k",
        # "ckp/hf_models/Qwen2.5-0.5B-Instruct",
        # "ckp/hf_models/Qwen2-0.5B-Reward",
        # "ckp/hf_models/phi-libri_ft_m1000_p8_new-QpHq_1000",
        "ckp/hf_models/phi-libri_ft_m1000_p8_new-QpHq/5000_hf",
        # "ckp/hf_models/phi-libri_ft_m1000_p8_new-QpHq/5000_hf_merged",
        "ckp/hf_models/phi4_mm_bias_merged",
        # "ckp/hf_models/phi4_mm_bias",
        # "ckp/hf_models/Phi4-7b-ASR-2506",
        # "ckp/hf_models/libri_ft_m200_p8_bp6_new_notag_ckp5000",
        "ckp/hf_models/phi4-7b-fast-api-s2-final-v4",
        # "ckp/hf_models/Phi4-7b-ASR-2506-v2",
        "ckp/hf_models/Phi-4-multimodal-instruct",
        # "ckp/hf_models/roberta-large-ner-english",  # to tag entities
        "tools",
        # "Evaluation/InhouseASR/EWER/en-US-entity-v3",
        "librispeech_biasing/words",
        "librispeech_biasing/ref",
        "LibriSpeech/test-clean",
        "LibriSpeech/test-other",
        # "parquet",
        # "LibriSpeech/train-clean-360/115/122944",
    ]

    for rel_dir in rel_dirs:
        print(f"Syncing directory: {rel_dir}")
        cmd = [
            "bbb",
            "sync",
            "--concurrency",
            "64",
            f"{remote_dir}/{rel_dir}",
            f"{local_dir}/{rel_dir}",
        ]
        run_cmd(cmd)

    rel_files = [
        "LibriSpeech/ls_30k_shuf.tsv",
        "LibriSpeech/debug.tsv",
    ]
    for rel_file in rel_files:
        print(f"Syncing file: {rel_file}")
        cmd = ["bbb", "cp", f"{remote_dir}/{rel_file}", f"{local_dir}/{rel_file}"]
        run_cmd(cmd)
    print("Data preparation completed.")
    done_tag.touch()


def update_envs(yaml_path):
    """Reads a YAML file, substitutes environment variables in its content"""
    print(f"Updating variables in {yaml_path}")
    os.environ["DATA_STORAGE"] = ORNG_USER.region_storage
    content = Path(yaml_path).read_text()
    expanded_content = os.path.expandvars(content)
    Path(yaml_path).write_text(expanded_content)


def run_cmd(cmd, cwd=None, check=True):
    """Run a shell command and print it."""
    if is_list(cmd):
        cmd = " ".join(cmd)
    print(f"Running: {cmd}")
    if not cwd:
        cwd = str(Path(__file__).parent)
    print(f"Working Dir: {cwd}")
    ret = subprocess.run(cmd, shell=True, check=check, cwd=cwd)
    print(f"Cmd: {cmd} returned: {ret.returncode}")
    return ret


@ray.remote
def broadcast_local_dir(folder, head_node):
    """Sync the Folder from the remote storage."""
    cur_node = os.uname().nodename
    # Ensure the Folder exists for each node
    Path(folder).mkdir(parents=True, exist_ok=True)

    if cur_node == head_node:
        print(f"Skipping checkpoint sync on head node: {cur_node}")
        return
    print(f"Syncing checkpoints from head node: {head_node} to current node: {cur_node}")
    cmd = ["rsync", "-avz", f"{head_node}:{folder}/", f"{folder}/"]
    run_cmd(cmd)
    print("Folder syncing completed.")


@ray.remote
def sync_remote_dir(dir_path, push=None):
    """Sync the remote directory to the local directory."""
    if is_remote_path(dir_path):
        local_dir = get_local_path(dir_path)
        remote_dir = dir_path
        push = False if push is None else push
    else:
        local_dir = dir_path if Path(dir_path).is_absolute() else local_home() / dir_path
        remote_dir = get_remote_path(local_dir)
        push = True if push is None else push

    local_dir = str(local_dir).rstrip("/")
    remote_dir = str(remote_dir).rstrip("/")

    if push:
        print(f"Push from {local_dir} to {remote_dir}")
        cmd = ["bbb", "sync", "--concurrency", "64", f"{local_dir}/", f"{remote_dir}/"]
        if not bf.exists(local_dir):
            print(f"Local directory {local_dir} does not exist, skipping sync.")
            return
    else:
        print(f"Pull from {remote_dir} to  {local_dir}")
        cmd = ["bbb", "sync", "--concurrency", "64", f"{remote_dir}/", f"{local_dir}/"]
        if not bf.exists(remote_dir):
            print(f"Remote directory {remote_dir} does not exist, skipping sync.")
            return
    run_cmd(cmd)
    print("Sync completed.")


@ray.remote
def release_gpus():
    """Release GPUs on the current node."""
    hostname = os.uname().nodename
    print(f"Releasing GPUs on node: {hostname}")
    list_cmd = "lsof /dev/nvidia* | awk '{print $2}' | grep -E '^[0-9]+$' | sort -u"
    kill_cmd = "lsof /dev/nvidia* | awk '{print $2}' | grep -E '^[0-9]+$' | sort -u | xargs -I {} kill -9 {}"
    print("Listing processes using NVIDIA devices:")
    run_cmd(list_cmd)
    print("Killing processes using NVIDIA devices:")
    run_cmd(kill_cmd)
    print("List processes using NVIDIA devices again:")
    run_cmd(list_cmd)
    print("GPUs released.")


@ray.remote
def list_gpus():
    """List available GPUs on the current node."""
    cmd = "nvidia-smi | grep Default"
    print("Listing available GPUs:")
    run_cmd(cmd)
    print("GPUs listed.")


@ray.remote
def job_log(cmd="tail", key=None, n=100, log_dir=None):
    log_dir = str(log_dir or os.environ.get("RCALL_LOGDIR", local_home() / "results"))
    pattern = f"*{key}*" if key else "*"
    cmd = f"{cmd} -n {n}  {log_dir}/{pattern}.log"
    print(f"Tailing logs in {log_dir} with command: {cmd}")
    run_cmd(cmd)


class RayNode:
    def __init__(self, indexs=None):
        """Initialize the RayHelper with the specified nodes."""
        print("Connecting to Ray cluster...")
        ray.init(address="auto")  # Connect to the running cluster
        print("Connected to Ray cluster.")
        nodes = [node for node in ray.nodes() if node["Alive"]]
        nodes = sorted(nodes, key=lambda x: x["NodeManagerHostname"])
        print("Found nodes:")
        for i, node in enumerate(nodes):
            print(f" - {i}: {node['NodeManagerHostname']}[{node['NodeManagerAddress']}]")

        self.indexs = to_list(indexs) if indexs is not None else list(range(len(nodes)))
        self.nodes = [nodes[i] for i in self.indexs]
        print(
            f"Initialized RayHelper with {len(self.nodes)} nodes: {[node['NodeManagerHostname'] for node in self.nodes]}"
        )

    @property
    def num_nodes(self):
        return len(self.nodes)

    def label(self, i=0):
        """Get the node IP address from environment variables."""
        node_ip = self.nodes[i]["NodeManagerAddress"]
        return f"node:{node_ip}"

    def hostname(self, i=0):
        """Get the node hostname from the list of nodes."""
        return self.nodes[i]["NodeManagerHostname"]
        # job_name = os.environ.get("RCALL_JOB_NAME", None)
        # assert job_name is not None, "RCALL_JOB_NAME must be set"
        # idx = self.indexs[i]
        # return f"{job_name}-{idx}"  # head node IP

    def run(self, func, *args, waiting=True, **kwargs):
        # Launch one task per node, each pinned to a specific node
        results = []
        for node in self.nodes:
            node_ip = node["NodeManagerAddress"]
            # Use custom resource label to ensure the function runs on this node
            # Each node has a resource label 'node:<ip>'
            node_label = f"node:{node_ip}"
            result = func.options(resources={node_label: 0.01}).remote(*args, **kwargs)
            results.append(result)
        if waiting:
            results = ray.get(results)
        return results

    def async_run(self, func, *args, **kwargs):
        """Run a function asynchronously on all nodes."""
        # Launch one task per node, each pinned to a specific node
        return self.run(func, *args, waiting=False, **kwargs)

    def list_nodes(self):
        """List all nodes in the Ray cluster."""
        print(f"Found {len(self.nodes)} nodes in the cluster:")
        for node in self.nodes:
            print(f" - {node['NodeName']}[{node['NodeManagerAddress']}] (Alive: {node['Alive']})")
            print(f"   Resources: {node['Resources']}")
        return self.nodes

    def check_gpus(self):
        """Check GPU availability on all Ray nodes."""
        n_total = ray.cluster_resources().get("GPU", 0)
        n_free = ray.available_resources().get("GPU", 0)
        print(f"Free GPUs: {n_free}/{n_total}")

    def release_gpus(self):
        """Release GPUs on all Ray nodes."""
        self.run(release_gpus)

    def broadcast_local_dir(self, folder=None):
        """Sync output directories across all Ray nodes."""
        folder = str(folder or local_home() / "outputs")
        head_node = self.hostname(0)
        self.run(broadcast_local_dir, folder, head_node=head_node)

    def log(self, cmd="tail", key=None, n=100, log_dir=None):
        """Tail logs from all Ray nodes."""
        return self.run(job_log, cmd, key, n, log_dir)

    def run_cmd(self, *args, **kwargs):
        """Run a command on all Ray nodes."""
        cmd = " ".join(args)
        for k, v in kwargs.items():
            cmd += f" --{k} {v}"
        print(f"Running: {cmd}")
        self.run(ray.remote(run_cmd), cmd)

    def prepare_env(self, forced=False):
        """Prepare the environment on all Ray nodes by installing necessary packages."""
        print("Preparing environment on all nodes...")
        self.run(prepare_env, forced=forced)

    def prepare_data(self, forced=False):
        """Prepare data on all Ray nodes by syncing from the remote storage."""
        print("Preparing data on all nodes...")
        self.run(prepare_data, forced=forced)

    def prepare_local_checkpoint(self, local_dir=None, remote_dir=None):
        """Prepare output on all Ray nodes by syncing from the remote storage."""
        if local_dir is None or remote_dir is None:
            local_dir, remote_dir = get_output_dirs()
        print(f"Preparing local checkpoint on all nodes: {local_dir} from {remote_dir}")
        self.run(prepare_local_checkpoint, local_dir, remote_dir)
        self.broadcast_local_dir(local_dir)  # ensure all nodes have the latest output

    def prepare_all(self, local_dir=None, remote_dir=None, forced=False):
        """Prepare the environment, data, and output on all Ray nodes."""
        results = []
        print("Preparing all nodes...")
        if forced:
            print("Releasing GPUs...")
            results += self.async_run(release_gpus)
        print("Preparing environment...")
        results += self.async_run(prepare_env, forced=forced)
        print("Preparing data...")
        results += self.async_run(prepare_data, forced=forced)
        if local_dir is None or remote_dir is None:
            local_dir, remote_dir = get_output_dirs()
        print(f"Preparing local output: {local_dir} from {remote_dir}")
        results += self.async_run(prepare_local_checkpoint, local_dir, remote_dir)
        results = ray.get(results)
        self.broadcast_local_dir(local_dir)

    def run_output_watcher(self, local_dir=None, remote_dir=None, interval=600, sync_all=False):
        """Run the output watcher on head."""
        if local_dir is None or remote_dir is None:
            local_dir, remote_dir = get_output_dirs()
        print(f"Running output watcher on head: {local_dir} from {remote_dir} every {interval / 60} minutes")
        head_label = self.label(0)
        return run_output_watcher(local_dir, remote_dir, interval, sync_all, head_label=head_label)

    def show_remote_dirs(self, rel_path=None):
        """Show remote directories."""
        rel_path = rel_path or ""
        print("Remote Home:", f"{ORNG_USER.home_path}/{rel_path}")
        print("Remote Data:", f"{ORNG_USER.data_path}/{rel_path}")
        print("Remote Output:", f"{ORNG_USER.output_path}/{rel_path}")

    def sync_remote_dir(self, dir_path, push=None):
        """Sync a remote directory to the local directory."""
        self.run(sync_remote_dir, dir_path, push=push)

    def search_models(self, model_path=None):
        """Search for models in the remote storage."""
        model_paths = search_models(model_path)
        if not model_paths:
            print(f"No models found for {model_path}.")
        else:
            print(f"Found {len(model_paths)} models:")
            for i, path in enumerate(model_paths):
                print(f"[{i}] {path}")
        return model_paths


if __name__ == "__main__":
    """Main entry point for the RayNode."""
    fire.Fire(RayNode)
    # Example usage: python ray_tool.py run_nodes --fun=some_function --args=arg1,arg2
    # This will initialize Ray and run the specified function on all nodes.

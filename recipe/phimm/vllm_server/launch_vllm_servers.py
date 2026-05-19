"""
Launch one in-process vLLM ``LLM`` worker per GPU (TP=1) on a single node.

Each worker process owns its own ``vllm.LLM`` engine (no separate vLLM HTTP
server is spawned anymore — see ``worker_server.py`` for the new design that
calls ``llm.generate`` directly, modelled on
``plugins/qwen35_audio/scripts/run_qwen35_audio_vllm.py``).

Per-GPU layout:
    - One worker process bound to ``CUDA_VISIBLE_DEVICES=<gpu_id>``.
    - Worker exposes a FastAPI app on an auto-selected free port.
    - Worker registers itself with the central FastAPI proxy.

Usage (Hydra config):
    python -m recipe.phimm.vllm_server.launch_vllm_servers \\
        model.local_path=/tmp/models/my-model \\
        cluster.proxy_url=http://proxy-host:8000

    # Override specific params:
    python -m recipe.phimm.vllm_server.launch_vllm_servers \\
        server.max_num_seqs=64 \\
        server.gpu_memory_utilization=0.9
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import hydra
import requests
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("launch_vllm_servers")


def get_node_ip() -> str:
    """Get the IP address of the current node."""
    hostname = socket.gethostname()
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror:
        return "127.0.0.1"


def wait_for_server(host: str, port: int, path: str = "/ready", timeout: float = 600.0) -> bool:
    """Wait until the worker at ``host:port`` responds 200 on ``path``."""
    url = f"http://{host}:{port}{path}"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    return False


def register_with_proxy(proxy_url: str, backend_url: str, max_retries: int = 10) -> bool:
    """Register a backend URL with the FastAPI proxy."""
    for _ in range(max_retries):
        try:
            resp = requests.post(
                f"{proxy_url}/admin/register",
                json={"url": backend_url},
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info("Registered %s with proxy (%s)", backend_url, resp.json())
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    logger.error("Failed to register %s after %d retries", backend_url, max_retries)
    return False


def find_available_port(reserved_ports: set[int]) -> int:
    """Find one currently available TCP port and reserve it in-memory."""
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = int(s.getsockname()[1])
        if port <= 8100:
            continue
        if port in reserved_ports:
            continue
        reserved_ports.add(port)
        return port
    raise RuntimeError("Failed to find an available port")


def launch_worker_server(
    gpu_id: int,
    model_path: str,
    port: int,
    cfg: DictConfig,
) -> subprocess.Popen:
    """Spawn the in-process vLLM worker server bound to a specific GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("VLLM_PLUGINS", "qwen35_audio")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("QWEN35_AUDIO_DISABLE_CUDNN", "1")

    cmd = [
        sys.executable, "-m", "recipe.phimm.vllm_server.worker_server",
        "--port", str(port),
        "--model", model_path,
        "--max-model-len", str(cfg.server.max_model_len),
        "--max-num-seqs", str(cfg.server.max_num_seqs),
        "--gpu-memory-utilization", str(cfg.server.gpu_memory_utilization),
        "--audio-workers", str(cfg.worker.audio_workers),
        "--batch-max-wait-seconds", str(cfg.worker.get("batch_max_wait_seconds", 0.02)),
    ]
    if cfg.server.get("max_num_batched_tokens") is not None:
        cmd.extend(["--max-num-batched-tokens", str(cfg.server.max_num_batched_tokens)])
    if cfg.server.get("enforce_eager", False):
        cmd.append("--enforce-eager")
    if not cfg.server.get("disable_prefix_caching", True):
        cmd.append("--enable-prefix-caching")
    for stop_id in cfg.worker.get("stop_token_ids", []) or []:
        cmd.extend(["--stop-token-id", str(stop_id)])
    if cfg.worker.get("enable_ngram", True):
        cmd.append("--enable-ngram")
        if cfg.worker.get("ngram_size") is not None:
            cmd.extend(["--ngram-size", str(cfg.worker.ngram_size)])
        if cfg.worker.get("ngram_window_size") is not None:
            cmd.extend(["--ngram-window-size", str(cfg.worker.ngram_window_size)])
    else:
        cmd.append("--no-enable-ngram")

    log_file = f"/tmp/worker_{port}.log"
    logger.info("Launching in-process LLM worker on GPU %d, port %d (log: %s)",
                gpu_id, port, log_file)
    fh = open(log_file, "w")
    return subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)


def _wait_and_register(
    gpu_id: int,
    worker_proc: subprocess.Popen,
    host: str,
    worker_port: int,
    proxy_url: str,
    timeout: float,
) -> tuple[int, bool]:
    """Wait for the worker's LLM to finish loading, then register with the proxy."""
    logger.info("GPU %d: waiting for worker on :%d (model load may take minutes) ...",
                gpu_id, worker_port)
    if not wait_for_server(host, worker_port, path="/ready", timeout=timeout):
        logger.error("GPU %d: worker failed to become ready", gpu_id)
        if worker_proc.poll() is not None:
            stdout = worker_proc.stdout.read().decode() if worker_proc.stdout else ""
            logger.error("GPU %d worker exited code %d:\n%s",
                         gpu_id, worker_proc.returncode, stdout[-2000:])
        return gpu_id, False

    backend_url = f"http://{host}:{worker_port}"
    ok = register_with_proxy(proxy_url, backend_url)
    if ok:
        logger.info("GPU %d ready: worker %s registered with proxy", gpu_id, backend_url)
    return gpu_id, ok


def prepare_model_path(cfg_model: DictConfig) -> str:
    """Resolve the model directory the LLM should load.

    Always returns ``cfg_model.local_path``. If ``cfg_model.remote_path`` is
    set and ``local_path`` does not yet exist (or is empty), the remote tree is
    synced to ``local_path`` once using ``bbb sync``.
    """
    from pathlib import Path

    local_path = cfg_model.local_path
    remote_path = cfg_model.get("remote_path")
    if not local_path:
        raise ValueError("model.local_path must be set")

    local = Path(local_path)
    is_cached = local.exists() and any(local.iterdir()) if local.is_dir() else False

    if is_cached:
        logger.info("Using cached model at %s", local_path)
        return local_path

    if not remote_path:
        if not local.exists():
            raise FileNotFoundError(
                f"model.local_path {local_path} does not exist and model.remote_path is unset"
            )
        return local_path

    logger.info("Syncing model %s → %s ...", remote_path, local_path)
    local.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["bbb", "sync", "--concurrency", "64",
         f"{remote_path.rstrip('/')}/", f"{local_path.rstrip('/')}/"],
        check=True,
    )
    logger.info("Model synced to %s", local_path)
    return local_path


def run_servers(cfg: DictConfig) -> None:
    """Main entry point — launch in-process LLM workers from Hydra config."""
    node_ip = get_node_ip()
    num_gpus = cfg.cluster.num_gpus

    # Resolve proxy URL
    if cfg.cluster.proxy_url:
        proxy_url = cfg.cluster.proxy_url
    else:
        proxy_host = cfg.proxy.host
        if proxy_host == "0.0.0.0":
            proxy_host = node_ip
        proxy_url = f"http://{proxy_host}:{cfg.proxy.port}"

    logger.info("Node IP: %s, launching %d in-process LLM workers", node_ip, num_gpus)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # Ensure model is cached locally (workers always load from local_path).
    model_path = prepare_model_path(cfg.model)

    # Auto-select free ports for every worker.
    reserved_ports: set[int] = {int(cfg.proxy.port)}
    worker_ports = [find_available_port(reserved_ports) for _ in range(num_gpus)]
    logger.info("Allocated worker ports: %s", worker_ports)

    worker_procs: list[subprocess.Popen] = []
    for gpu_id in range(num_gpus):
        proc = launch_worker_server(gpu_id, model_path, worker_ports[gpu_id], cfg)
        worker_procs.append(proc)

    def shutdown(_signum, _frame):
        logger.info("Shutting down all worker processes...")
        for proc in worker_procs:
            proc.terminate()
        for proc in worker_procs:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Wait for every worker in PARALLEL and register each as it becomes ready
    startup_timeout = cfg.cluster.startup_timeout
    logger.info("Polling all %d workers concurrently...", num_gpus)
    with ThreadPoolExecutor(max_workers=num_gpus) as pool:
        futures = {
            pool.submit(
                _wait_and_register,
                gpu_id, worker_procs[gpu_id],
                node_ip, worker_ports[gpu_id],
                proxy_url, startup_timeout,
            ): gpu_id
            for gpu_id in range(num_gpus)
        }
        succeeded = 0
        for future in as_completed(futures):
            _, ok = future.result()
            if ok:
                succeeded += 1

    logger.info("%d/%d workers ready. Monitoring...", succeeded, num_gpus)

    # Monitor loop
    try:
        while True:
            for i in range(num_gpus):
                if worker_procs[i].poll() is not None:
                    logger.warning(
                        "GPU %d worker (PID %d) exited code %d",
                        i, worker_procs[i].pid, worker_procs[i].returncode,
                    )
            time.sleep(10)
    except KeyboardInterrupt:
        shutdown(None, None)


@hydra.main(config_path="config", config_name="vllm", version_base=None)
def main(cfg: DictConfig) -> None:
    run_servers(cfg)


if __name__ == "__main__":
    main()

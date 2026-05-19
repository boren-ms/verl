"""
Launch multiple vLLM server instances on a single node (one per GPU with TP=1),
each with a worker sidecar that handles audio loading locally.

Per-GPU architecture:
    - vLLM on an auto-selected available local port
    - Worker sidecar on an auto-selected available local port
  - Worker handles audio loading from blob/local storage, builds chat messages,
    forwards to local vLLM — the proxy only sees the worker port.

The worker sidecar eliminates the client-side audio loading bottleneck:
  Client sends: {"audio_path": "az://...", "prompt": "..."}  (tiny, fast)
  Worker does:  load audio → encode → build chat msg → call vLLM → return result

GPU utilization optimizations (targeting >90%):
- gpu_memory_utilization=0.95: maximize KV cache for larger batches
- max_num_seqs=128: allow more concurrent sequences for continuous batching
- max_num_batched_tokens=65536: increase token budget per iteration
- performance_mode=throughput: optimize scheduler for throughput
- CUDA graphs (no enforce_eager): reduce kernel launch overhead
- Parallel server startup and health polling

Usage (Hydra config):
    python -m recipe.phimm.vllm_server.launch_vllm_servers \\
        model.local_path=/tmp/models/my-model \\
        cluster.proxy_url=http://proxy-host:8000

    # Override specific params:
    python -m recipe.phimm.vllm_server.launch_vllm_servers \\
        server.max_num_seqs=64 \\
        server.gpu_memory_utilization=0.9

    # Use a different config file:
    python -m recipe.phimm.vllm_server.launch_vllm_servers \\
        --config-name=config_librispeech
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


def wait_for_server(host: str, port: int, timeout: float = 600.0) -> bool:
    """Wait until the vLLM server at host:port responds to /health."""
    url = f"http://{host}:{port}/health"
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
    for i in range(max_retries):
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


def allocate_port_pairs(num_gpus: int, reserved_ports: set[int] | None = None) -> list[tuple[int, int]]:
    """Allocate (vLLM_port, worker_port) pairs for each GPU."""
    reserved = set(reserved_ports or set())
    pairs: list[tuple[int, int]] = []
    for _ in range(num_gpus):
        vllm_port = find_available_port(reserved)
        worker_port = find_available_port(reserved)
        pairs.append((vllm_port, worker_port))
    return pairs


def launch_vllm_server(
    gpu_id: int,
    model_path: str,
    port: int,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """Launch a single vLLM server on a specific GPU (internal port)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Enable the qwen35_audio out-of-tree plugin (registers Qwen3_5AudioForCausalLM
    # with vLLM's ModelRegistry and installs the audio multimodal processor).
    # See plugins/qwen35_audio/README.md for the verified stack.
    env.setdefault("VLLM_PLUGINS", "qwen35_audio")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("QWEN35_AUDIO_DISABLE_CUDNN", "1")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--tensor-parallel-size", "1",
        "--trust-remote-code",
        "--load-format", "auto",
        "--dtype", "bfloat16",
        "--disable-log-stats",
        "--limit-mm-per-prompt", '{"audio": 1}',
        "--logits-processors",
        "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor",
    ]
    if extra_args:
        cmd.extend(extra_args)

    log_file = f"/tmp/vllm_{port}.log"
    logger.info("Launching vLLM on GPU %d, internal port %d (log: %s)", gpu_id, port, log_file)
    fh = open(log_file, "w")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
    )
    return proc


def launch_worker_server(
    vllm_port: int,
    worker_port: int,
    model_path: str,
    num_workers: int = 4,
    max_vllm_concurrency: int = 128,
) -> subprocess.Popen:
    """Launch a worker sidecar server that wraps a local vLLM instance."""
    cmd = [
        sys.executable, "-m", "recipe.phimm.vllm_server.worker_server",
        "--vllm-port", str(vllm_port),
        "--port", str(worker_port),
        "--model", model_path,
        "--num-workers", str(num_workers),
        "--max-vllm-concurrency", str(max_vllm_concurrency),
    ]

    log_file = f"/tmp/worker_{worker_port}.log"
    logger.info("Launching worker sidecar on port %d → vLLM port %d (log: %s)", worker_port, vllm_port, log_file)
    fh = open(log_file, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=fh,
        stderr=subprocess.STDOUT,
    )
    return proc


def _wait_and_register(gpu_id: int, vllm_proc: subprocess.Popen, worker_proc: subprocess.Popen,
                       host: str, vllm_port: int, worker_port: int,
                       proxy_url: str, timeout: float) -> tuple[int, bool]:
    """Wait for vLLM + worker to be ready, then register worker with proxy."""
    logger.info("GPU %d: waiting for vLLM on :%d ...", gpu_id, vllm_port)
    if not wait_for_server("localhost", vllm_port, timeout=timeout):
        logger.error("GPU %d: vLLM failed to start", gpu_id)
        if vllm_proc.poll() is not None:
            stdout = vllm_proc.stdout.read().decode() if vllm_proc.stdout else ""
            logger.error("GPU %d vLLM exited code %d:\n%s", gpu_id, vllm_proc.returncode, stdout[-2000:])
        return gpu_id, False

    logger.info("GPU %d: vLLM ready, waiting for worker on :%d ...", gpu_id, worker_port)
    if not wait_for_server(host, worker_port, timeout=60):
        logger.error("GPU %d: worker sidecar failed to start", gpu_id)
        if worker_proc.poll() is not None:
            stdout = worker_proc.stdout.read().decode() if worker_proc.stdout else ""
            logger.error("GPU %d worker exited code %d:\n%s", gpu_id, worker_proc.returncode, stdout[-2000:])
        return gpu_id, False

    # Register the WORKER port (not vLLM port) with the proxy
    url = f"http://{host}:{worker_port}"
    ok = register_with_proxy(proxy_url, url)
    if ok:
        logger.info("GPU %d ready: worker :%d → vLLM :%d, registered with proxy", gpu_id, worker_port, vllm_port)
    return gpu_id, ok


def prepare_model_path(cfg_model: DictConfig) -> str:
    """Resolve the model directory vLLM should load.

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
            raise FileNotFoundError(f"model.local_path {local_path} does not exist and model.remote_path is unset")
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
    """Main entry point — launch vLLM + worker servers from Hydra config."""
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

    logger.info("Node IP: %s, launching %d GPU workers (vLLM + sidecar)", node_ip, num_gpus)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # Ensure model is cached locally (vLLM always loads from local_path).
    model_path = prepare_model_path(cfg.model)

    # Auto-select free ports for every (vLLM, worker) pair.
    reserved_ports = {int(cfg.proxy.port)}
    port_pairs = allocate_port_pairs(num_gpus, reserved_ports=reserved_ports)
    logger.info("Allocated port pairs (vLLM, worker): %s", port_pairs)

    vllm_procs: list[subprocess.Popen] = []
    worker_procs: list[subprocess.Popen] = []

    # Build vLLM extra args from config
    for gpu_id in range(num_gpus):
        vllm_port, worker_port = port_pairs[gpu_id]

        extra = [
            "--gpu-memory-utilization", str(cfg.server.gpu_memory_utilization),
            "--max-model-len", str(cfg.server.max_model_len),
            "--max-num-seqs", str(cfg.server.max_num_seqs),
            "--max-num-batched-tokens", str(cfg.server.max_num_batched_tokens),
            "--performance-mode", cfg.server.performance_mode,
            "--enable-chunked-prefill",
        ]
        if cfg.server.enforce_eager:
            extra.append("--enforce-eager")
        if not cfg.server.disable_prefix_caching:
            extra.append("--enable-prefix-caching")
        chat_template = cfg.server.get("chat_template")
        if chat_template:
            extra.extend(["--chat-template", str(chat_template)])

        vllm_proc = launch_vllm_server(gpu_id, model_path, vllm_port, extra_args=extra)
        worker_proc = launch_worker_server(
            vllm_port, worker_port, model_path,
            num_workers=cfg.worker.audio_workers,
            max_vllm_concurrency=int(cfg.worker.get("max_vllm_concurrency", cfg.server.max_num_seqs)),
        )
        vllm_procs.append(vllm_proc)
        worker_procs.append(worker_proc)

    all_procs = vllm_procs + worker_procs

    def shutdown(signum, frame):
        logger.info("Shutting down all processes...")
        for proc in all_procs:
            proc.terminate()
        for proc in all_procs:
            proc.wait(timeout=30)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Wait for all servers in PARALLEL and register as each becomes ready
    startup_timeout = cfg.cluster.startup_timeout
    logger.info("Polling all %d GPU workers concurrently...", num_gpus)
    with ThreadPoolExecutor(max_workers=num_gpus) as pool:
        futures = {
            pool.submit(
                _wait_and_register,
                gpu_id, vllm_procs[gpu_id], worker_procs[gpu_id],
                node_ip,
                port_pairs[gpu_id][0],
                port_pairs[gpu_id][1],
                proxy_url, startup_timeout,
            ): gpu_id
            for gpu_id in range(num_gpus)
        }
        succeeded = 0
        for future in as_completed(futures):
            gpu_id, ok = future.result()
            if ok:
                succeeded += 1

    logger.info("%d/%d GPU workers ready. Monitoring...", succeeded, num_gpus)

    # Monitor loop
    try:
        while True:
            for i in range(num_gpus):
                if vllm_procs[i].poll() is not None:
                    logger.warning("GPU %d vLLM (PID %d) exited code %d",
                                   i, vllm_procs[i].pid, vllm_procs[i].returncode)
                if worker_procs[i].poll() is not None:
                    logger.warning("GPU %d worker (PID %d) exited code %d",
                                   i, worker_procs[i].pid, worker_procs[i].returncode)
            time.sleep(10)
    except KeyboardInterrupt:
        shutdown(None, None)


@hydra.main(config_path="config", config_name="vllm", version_base=None)
def main(cfg: DictConfig) -> None:
    run_servers(cfg)


if __name__ == "__main__":
    main()

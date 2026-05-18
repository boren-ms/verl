"""
Worker sidecar server — runs alongside each vLLM instance (one per GPU).

Handles the CPU-heavy audio loading and chat message construction on the GPU
node, so the client only needs to send a lightweight audio path. This avoids:
- Slow blob storage reads on the client side
- Large base64-encoded audio payloads over the network
- Client-side CPU bottleneck blocking GPU inference

Architecture (per GPU):
  ┌────────────────────────────────────┐
  │  Worker Server (port 8101)         │
  │  ┌──────────────────────────────┐  │
  │  │ /asr/transcribe              │  │
  │  │  1. Load audio from path     │  │
  │  │  2. Encode to base64         │  │
  │  │  3. Build chat message       │  │
  │  │  4. Forward to local vLLM    │──┼──► vLLM (port 8201)
  │  └──────────────────────────────┘  │
  │  /v1/* → proxy to vLLM             │
  │  /health → proxy to vLLM           │
  └────────────────────────────────────┘

Usage:
    python -m recipe.phimm.vllm_server.worker_server \
        --vllm-port 8201 \
        --port 8101 \
        --model /path/to/model
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parents[3]))

logger = logging.getLogger("worker_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Audio utilities (run in thread pool to avoid blocking the event loop)
# ---------------------------------------------------------------------------

# Configure blobfile: limit retries (default=unlimited!) and reduce timeouts
# to prevent permanent thread stalls when Azure connections go stale.
import blobfile as bf
bf.configure(
    retry_limit=3,           # Fail after 3 retries (default: unlimited = hang forever)
    read_timeout=15,         # 15s read timeout (default: 30s)
    connect_timeout=5,       # 5s connect timeout (default: 10s)
    connection_pool_max_size=8,  # Smaller pool = less stale connection state
)

# Limit concurrent blob reads to prevent Azure HTTP connection pool exhaustion.
# With 16 thread workers per process, allowing all to hit Azure simultaneously
# causes connection pool deadlocks after ~2500 requests. Limiting to 6 concurrent
# blob reads per worker process keeps 8×6=48 total Azure connections (well within limits).
_blob_semaphore = threading.Semaphore(6)


def _load_and_encode_audio(audio_path: str, max_dur: float = 40.0) -> str:
    """Load audio from blob/local path, limit duration, encode to base64 WAV.

    Uses local file cache when available (instant disk reads), falls back to
    blob with semaphore-limited concurrent connections.
    """
    # Try local cache first: az://orngwus2cresco/data/boren/data/X → /root/data/X
    local_path = None
    if audio_path.startswith("az://orngwus2cresco/data/boren/data/"):
        local_path = "/root/data/" + audio_path[len("az://orngwus2cresco/data/boren/data/"):]

    if local_path and os.path.exists(local_path):
        with open(local_path, "rb") as f:
            raw_data = f.read()
    else:
        # Fall back to blob with concurrency limit
        with _blob_semaphore:
            with bf.BlobFile(audio_path, "rb") as f:
                raw_data = f.read()

    audio, sr = sf.read(io.BytesIO(raw_data))

    # Mono conversion
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Limit duration
    if max_dur and len(audio) / sr > max_dur:
        audio = audio[: int(sr * max_dur)]

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _build_chat_messages(prompt_text: str, audio_b64: str) -> list[dict]:
    """Build OpenAI-compatible chat messages with inline audio."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_b64,
                        "format": "wav",
                    },
                },
                {
                    "type": "text",
                    "text": prompt_text,
                },
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class TranscribeRequest(BaseModel):
    audio_path: str = Field(..., description="Path to audio file (local or az:// blob)")
    prompt: str = Field(
        default="Transcribe the audio clip into text.",
        description="ASR prompt text",
    )
    max_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.0)
    max_audio_dur: float = Field(default=40.0, description="Max audio duration in seconds")


# ---------------------------------------------------------------------------
# Worker server factory
# ---------------------------------------------------------------------------

def create_worker_app(vllm_url: str, model_name: str, num_workers: int = 8,
                      max_vllm_concurrency: int = 256) -> FastAPI:
    """Create a FastAPI worker app that wraps a local vLLM instance.

    Args:
        vllm_url: URL of the local vLLM server (e.g. http://localhost:8201)
        model_name: Model name to use in OpenAI API calls
        num_workers: Thread pool size for audio loading
        max_vllm_concurrency: Max concurrent requests to vLLM (prevents event loop saturation)
    """

    app = FastAPI(title="vLLM Worker Server")
    thread_pool = ThreadPoolExecutor(max_workers=num_workers)
    # Limit concurrent requests to vLLM to prevent overwhelming its event loop
    vllm_semaphore = asyncio.Semaphore(max_vllm_concurrency)
    _client: httpx.AsyncClient | None = None
    _stats = {"completed": 0, "active": 0, "errors": 0}

    @app.on_event("startup")
    async def start_monitor():
        async def _monitor():
            while True:
                await asyncio.sleep(10)
                logger.info(
                    "HEARTBEAT: completed=%d active=%d errors=%d pool_threads=%d",
                    _stats["completed"], _stats["active"], _stats["errors"],
                    thread_pool._work_queue.qsize(),
                )
        asyncio.create_task(_monitor())

    async def get_client() -> httpx.AsyncClient:
        nonlocal _client
        if _client is None:
            _client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            )
        return _client

    @app.on_event("shutdown")
    async def shutdown():
        nonlocal _client
        if _client:
            await _client.aclose()
        thread_pool.shutdown(wait=False)

    # ---- Core endpoint: ASR transcription ----

    @app.post("/asr/transcribe")
    async def transcribe(req: TranscribeRequest):
        """Load audio from path, build chat message, forward to local vLLM.

        The heavy audio I/O runs in a thread pool so multiple requests can
        load audio concurrently without blocking each other.
        """
        _stats["active"] += 1
        loop = asyncio.get_event_loop()

        # CPU/IO-bound: load audio in thread pool
        try:
            audio_b64 = await loop.run_in_executor(
                thread_pool, _load_and_encode_audio, req.audio_path, req.max_audio_dur,
            )
        except Exception as e:
            _stats["active"] -= 1
            _stats["errors"] += 1
            raise HTTPException(status_code=400, detail=f"Failed to load audio: {e}")

        # Build chat message
        messages = _build_chat_messages(req.prompt, audio_b64)

        # Forward to local vLLM (rate-limited to prevent event loop saturation)
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }

        client = await get_client()
        async with vllm_semaphore:
            try:
                resp = await client.post(f"{vllm_url}/v1/chat/completions", json=payload)
                _stats["active"] -= 1
                _stats["completed"] += 1
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "application/json"),
                )
            except httpx.ConnectError:
                _stats["active"] -= 1
                _stats["errors"] += 1
                raise HTTPException(status_code=502, detail="Local vLLM server not available")
            except httpx.ReadTimeout:
                _stats["active"] -= 1
                _stats["errors"] += 1
                raise HTTPException(status_code=504, detail="Local vLLM server timed out")

    # ---- Proxy: forward /v1/* to local vLLM ----

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def proxy_v1(request: Request, path: str):
        client = await get_client()
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        try:
            resp = await client.request(
                method=request.method,
                url=f"{vllm_url}/v1/{path}",
                content=body,
                headers=headers,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )
        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            raise HTTPException(status_code=502, detail=str(e))

    # ---- Health: lightweight self-check (does NOT proxy to vLLM) ----
    # This ensures the worker responds to health checks even when vLLM is busy.

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


# ---------------------------------------------------------------------------
# CLI — standalone launch (also used by launch_vllm_servers.py)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="vLLM worker sidecar server")
    parser.add_argument("--vllm-port", type=int, required=True, help="Port of the local vLLM server")
    parser.add_argument("--port", type=int, required=True, help="Port for this worker server")
    parser.add_argument("--model", required=True, help="Model name for API calls")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--num-workers", type=int, default=16, help="Thread pool workers for audio loading")
    parser.add_argument("--max-vllm-concurrency", type=int, default=8,
                        help="Max concurrent requests to local vLLM (prevents event loop saturation)")
    args = parser.parse_args()

    vllm_url = f"http://localhost:{args.vllm_port}"
    # Use full model path as model name (vLLM registers the model with its full path)
    model_name = args.model.rstrip("/")
    app = create_worker_app(vllm_url, model_name, num_workers=args.num_workers,
                            max_vllm_concurrency=args.max_vllm_concurrency)

    logger.info("Worker server on :%d → vLLM on :%d (model=%s)", args.port, args.vllm_port, model_name)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

"""
Worker server — runs one in-process vLLM ``LLM`` engine per GPU.

This replaces the previous architecture (vLLM OpenAI HTTP server + sidecar
proxy). Instead of forwarding chat-completion requests over HTTP and relying
on vLLM's chat-template / multimodal request parsing, we use vLLM's Python
generation API directly — the same pattern as
``plugins/qwen35_audio/scripts/run_qwen35_audio_vllm.py``:

* Audio is loaded from blob/local paths on the worker (co-located with GPU).
* The prompt string is rendered with the model's expected ``<audio>`` /
  chat-special-token format directly — no Jinja chat template is required.
* The raw waveform + sample-rate is passed via ``multi_modal_data={"audio": ...}``.
* ``LLM.generate`` is called with a batch of pending requests (micro-batching)
  to keep the GPU saturated while still giving each HTTP caller a single
  response.

Architecture (per GPU):
  ┌────────────────────────────────────────────────────────────────┐
  │  Worker process (CUDA_VISIBLE_DEVICES=<gpu_id>, port=<port>)   │
  │                                                                │
  │  FastAPI                                                       │
  │   ├─ POST /asr/transcribe ── load audio ──┐                    │
  │   │                                       ▼                    │
  │   │                          ┌─────────────────────────┐       │
  │   │                          │  LLMBatcher (bg thread) │       │
  │   │                          │   gather pending reqs   │       │
  │   │                          │   → llm.generate(batch) │       │
  │   │                          └──────────┬──────────────┘       │
  │   │                                     ▼                      │
  │   │                          ┌─────────────────────────┐       │
  │   │                          │   vllm.LLM (in-process) │       │
  │   │                          └─────────────────────────┘       │
  │   │                                     │                      │
  │   │      ◄────── response text ─────────┘                      │
  │   ├─ GET /health    → {"status": "ok"}                         │
  │   └─ GET /ready     → only OK once the LLM has finished loading│
  └────────────────────────────────────────────────────────────────┘

The proxy / eval client API is unchanged: clients still POST a tiny
``{"audio_path": ..., "prompt": ...}`` payload to ``/asr/transcribe`` and
receive an OpenAI-chat-completions-shaped JSON response so existing
``recipe.phimm.vllm_server.eval_asr`` code keeps working untouched.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import functools
import logging
import os
import queue
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import blobfile as bf
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parents[3]))

from recipe.phimm.utils.audio import load_audio  # noqa: E402

logger = logging.getLogger("worker_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

PROMPT_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
DEFAULT_STOP_TOKEN_IDS = (248044, 248046)

def _load_audio_array(audio_path: str, max_dur: float) -> tuple[np.ndarray, int]:
    """Return a ``(waveform_float32, sample_rate)`` tuple for ``audio_path``."""
    if not audio_path:
        raise ValueError("audio_path must be provided.")
    audio, sr = load_audio({"audio_path": audio_path}, max_dur=max_dur)
    audio = np.asarray(audio, dtype=np.float32)
    return audio, int(sr)


# ---------------------------------------------------------------------------
# LLM batcher — collects pending requests, calls ``llm.generate`` on a thread
# ---------------------------------------------------------------------------

class LLMBatcher:
    """Micro-batches incoming generation requests and runs them on the LLM.

    vLLM's ``LLM.generate(prompts, sampling_params)`` accepts a list of
    prompts and (optionally) a parallel list of ``SamplingParams``, batching
    them together with continuous-batching semantics under the hood. We expose
    a per-request ``submit`` API and let a single background thread drain the
    queue, so multiple concurrent HTTP handlers can share a single GPU efficiently.
    """

    _SENTINEL: tuple = ("__shutdown__",)

    def __init__(
        self,
        llm,
        *,
        max_batch_size: int,
        max_wait_seconds: float,
    ) -> None:
        self.llm = llm
        self.max_batch_size = max(int(max_batch_size), 1)
        self.max_wait_seconds = max(float(max_wait_seconds), 0.0)
        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name="LLMBatcher", daemon=True)
        self._thread.start()

    def submit(self, prompt: str, mm_audio, sampling_params) -> concurrent.futures.Future:
        if self._stopping:
            raise RuntimeError("LLMBatcher is shutting down")
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((prompt, mm_audio, sampling_params, fut))
        return fut

    def shutdown(self) -> None:
        self._stopping = True
        self._queue.put(self._SENTINEL)
        self._thread.join(timeout=30)

    def _gather_batch(self) -> list[tuple]:
        items: list[tuple] = []
        try:
            first = self._queue.get()
        except Exception:
            return items
        if first is self._SENTINEL:
            items.append(first)
            return items
        items.append(first)

        deadline = time.monotonic() + self.max_wait_seconds
        while len(items) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                nxt = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if nxt is self._SENTINEL:
                # Re-queue so the next loop iteration can see it after we
                # finish the current batch.
                self._queue.put(self._SENTINEL)
                break
            items.append(nxt)
        return items

    def _run(self) -> None:
        logger.info("LLMBatcher started (max_batch=%d, max_wait=%.3fs)",
                    self.max_batch_size, self.max_wait_seconds)
        while True:
            batch = self._gather_batch()
            if not batch:
                continue
            if batch[0] is self._SENTINEL:
                logger.info("LLMBatcher received shutdown sentinel")
                return

            prompts_payload = [
                {"prompt": p, "multi_modal_data": {"audio": [mm]}}
                for p, mm, _, _ in batch
            ]
            params_list = [sp for _, _, sp, _ in batch]
            futures = [fut for _, _, _, fut in batch]

            t0 = time.time()
            try:
                outputs = self.llm.generate(
                    prompts_payload,
                    sampling_params=params_list,
                    use_tqdm=False,
                )
            except Exception as e:
                logger.exception("LLM.generate failed on batch of %d", len(batch))
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e)
                continue
            elapsed = time.time() - t0

            for fut, out in zip(futures, outputs):
                if fut.done():
                    continue
                try:
                    text = out.outputs[0].text
                except (AttributeError, IndexError) as e:
                    fut.set_exception(e)
                    continue
                fut.set_result(text)

            logger.info(
                "Batch %d done in %.2fs (%.1f req/s)",
                len(batch), elapsed, len(batch) / max(elapsed, 1e-3),
            )


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class TranscribeRequest(BaseModel):
    audio_path: str = Field(
        ...,
        description=(
            "Path to audio (local, az:// blob, or chunk spec 'file:count:index')."
        ),
    )
    prompt: str = Field(
        default="Transcribe the audio clip into text.",
        description="ASR prompt text (will be wrapped in the model's chat format).",
    )
    max_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.0)
    max_audio_dur: float = Field(default=40.0, description="Max audio duration in seconds")


# ---------------------------------------------------------------------------
# Worker app factory
# ---------------------------------------------------------------------------

def _build_llm(
    *,
    model_path: str,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    enable_prefix_caching: bool,
    max_num_batched_tokens: int | None,
):
    from vllm import LLM

    logger.info(
        "Loading vLLM LLM from %s (max_model_len=%d, max_num_seqs=%d, gpu_mem=%.2f)",
        model_path, max_model_len, max_num_seqs, gpu_memory_utilization,
    )
    kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        load_format="auto",
        dtype="bfloat16",
        tensor_parallel_size=1,
        limit_mm_per_prompt={"audio": 1},
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        enable_prefix_caching=enable_prefix_caching,
    )
    if max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
    t0 = time.time()
    llm = LLM(**kwargs)
    logger.info("LLM loaded in %.1fs", time.time() - t0)
    return llm


def create_worker_app(
    *,
    model_path: str,
    max_model_len: int = 4096,
    max_num_seqs: int = 256,
    gpu_memory_utilization: float = 0.95,
    enforce_eager: bool = False,
    enable_prefix_caching: bool = False,
    max_num_batched_tokens: int | None = None,
    audio_workers: int = 8,
    batch_max_wait_seconds: float = 0.02,
    stop_token_ids: tuple[int, ...] | None = None,
) -> FastAPI:
    """Build a FastAPI app that owns an in-process ``vllm.LLM`` engine."""

    from vllm import SamplingParams

    stop_token_ids = tuple(stop_token_ids or DEFAULT_STOP_TOKEN_IDS)

    app = FastAPI(title="vLLM Worker (in-process LLM)")
    thread_pool = ThreadPoolExecutor(max_workers=audio_workers)
    _stats = {"completed": 0, "active": 0, "errors": 0}
    _state: dict = {"llm": None, "batcher": None, "ready": False}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        loop = asyncio.get_event_loop()
        llm = await loop.run_in_executor(
            None,
            functools.partial(
                _build_llm,
                model_path=model_path,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=enforce_eager,
                enable_prefix_caching=enable_prefix_caching,
                max_num_batched_tokens=max_num_batched_tokens,
            ),
        )
        batcher = LLMBatcher(
            llm,
            max_batch_size=max_num_seqs,
            max_wait_seconds=batch_max_wait_seconds,
        )
        _state["llm"] = llm
        _state["batcher"] = batcher
        _state["ready"] = True
        logger.info("Worker ready (model=%s)", model_path)

        async def _monitor():
            while True:
                await asyncio.sleep(10)
                logger.info(
                    "HEARTBEAT: completed=%d active=%d errors=%d",
                    _stats["completed"], _stats["active"], _stats["errors"],
                )
        monitor_task = asyncio.create_task(_monitor())
        try:
            yield
        finally:
            monitor_task.cancel()
            _state["ready"] = False
            try:
                batcher.shutdown()
            except Exception:
                logger.exception("Failed to cleanly shut down LLMBatcher")
            thread_pool.shutdown(wait=False)

    app.router.lifespan_context = lifespan

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail="LLM not yet loaded")
        return {"status": "ready"}

    @app.post("/asr/transcribe")
    async def transcribe(req: TranscribeRequest):
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail="LLM not yet loaded")

        _stats["active"] += 1
        loop = asyncio.get_event_loop()
        request_id = uuid.uuid4().hex[:12]

        try:
            audio, sr = await loop.run_in_executor(
                thread_pool,
                functools.partial(
                    _load_audio_array,
                    audio_path=req.audio_path,
                    max_dur=req.max_audio_dur,
                ),
            )
        except Exception as e:
            _stats["active"] -= 1
            _stats["errors"] += 1
            raise HTTPException(status_code=400, detail=f"Failed to load audio: {e}") from e

        prompt = PROMPT_TEMPLATE.format(prompt=req.prompt)
        sampling_params = SamplingParams(
            temperature=float(req.temperature),
            max_tokens=int(req.max_tokens),
            stop_token_ids=list(stop_token_ids),
            repetition_penalty=1.0,
        )

        try:
            fut = _state["batcher"].submit(prompt, (audio, sr), sampling_params)
            text = await asyncio.wrap_future(fut)
        except Exception as e:
            _stats["active"] -= 1
            _stats["errors"] += 1
            logger.exception("Generation failed for request %s", request_id)
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}") from e

        _stats["active"] -= 1
        _stats["completed"] += 1

        return {"text": text}

    return app


# ---------------------------------------------------------------------------
# CLI — standalone launch (also used by launch_vllm_servers.py)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM in-process worker server")
    parser.add_argument("--port", type=int, required=True, help="Port for this worker server")
    parser.add_argument("--model", required=True, help="Path to local model directory")
    parser.add_argument("--host", default="0.0.0.0")

    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")

    parser.add_argument("--audio-workers", type=int, default=8,
                        help="Thread pool size for audio loading (CPU/IO bound).")
    parser.add_argument("--batch-max-wait-seconds", type=float, default=0.02,
                        help="Micro-batch coalescing window for LLM.generate calls.")
    parser.add_argument("--stop-token-id", action="append", type=int, default=None,
                        help="Override stop token id(s) for SamplingParams (repeatable).")

    args = parser.parse_args()

    # Mirror the qwen35_audio plugin smoke-test environment so the in-process
    # LLM picks up the right model architecture & avoids the cuDNN attention
    # path that has been observed to crash on H100s.
    os.environ.setdefault("VLLM_PLUGINS", "qwen35_audio")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("QWEN35_AUDIO_DISABLE_CUDNN", "1")

    app = create_worker_app(
        model_path=args.model.rstrip("/"),
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.enable_prefix_caching,
        max_num_batched_tokens=args.max_num_batched_tokens,
        audio_workers=args.audio_workers,
        batch_max_wait_seconds=args.batch_max_wait_seconds,
        stop_token_ids=tuple(args.stop_token_id) if args.stop_token_id else None,
    )

    logger.info("Worker server on :%d (model=%s)", args.port, args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

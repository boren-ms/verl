"""
FastAPI proxy that load-balances OpenAI-compatible requests across multiple vLLM backends.

Each vLLM backend runs `vllm serve` with TP=1 on a single GPU. The proxy accepts
requests on a single endpoint and distributes them to all registered backends
using **least-connections** routing for optimal GPU utilization (>90%).

Dynamic registration: New vLLM servers can register at any time via POST
/admin/register. The proxy immediately includes them in the load-balancing
pool. Backends that fail health checks are temporarily excluded but kept
in the registry so they rejoin automatically once healthy again.

GPU utilization optimizations:
- Least-connections routing: sends requests to the GPU with fewest in-flight
  requests, preventing idle GPUs while others queue.
- Zero-copy forwarding: forwards raw response bytes without JSON parse/reserialize.
- High-concurrency httpx pool: 1000 connections to sustain heavy request loads.
- Concurrent health checks: all backends pinged in parallel.

Usage:
    python -m recipe.phimm.vllm_server.fastapi_proxy --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
import uvicorn

logger = logging.getLogger("vllm_proxy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Backend registry — least-connections routing for max GPU utilization
# ---------------------------------------------------------------------------

class BackendRegistry:
    """Registry of vLLM backend URLs with least-connections load balancing.

    Supports dynamic registration/unregistration at any time. Routes each
    request to the healthy backend with the fewest in-flight requests,
    ensuring all GPUs stay busy even when request latencies vary.
    """

    def __init__(self, max_inflight_per_backend: int = 4):
        self._backends: list[str] = []  # e.g. ["http://10.0.0.1:8001", ...]
        self._healthy: dict[str, bool] = {}
        self._pending: dict[str, int] = {}  # url -> in-flight request count
        self._lock = asyncio.Lock()
        self._counter: int = 0  # fallback tiebreaker for round-robin among equal-load backends
        self._changed = asyncio.Event()  # signalled on every register/unregister
        # Per-backend concurrency cap; cluster capacity = cap * healthy_count.
        self.max_inflight_per_backend: int = max_inflight_per_backend

    async def register(self, url: str) -> dict:
        """Register a backend. Idempotent — registering the same URL twice is a no-op."""
        async with self._lock:
            is_new = url not in self._backends
            if is_new:
                self._backends.append(url)
                self._healthy[url] = True
                self._pending[url] = 0
                self._changed.set()
                logger.info("Registered backend %s (total: %d)", url, len(self._backends))
            return {"url": url, "is_new": is_new, "total": len(self._backends)}

    async def register_batch(self, urls: list[str]) -> dict:
        """Register multiple backends atomically."""
        async with self._lock:
            added = []
            for url in urls:
                if url not in self._backends:
                    self._backends.append(url)
                    self._healthy[url] = True
                    self._pending[url] = 0
                    added.append(url)
            if added:
                self._changed.set()
                logger.info("Batch-registered %d backends: %s (total: %d)", len(added), added, len(self._backends))
            return {"added": added, "total": len(self._backends)}

    async def unregister(self, url: str) -> dict:
        """Remove a backend from the pool."""
        async with self._lock:
            removed = url in self._backends
            if removed:
                self._backends.remove(url)
                self._healthy.pop(url, None)
                self._pending.pop(url, None)
                self._changed.set()
                logger.info("Unregistered backend %s (total: %d)", url, len(self._backends))
            return {"url": url, "removed": removed, "total": len(self._backends)}

    async def next_backend(self) -> str:
        """Pick the healthy backend with the fewest in-flight requests.

        Ties are broken by round-robin counter to spread load evenly when
        multiple backends have the same number of pending requests.
        """
        async with self._lock:
            n = len(self._backends)
            if n == 0:
                raise HTTPException(status_code=503, detail="No backends registered")

            # Find all healthy backends
            healthy = [(url, self._pending.get(url, 0)) for url in self._backends
                       if self._healthy.get(url, False)]
            if not healthy:
                raise HTTPException(
                    status_code=503,
                    detail=f"No healthy backends (total={n})",
                )

            # Pick the one with fewest in-flight; break ties with counter
            min_pending = min(p for _, p in healthy)
            candidates = [url for url, p in healthy if p == min_pending]
            idx = self._counter % len(candidates)
            self._counter += 1
            chosen = candidates[idx]
            self._pending[chosen] = self._pending.get(chosen, 0) + 1
            return chosen

    async def release_backend(self, url: str):
        """Decrement in-flight counter after a request completes."""
        async with self._lock:
            if url in self._pending:
                self._pending[url] = max(0, self._pending[url] - 1)

    async def list_backends(self) -> list[dict]:
        async with self._lock:
            return [
                {
                    "url": b,
                    "healthy": self._healthy.get(b, False),
                    "in_flight": self._pending.get(b, 0),
                }
                for b in self._backends
            ]

    async def set_health(self, url: str, healthy: bool):
        async with self._lock:
            was_healthy = self._healthy.get(url)
            if url in self._healthy:
                self._healthy[url] = healthy
            if was_healthy != healthy:
                state = "UP" if healthy else "DOWN"
                logger.info("Backend %s is now %s", url, state)

    async def healthy_count(self) -> tuple[int, int]:
        """Return (healthy, total) counts."""
        async with self._lock:
            healthy = sum(1 for v in self._healthy.values() if v)
            return healthy, len(self._backends)

    async def load_snapshot(self) -> dict:
        """Return current load/capacity snapshot for readiness reporting."""
        async with self._lock:
            healthy = sum(1 for v in self._healthy.values() if v)
            total = len(self._backends)
            in_flight = sum(
                self._pending.get(u, 0) for u in self._backends if self._healthy.get(u, False)
            )
            capacity = healthy * self.max_inflight_per_backend
            return {
                "healthy": healthy,
                "total": total,
                "in_flight": in_flight,
                "capacity": capacity,
                "max_inflight_per_backend": self.max_inflight_per_backend,
            }

    async def try_acquire_backend(self) -> str | None:
        """Like next_backend(), but returns None if all healthy backends are at capacity.

        Routes to the healthy backend with the fewest in-flight requests and
        only admits the request when that backend is still under the per-backend cap.
        """
        async with self._lock:
            healthy = [
                (url, self._pending.get(url, 0))
                for url in self._backends
                if self._healthy.get(url, False)
            ]
            if not healthy:
                return None
            min_pending = min(p for _, p in healthy)
            if min_pending >= self.max_inflight_per_backend:
                return None
            candidates = [url for url, p in healthy if p == min_pending]
            idx = self._counter % len(candidates)
            self._counter += 1
            chosen = candidates[idx]
            self._pending[chosen] = self._pending.get(chosen, 0) + 1
            return chosen

    async def wait_for_backends(self, min_count: int, timeout: float = 600.0) -> bool:
        """Block until at least min_count healthy backends are registered."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            healthy, total = await self.healthy_count()
            if healthy >= min_count:
                return True
            self._changed.clear()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=min(remaining, 5.0))
            except asyncio.TimeoutError:
                pass
        return False


registry = BackendRegistry()

# ---------------------------------------------------------------------------
# Health checker — concurrent checks, auto-recovers backends
# ---------------------------------------------------------------------------

async def health_check_loop(interval: float = 10.0):
    """Periodically ping each backend's /health endpoint concurrently.

    All backends are checked in parallel to minimize check duration.
    Backends that were marked unhealthy are automatically restored when
    they start responding again. Uses a generous timeout since workers
    may be busy with audio loading.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            backends = await registry.list_backends()
            if backends:
                async def _check(url: str):
                    try:
                        resp = await client.get(f"{url}/health")
                        await registry.set_health(url, resp.status_code == 200)
                    except httpx.ConnectError:
                        # Only mark unhealthy on connection refusal (process died)
                        await registry.set_health(url, False)
                    except (httpx.ReadTimeout, httpx.TimeoutException):
                        # Timeout likely means worker is busy, don't mark unhealthy
                        pass
                await asyncio.gather(*[_check(b["url"]) for b in backends])
            await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(health_check_loop(interval=10.0))
    yield
    task.cancel()
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None

app = FastAPI(title="vLLM Multi-Node Proxy", lifespan=lifespan)

_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        # High-concurrency pool: 1000 connections to sustain heavy request loads
        limits = httpx.Limits(
            max_connections=4096,
            max_keepalive_connections=200,
            keepalive_expiry=30.0,
        )
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=60.0),
            limits=limits,
        )
    return _client


# ---- Admin endpoints (always available for dynamic management) ----

@app.post("/admin/register")
async def register_backend(request: Request):
    """Register one or more backends. Accepts {"url": "..."} or {"urls": [...]}.
    Can be called at any time — new backends immediately join the load-balancing pool."""
    body = await request.json()

    urls = body.get("urls", [])
    if not urls and body.get("url"):
        urls = [body["url"]]
    if not urls:
        raise HTTPException(status_code=400, detail="Provide 'url' (string) or 'urls' (list)")

    urls = [u.rstrip("/") for u in urls]
    if len(urls) == 1:
        result = await registry.register(urls[0])
    else:
        result = await registry.register_batch(urls)
    return {"status": "ok", **result}


@app.post("/admin/unregister")
async def unregister_backend(request: Request):
    """Remove a backend from the pool."""
    body = await request.json()
    url = body.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url' field")
    result = await registry.unregister(url.rstrip("/"))
    return {"status": "ok", **result}


@app.get("/admin/backends")
async def list_backends():
    """List all registered backends with health and in-flight status."""
    return await registry.list_backends()


@app.get("/admin/wait")
async def wait_for_backends(min_backends: int = 1, timeout: float = 300.0):
    """Block until at least min_backends healthy backends are available.
    Useful for clients that need to wait for servers to come online."""
    ok = await registry.wait_for_backends(min_backends, timeout=timeout)
    healthy, total = await registry.healthy_count()
    if not ok:
        raise HTTPException(
            status_code=503,
            detail=f"Timed out waiting for {min_backends} backends (healthy={healthy}, total={total})",
        )
    return {"status": "ok", "healthy": healthy, "total": total}


@app.get("/health")
async def proxy_health():
    backends = await registry.list_backends()
    healthy = sum(1 for b in backends if b["healthy"])
    return {"status": "ok", "healthy_backends": healthy, "total_backends": len(backends)}


@app.get("/ready")
async def proxy_ready():
    """Readiness probe: 200 only when at least 1 backend is healthy AND the
    cluster is not overloaded (total in-flight < capacity). Returns 503 with
    a structured reason otherwise so clients can back off."""
    snap = await registry.load_snapshot()
    if snap["healthy"] < 1:
        raise HTTPException(
            status_code=503,
            detail={"reason": "no_healthy_backends", **snap},
        )
    if snap["in_flight"] >= snap["capacity"]:
        raise HTTPException(
            status_code=503,
            detail={"reason": "overloaded", **snap},
        )
    return {"status": "ready", **snap}


# ---- Proxy forwarding: zero-copy with retry ----

async def _forward_request(request: Request, path: str, max_retries: int = 5):
    """Forward a request to the least-loaded healthy backend.

    Uses zero-copy forwarding: raw response bytes are passed through without
    JSON deserialization/reserialization, eliminating proxy CPU overhead.
    On failure, marks backend unhealthy and retries on the next-least-loaded.
    Applies per-backend concurrency limits; returns 503 with Retry-After when
    the cluster is at capacity so callers can back off.
    """
    # Admission control: bounce immediately when overloaded rather than queueing
    # unbounded work that would also keep /ready stuck in overload.
    backend_url = await registry.try_acquire_backend()
    if backend_url is None:
        snap = await registry.load_snapshot()
        reason = "overloaded" if snap["healthy"] >= 1 else "no_healthy_backends"
        return Response(
            content=f'{{"error":"{reason}","in_flight":{snap["in_flight"]},"capacity":{snap["capacity"]}}}',
            status_code=503,
            media_type="application/json",
            headers={"Retry-After": "1"},
        )

    client = await get_client()
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

    last_error = None
    for attempt in range(1 + max_retries):
        target_url = f"{backend_url}/{path}"
        try:
            resp = await client.request(
                method=request.method,
                url=target_url,
                content=body,
                headers=headers,
            )
            await registry.release_backend(backend_url)
            # Zero-copy: forward raw bytes and content-type directly
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )
        except httpx.ConnectError as e:
            await registry.set_health(backend_url, False)
            await registry.release_backend(backend_url)
            last_error = e
            logger.warning("Backend %s connect failed (attempt %d/%d): %s",
                           backend_url, attempt + 1, 1 + max_retries, e)
        except httpx.ReadTimeout as e:
            await registry.release_backend(backend_url)
            last_error = e
            logger.warning("Backend %s read timeout (attempt %d/%d): %s",
                           backend_url, attempt + 1, 1 + max_retries, e)

        # Retry on next backend if still under capacity; otherwise give up.
        next_url = await registry.try_acquire_backend()
        if next_url is None:
            break
        backend_url = next_url

    raise HTTPException(status_code=502, detail=f"All retries exhausted: {last_error}")


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy_v1(request: Request, path: str):
    return await _forward_request(request, f"v1/{path}")


@app.post("/asr/transcribe")
async def proxy_asr_transcribe(request: Request):
    """Forward single or batched transcription payloads to workers."""
    return await _forward_request(request, "asr/transcribe")


@app.api_route("/asr/{path:path}", methods=["GET", "POST"])
async def proxy_asr(request: Request, path: str):
    """Forward ASR requests to worker sidecars (audio loading happens on workers)."""
    return await _forward_request(request, f"asr/{path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="vLLM multi-node proxy")
    parser.add_argument("--host", default="0.0.0.0", help="Proxy listen host")
    parser.add_argument("--port", type=int, default=8000, help="Proxy listen port")
    parser.add_argument("--backends", nargs="*", default=[], help="Pre-register backend URLs")
    parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn workers (1 recommended for shared registry)")
    parser.add_argument("--max-inflight-per-backend", type=int, default=64,
                        help="Per-backend in-flight request cap; cluster capacity = cap * healthy_backends. "
                             "When exceeded, /ready returns 503 (overloaded) and new requests get 503 + Retry-After.")
    args = parser.parse_args()

    registry.max_inflight_per_backend = args.max_inflight_per_backend

    # Pre-register backends if provided
    async def _pre_register():
        for url in args.backends:
            await registry.register(url.rstrip("/"))

    if args.backends:
        asyncio.run(_pre_register())

    logger.info("Starting vLLM proxy on %s:%d (workers=%d)", args.host, args.port, args.workers)
    logger.info("Routing: least-connections | Register: POST /admin/register")
    uvicorn.run(
        "recipe.phimm.vllm_server.fastapi_proxy:app",
        host=args.host,
        port=args.port,
        log_level="info",
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

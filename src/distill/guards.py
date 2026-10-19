"""ASGI guards for the router API: security headers + no stack traces, body size cap, per-client rate limit,
and admission control (bounded concurrency + bounded queue, then 429 + Retry-After).

Why admission control: the local vLLM sweep collapses above c=64 (p95 0.5 s -> 4.8 s at c=128, request errors
at c=128/256), most likely in the Windows -> WSL2 port forward rather than on the GPU. Whatever the cause,
unbounded concurrency turns overload into timeouts for everyone. Capping in-flight work per instance and
rejecting the excess quickly with 429 + Retry-After keeps latency for admitted requests near the knee and lets
clients (and Cloud Run's autoscaler) back off. Limits are per instance; Cloud Run's
``max_instance_request_concurrency`` bounds what one instance can receive.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

log = logging.getLogger("distill.guards")

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

SECURITY_HEADERS: list[tuple[bytes, bytes]] = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'"),
    (b"cache-control", b"no-store"),
    (b"cross-origin-resource-policy", b"same-origin"),
    (b"strict-transport-security", b"max-age=31536000"),
]


async def send_json(
    send: Send, status: int, body: dict[str, Any], extra_headers: list[tuple[bytes, bytes]] | None = None
) -> None:
    payload = json.dumps(body).encode("utf-8")
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": headers + (extra_headers or [])})
    await send({"type": "http.response.body", "body": payload})


def _header(scope: Scope, name: bytes) -> bytes | None:
    for k, v in scope.get("headers") or []:
        if k.lower() == name:
            return bytes(v)
    return None


class SecurityHeaders:
    """Outermost guard: hardening headers on every response; unhandled errors -> generic 500, no trace."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def wrapped(msg: Message) -> None:
            nonlocal started
            if msg["type"] == "http.response.start":
                started = True
                hdrs = [(k, v) for k, v in msg.get("headers", []) if k.lower() != b"server"]
                present = {k.lower() for k, _ in hdrs}
                hdrs += [(k, v) for k, v in SECURITY_HEADERS if k not in present]
                msg["headers"] = hdrs
            await send(msg)

        try:
            await self.app(scope, receive, wrapped)
        except Exception:
            # Log type + redacted trace server-side (JSON logs); the client gets no internals.
            log.exception("unhandled_error", extra={"path": scope.get("path", "")})
            if not started:
                await send_json(wrapped, 500, {"detail": "internal error"})


class BodySizeLimit:
    """Reject request bodies over ``max_bytes`` with 413 (checks Content-Length and the streamed size)."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return
        cl = _header(scope, b"content-length")
        if cl is not None:
            try:
                if int(cl) > self.max_bytes:
                    await send_json(send, 413, {"detail": "request body too large"})
                    return
            except ValueError:
                await send_json(send, 400, {"detail": "bad content-length"})
                return
        chunks: list[bytes] = []
        size = 0
        while True:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            chunk = msg.get("body", b"")
            size += len(chunk)
            if size > self.max_bytes:
                await send_json(send, 413, {"detail": "request body too large"})
                return
            chunks.append(chunk)
            if not msg.get("more_body"):
                break
        body = b"".join(chunks)
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def client_key(scope: Scope) -> str:
    """Rate-limit identity: hash of the credential if present (API key / bearer), else the peer address.

    Behind Cloud Run every caller holds an IAM identity token, so the credential is the stable identity; the
    peer address is a fallback for unauthenticated local use only.
    """
    cred = _header(scope, b"x-api-key") or _header(scope, b"authorization")
    if cred:
        return "k:" + hashlib.sha256(cred).hexdigest()[:16]
    client = scope.get("client") or ("unknown", 0)
    return f"ip:{client[0]}"


class RateLimit:
    """Per-client token bucket on /v1/* (refill ``rps``, capacity ``burst``) -> 429 + Retry-After."""

    def __init__(
        self,
        app: ASGIApp,
        rps: float,
        burst: int,
        *,
        prefix: str = "/v1/",
        max_clients: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.app = app
        self.rps = rps
        self.burst = burst
        self.prefix = prefix
        self.max_clients = max_clients
        self.clock = clock
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()

    def _take(self, key: str) -> float:
        """Consume one token; return 0 if allowed, else seconds until a token is available."""
        now = self.clock()
        tokens, last = self._buckets.pop(key, (float(self.burst), now))
        tokens = min(float(self.burst), tokens + (now - last) * self.rps)
        wait = 0.0
        if tokens >= 1.0:
            tokens -= 1.0
        else:
            wait = (1.0 - tokens) / self.rps
        self._buckets[key] = (tokens, now)
        while len(self._buckets) > self.max_clients:
            self._buckets.popitem(last=False)
        return wait

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.rps <= 0 or not str(scope.get("path", "")).startswith(self.prefix):
            await self.app(scope, receive, send)
            return
        wait = self._take(client_key(scope))
        if wait > 0:
            retry = str(max(1, math.ceil(wait))).encode()
            log.info("rate_limited", extra={"retry_after_s": int(retry)})
            await send_json(send, 429, {"detail": "rate limit exceeded"}, [(b"retry-after", retry)])
            return
        await self.app(scope, receive, send)


class AdmissionControl:
    """Bound in-flight /v1/* work per instance; queue a bounded number of waiters; shed the rest with 429."""

    def __init__(
        self,
        app: ASGIApp,
        max_inflight: int,
        max_queue: int,
        queue_timeout_s: float,
        *,
        prefix: str = "/v1/",
        retry_after_s: int = 1,
    ) -> None:
        self.app = app
        self.max_inflight = max_inflight
        self.max_queue = max_queue
        self.queue_timeout_s = queue_timeout_s
        self.prefix = prefix
        self.retry_after = str(retry_after_s).encode()
        self.inflight = 0
        self.waiting = 0
        self.shed = 0
        self._sem: asyncio.Semaphore | None = None

    async def _reject(self, send: Send, why: str) -> None:
        self.shed += 1
        log.warning("request_shed", extra={"why": why, "inflight": self.inflight, "waiting": self.waiting})
        hdrs = [(b"retry-after", self.retry_after)]
        await send_json(send, 429, {"detail": "overloaded, retry later"}, hdrs)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not str(scope.get("path", "")).startswith(self.prefix):
            await self.app(scope, receive, send)
            return
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_inflight)
        sem = self._sem
        if sem.locked():
            if self.waiting >= self.max_queue:
                await self._reject(send, "queue_full")
                return
            self.waiting += 1
            try:
                await asyncio.wait_for(sem.acquire(), timeout=self.queue_timeout_s)
            except TimeoutError:
                await self._reject(send, "queue_timeout")
                return
            finally:
                self.waiting -= 1
        else:
            await sem.acquire()
        self.inflight += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.inflight -= 1
            sem.release()

    def snapshot(self) -> dict[str, int]:
        return {
            "inflight": self.inflight,
            "waiting": self.waiting,
            "shed_total": self.shed,
            "max_inflight": self.max_inflight,
            "max_queue": self.max_queue,
        }

"""Student client: the fine-tuned Qwen2.5 served by vLLM's OpenAI-compatible API.

Confidence = joint probability of the generated label tokens (exp of the summed token logprobs) returned
by vLLM. The output is untrusted: anything outside the 77-label enum is an invalid output.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import httpx

from distill.labels import STUDENT_SYSTEM, canonical_label, clean_query


@dataclass
class StudentResult:
    label: str | None  # None = invalid output (not in the enum)
    raw: str
    confidence: float
    latency_s: float
    completion_tokens: int = 0


def student_messages(query: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": clean_query(query)}]


def parse_completion(data: dict[str, Any]) -> tuple[str, float, int]:
    """Return (raw text, joint label probability, completion tokens) from a chat-completions payload."""
    choice = (data.get("choices") or [{}])[0]
    raw = str((choice.get("message") or {}).get("content") or "")
    content_lp = ((choice.get("logprobs") or {}).get("content")) or []
    total = 0.0
    for tok in content_lp:
        lp = tok.get("logprob")
        if isinstance(lp, int | float) and math.isfinite(lp):
            total += float(lp)
        else:
            total += -1e9
    conf = math.exp(total) if content_lp else 0.0
    usage = data.get("usage") or {}
    return raw, conf, int(usage.get("completion_tokens", 0) or 0)


# Fixed metadata-server endpoint (never user-controlled -> no SSRF surface).
_METADATA_IDENTITY_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity"
)


class GcpIdTokenSource:
    """Google-signed ID token for service-to-service calls on Cloud Run (vLLM service is IAM-gated)."""

    def __init__(self, audience: str, client: httpx.AsyncClient) -> None:
        self.audience = audience
        self._client = client
        self._token = ""
        self._expires = 0.0

    async def token(self) -> str:
        if time.monotonic() < self._expires:
            return self._token
        r = await self._client.get(
            _METADATA_IDENTITY_URL,
            params={"audience": self.audience},
            headers={"Metadata-Flavor": "Google"},
            timeout=3.0,
        )
        r.raise_for_status()
        self._token = r.text.strip()
        self._expires = time.monotonic() + 45 * 60  # tokens live 60 min
        return self._token


class StudentClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 10.0,
        client: httpx.AsyncClient | None = None,
        gcp_id_token: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s)
        root = self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url
        self._root = root
        self._auth = GcpIdTokenSource(root, self._client) if gcp_id_token else None

    async def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {await self._auth.token()}"} if self._auth else {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def classify(self, query: str) -> StudentResult:
        body = {
            "model": self.model,
            "messages": student_messages(query),
            "temperature": 0,
            "max_tokens": 24,
            "logprobs": True,
            "top_logprobs": 1,
        }
        t0 = time.perf_counter()
        resp = await self._client.post(
            self.base_url + "/chat/completions", json=body, headers=await self._headers()
        )
        resp.raise_for_status()
        latency = time.perf_counter() - t0
        raw, conf, ntok = parse_completion(resp.json())
        return StudentResult(canonical_label(raw), raw, conf, latency, ntok)

    async def ready(self) -> bool:
        """vLLM exposes /health at the server root (outside /v1)."""
        try:
            r = await self._client.get(self._root + "/health", timeout=2.0, headers=await self._headers())
            return r.status_code == 200
        except httpx.HTTPError:
            return False

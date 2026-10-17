"""Router API: POST /v1/classify -> student (vLLM) first, teacher (Groq) on low confidence / invalid output.

/healthz = liveness (process up). /readyz = dependencies (student server reachable; teacher key present when
escalation is enabled). Graceful shutdown: uvicorn drains on SIGTERM, then the lifespan closes clients.
"""

from __future__ import annotations

import hmac
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from distill.costs import load_pricing
from distill.labels import MAX_QUERY_CHARS
from distill.logs import query_fingerprint, setup_logging
from distill.router import Router
from distill.settings import PROJECT, Settings, get_settings
from distill.spend import SpendLedger
from distill.student import StudentClient
from distill.teacher import GroqTeacher

log = logging.getLogger("distill.api")


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class ClassifyResponse(BaseModel):
    label: str | None
    source: str
    reason: str
    student_confidence: float | None
    latency_ms: float


def build_router(s: Settings) -> tuple[Router, StudentClient, GroqTeacher | None]:
    student = StudentClient(
        s.student_base_url, s.student_model, s.student_timeout_s, gcp_id_token=s.student_gcp_id_token
    )
    teacher: GroqTeacher | None = None
    if s.teacher_enabled and s.groq_api_key is not None:
        price, _, _ = load_pricing(s.pricing_file)
        ledger = SpendLedger(s.llm_spend_ledger, PROJECT, s.project_spend_cap_usd, s.llm_spend_cap_usd)
        teacher = GroqTeacher(
            api_key=s.groq_api_key.get_secret_value(),
            base_url=s.groq_base_url,
            model=s.teacher_model,
            price=price,
            ledger=ledger,
            cache_dir=None,  # live traffic is not cached
            reasoning_effort=s.teacher_reasoning_effort,
            timeout_s=s.teacher_timeout_s,
            max_attempts=3,
        )
    router = Router(
        student,
        teacher,
        s.router_threshold,
        teacher_enabled=s.teacher_enabled,
        teacher_max_concurrency=s.teacher_max_concurrency,
    )
    return router, student, teacher


def create_app(settings: Settings | None = None, router: Router | None = None) -> FastAPI:
    s = settings or get_settings()
    setup_logging(s.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        student = teacher = None
        if router is None:
            app.state.router, student, teacher = build_router(s)
        else:
            app.state.router = router
        yield
        if student is not None:
            await student.aclose()
        if teacher is not None:
            await teacher.aclose()

    app = FastAPI(title="distill-router", version="0.1.0", lifespan=lifespan, docs_url=None, redoc_url=None)

    def auth(authorization: Annotated[str | None, Header()] = None) -> None:
        if s.router_api_key is None:
            return
        expected = "Bearer " + s.router_api_key.get_secret_value()
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        r: Router = request.app.state.router
        student_ok = await r.student.ready() if isinstance(r.student, StudentClient) else True
        teacher_ok = (not s.teacher_enabled) or r.teacher is not None
        ok = student_ok and teacher_ok
        body = {"student": student_ok, "teacher_configured": teacher_ok, "teacher_enabled": s.teacher_enabled}
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.post("/v1/classify", response_model=ClassifyResponse, dependencies=[Depends(auth)])
    async def classify(req: ClassifyRequest, request: Request) -> ClassifyResponse:
        r: Router = request.app.state.router
        res = await r.classify(req.text)
        log.info(
            "classified",
            extra={
                **query_fingerprint(req.text),
                "source": res.source,
                "reason": res.reason.value,
                "latency_ms": round(res.latency_ms, 1),
                "teacher_usd": res.teacher_usd,
            },
        )
        if res.label is None:
            raise HTTPException(
                status_code=503, detail="no valid label available (student invalid, teacher off)"
            )
        return ClassifyResponse(
            label=res.label,
            source=res.source,
            reason=res.reason.value,
            student_confidence=res.student_confidence,
            latency_ms=res.latency_ms,
        )

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("HOST", "0.0.0.0"),  # noqa: S104  # nosec B104 - container must listen on all interfaces
        port=int(os.environ.get("PORT", "8080")),
        timeout_graceful_shutdown=20,
        log_config=None,
    )


if __name__ == "__main__":
    main()

"""Router API: POST /v1/classify -> student (vLLM) first, teacher (Groq) on low confidence / invalid output.

/healthz = liveness (process up). /readyz = dependencies (student server reachable; teacher configured when
escalation is enabled; shared spend store reachable). Graceful shutdown: uvicorn drains on SIGTERM, then the
lifespan returns the unused spend lease and closes clients.

Request path (outermost first): SecurityHeaders (+ generic 500) -> BodySizeLimit (413) -> RateLimit (429) ->
AdmissionControl (429 when overloaded) -> FastAPI (auth 401, validation 422, routing).
"""

from __future__ import annotations

import hmac
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from distill.costs import load_pricing
from distill.guards import AdmissionControl, ASGIApp, BodySizeLimit, RateLimit, SecurityHeaders
from distill.labels import MAX_QUERY_CHARS
from distill.logs import query_fingerprint, setup_logging
from distill.router import FAULTS, Router, TeacherBreaker
from distill.settings import PROJECT, Settings, get_settings
from distill.shared_spend import FirestoreDoc, LeaseBudget, SharedBudget
from distill.spend import SpendLedger
from distill.student import StudentClient
from distill.teacher import GroqTeacher

log = logging.getLogger("distill.api")

VERSION = "0.1.0"


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
    # True when the label is the student's answer that did NOT pass the confidence/validity gate and was not
    # confirmed by the teacher (kill switch on, or teacher unavailable under the "degrade" policy).
    low_confidence: bool = False
    teacher_unavailable: bool = False


class KillSwitch(BaseModel):
    teacher_enabled: bool


def build_shared_budget(s: Settings) -> SharedBudget | None:
    if s.spend_store != "firestore":
        return None
    if s.firestore_project is None:  # also enforced by Settings validation
        raise ValueError("SPEND_STORE=firestore needs FIRESTORE_PROJECT")
    doc = FirestoreDoc(
        s.firestore_project,
        s.firestore_database,
        "spend_caps",
        PROJECT,
        emulator_host=s.firestore_emulator_host,
    )
    return LeaseBudget(doc, s.shared_spend_cap_usd, s.spend_lease_usd)


def build_router(s: Settings) -> tuple[Router, StudentClient, GroqTeacher | None, SharedBudget | None]:
    prod = s.env == "prod"
    student = StudentClient(
        s.student_base_url,
        s.student_model,
        s.student_timeout_s,
        gcp_id_token=s.student_gcp_id_token,
        reject_fake_backend=prod,
    )
    teacher: GroqTeacher | None = None
    shared = build_shared_budget(s)
    if s.groq_api_key is not None:
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
            shared_budget=shared,
            reject_fake_backend=prod,
        )
    router = Router(
        student,
        teacher,
        s.router_threshold,
        teacher_enabled=s.teacher_enabled,
        teacher_max_concurrency=s.teacher_max_concurrency,
        teacher_deadline_s=s.teacher_deadline_s,
        breaker=TeacherBreaker(s.teacher_breaker_failures, s.teacher_breaker_cooldown_s),
    )
    return router, student, teacher, shared


def _check_key(given: str | None, expected: str) -> bool:
    return given is not None and hmac.compare_digest(given.encode(), expected.encode())


def create_app(
    settings: Settings | None = None, router: Router | None = None, shared: SharedBudget | None = None
) -> ASGIApp:
    s = settings or get_settings()
    setup_logging(s.log_level)
    faults_on = s.fault_injection_enabled and s.env != "prod"  # Settings also refuses this combination

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        student = teacher = None
        app.state.shared = shared
        if router is None:
            app.state.router, student, teacher, app.state.shared = build_router(s)
        else:
            app.state.router = router
        log.info(
            "startup",
            extra={
                "env": s.env,
                "teacher_enabled": app.state.router.teacher_enabled,
                "spend_store": s.spend_store,
                "fault_injection": faults_on,
                "teacher_unavailable_policy": s.teacher_unavailable_policy,
            },
        )
        yield
        if app.state.shared is not None:
            await app.state.shared.aclose()  # return the unused lease
        if student is not None:
            await student.aclose()
        if teacher is not None:
            await teacher.aclose()

    api = FastAPI(
        title="distill-router",
        version=VERSION,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @api.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Starlette sends this 500 and re-raises; SecurityHeaders logs it (redacted). No internals leak.
        return JSONResponse({"detail": "internal error"}, status_code=500)

    def auth(
        x_api_key: Annotated[str | None, Header()] = None,
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        if s.router_api_key is None:
            return
        key = s.router_api_key.get_secret_value()
        bearer = authorization[7:] if authorization and authorization.startswith("Bearer ") else None
        if not (_check_key(x_api_key, key) or _check_key(bearer, key)):
            raise HTTPException(status_code=401, detail="unauthorized")

    def admin(x_admin_key: Annotated[str | None, Header()] = None) -> None:
        if s.admin_api_key is None:
            raise HTTPException(status_code=404, detail="Not Found")  # admin API disabled
        if not _check_key(x_admin_key, s.admin_api_key.get_secret_value()):
            raise HTTPException(status_code=401, detail="unauthorized")

    @api.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        r: Router = request.app.state.router
        student_ok = await r.student.ready() if isinstance(r.student, StudentClient) else True
        teacher_ok = (not r.teacher_enabled) or r.teacher is not None
        if s.teacher_enabled and r.teacher is None:
            teacher_ok = False  # escalation configured on but no key: misconfiguration
        shared: SharedBudget | None = request.app.state.shared
        store_ok = await shared.ready() if shared is not None else True
        ok = student_ok and teacher_ok
        body: dict[str, Any] = {
            "student": student_ok,
            "teacher_configured": teacher_ok,
            "teacher_enabled": r.teacher_enabled,
            "teacher_breaker": r.breaker.state,
            "spend_store": s.spend_store,
            "spend_store_ok": store_ok,
            "fault_injection": faults_on,
            "teacher_unavailable_policy": s.teacher_unavailable_policy,
            "env": s.env,
            "version": VERSION,
        }
        return JSONResponse(body, status_code=200 if ok else 503)

    @api.post("/v1/classify", response_model=ClassifyResponse, dependencies=[Depends(auth)])
    async def classify(
        req: ClassifyRequest,
        request: Request,
        x_distill_fault: Annotated[str | None, Header()] = None,
    ) -> ClassifyResponse | JSONResponse:
        r: Router = request.app.state.router
        faults: frozenset[str] = frozenset()
        if faults_on and x_distill_fault:
            faults = frozenset(f.strip() for f in x_distill_fault.split(",")) & FAULTS
        res = await r.classify(req.text, faults)
        log.info(
            "classified",
            extra={
                **query_fingerprint(req.text),
                "source": res.source,
                "reason": res.reason.value,
                "latency_ms": round(res.latency_ms, 1),
                "teacher_usd": res.teacher_usd,
                "teacher_unavailable": res.teacher_unavailable,
                "faults": sorted(faults),
            },
        )
        if res.teacher_unavailable and s.teacher_unavailable_policy == "fail":
            return JSONResponse(
                {"detail": "teacher unavailable", "reason": res.reason.value},
                status_code=503,
                headers={"Retry-After": "30"},
            )
        if res.label is None:
            return JSONResponse(
                {"detail": "no valid label available (student invalid, teacher off or unavailable)"},
                status_code=503,
                headers={"Retry-After": "30"},
            )
        return ClassifyResponse(
            label=res.label,
            source=res.source,
            reason=res.reason.value,
            student_confidence=res.student_confidence,
            latency_ms=res.latency_ms,
            low_confidence=res.low_confidence,
            teacher_unavailable=res.teacher_unavailable,
        )

    @api.post("/admin/kill-switch", dependencies=[Depends(admin)])
    async def kill_switch(body: KillSwitch, request: Request) -> dict[str, bool]:
        r: Router = request.app.state.router
        r.set_teacher_enabled(body.teacher_enabled)
        return {"teacher_enabled": r.teacher_enabled}

    @api.get("/admin/stats", dependencies=[Depends(admin)])
    async def stats(request: Request) -> dict[str, Any]:
        r: Router = request.app.state.router
        shared: SharedBudget | None = request.app.state.shared
        out: dict[str, Any] = {
            "router": dict(r.stats),
            "breaker": r.breaker.state,
            "admission": admission.snapshot(),
            "teacher_enabled": r.teacher_enabled,
        }
        if isinstance(shared, LeaseBudget):
            out["shared_budget"] = shared.snapshot()
        return out

    admission = AdmissionControl(api, s.max_inflight, s.max_queue, s.queue_timeout_s)
    limited = RateLimit(admission, s.rate_limit_rps, s.rate_limit_burst)
    return SecurityHeaders(BodySizeLimit(limited, s.max_body_bytes))


def main() -> None:
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("HOST", "0.0.0.0"),  # noqa: S104 (container must listen on all interfaces)  # nosec B104
        port=int(os.environ.get("PORT", "8080")),
        timeout_graceful_shutdown=20,
        log_config=None,
        server_header=False,
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()

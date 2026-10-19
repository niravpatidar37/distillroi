"""Production hardening: prod gates for the fake LLM / fault injection, overload protection (429 +
Retry-After), teacher-unavailable policy + circuit breaker, runtime kill switch, security headers, body
cap."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from pydantic import ValidationError

from distill.app import create_app
from distill.costs import TeacherPrice
from distill.guards import AdmissionControl, BodySizeLimit, RateLimit, SecurityHeaders
from distill.router import Reason, Router, TeacherBreaker
from distill.settings import Settings
from distill.spend import BudgetExceeded, SpendLedger
from distill.student import FakeBackendRejected, StudentClient, StudentResult
from distill.teacher import DailyQuotaExhausted, GroqTeacher, TeacherError, Usage

ROOT = Path(__file__).resolve().parents[1]


def _load_fake_llm():
    spec = importlib.util.spec_from_file_location("fake_llm", ROOT / "smoke" / "fake_llm.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeStudent:
    def __init__(self, label: str | None = "card_arrival", conf: float = 0.95, exc: Exception | None = None):
        self.label, self.conf, self.exc, self.calls = label, conf, exc, 0

    async def classify(self, query: str) -> StudentResult:
        self.calls += 1
        if self.exc:
            raise self.exc
        return StudentResult(self.label, self.label or "garbage", self.conf, 0.01)


class FakeTeacher:
    def __init__(self, label: str | None = "lost_or_stolen_card", exc: Exception | None = None, delay=0.0):
        self.label, self.exc, self.delay, self.calls = label, exc, delay, 0

    async def classify_one(self, query: str) -> tuple[str | None, Usage]:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.label, Usage(input_tokens=700, output_tokens=30, usd=0.0001)


def _client(router: Router, **kw: object) -> TestClient:
    s = Settings(_env_file=None, **kw)  # type: ignore[call-arg]
    return TestClient(create_app(settings=s, router=router))


# ---------------------------------------------------------------- prod gates (fake LLM can never serve prod)


@pytest.mark.parametrize(
    "kw",
    [
        {"fault_injection_enabled": True},
        {"firestore_emulator_host": "firestore:8080"},
        {"student_base_url": "http://fake-llm:8000/v1"},
        {"groq_base_url": "http://fake-llm:8000/v1"},
    ],
)
def test_prod_refuses_test_only_configuration(kw):
    base = {"student_base_url": "https://vllm.example/v1"}
    Settings(_env_file=None, env="staging", **{**base, **kw})  # type: ignore[call-arg]  # allowed outside prod
    with pytest.raises(ValidationError):
        Settings(_env_file=None, env="prod", **{**base, **kw})  # type: ignore[call-arg]


def test_blank_keys_from_compose_mean_not_configured():
    s = Settings(_env_file=None, router_api_key="", admin_api_key="  ", groq_api_key="")  # type: ignore[call-arg]
    assert s.router_api_key is None and s.admin_api_key is None and s.groq_api_key is None


def test_prod_accepts_https_backends_without_test_features():
    s = Settings(_env_file=None, env="prod", student_base_url="https://vllm.example/v1")  # type: ignore[call-arg]
    assert s.fault_injection_enabled is False


def test_fake_llm_server_refuses_to_start_in_prod(monkeypatch):
    mod = _load_fake_llm()
    monkeypatch.setenv("ENV", "prod")
    with pytest.raises(SystemExit):
        mod.create_app()
    with pytest.raises(SystemExit):
        mod.main()
    monkeypatch.setenv("ENV", "ci")
    assert mod.create_app() is not None


async def _fake_student_client(reject: bool) -> StudentClient:
    mod = _load_fake_llm()
    transport = httpx.ASGITransport(app=mod.create_app())
    base = "http://fake-llm.test/v1"
    return StudentClient(
        base, "distill-student", client=httpx.AsyncClient(transport=transport), reject_fake_backend=reject
    )


async def test_fake_llm_mimics_vllm_and_router_uses_it_outside_prod():
    c = await _fake_student_client(reject=False)
    ok = await c.classify("My new card still has not arrived")
    vague = await c.classify("hmm")
    assert ok.label == "card_arrival" and ok.confidence > 0.9
    assert vague.label is not None and vague.confidence < 0.5
    assert await c.ready()
    await c.aclose()


async def test_router_refuses_fake_backend_replies_in_prod_mode():
    c = await _fake_student_client(reject=True)
    with pytest.raises(FakeBackendRejected):
        await c.classify("My new card still has not arrived")
    r = await Router(c, None, 0.5, teacher_enabled=False).classify("My new card still has not arrived")
    assert r.label is None and r.reason is Reason.STUDENT_ERROR  # served nothing from the fake
    await c.aclose()


@respx.mock
async def test_teacher_refuses_fake_backend_replies_in_prod_mode(tmp_path):
    respx.post("https://groq.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"x-distill-fake": "1"},
            json={"choices": [{"message": {"content": '{"1": "card_arrival"}'}}], "usage": {}},
        )
    )
    t = GroqTeacher(
        api_key="k",
        base_url="https://groq.test/v1",
        model="m",
        price=TeacherPrice(0.15, 0.6),
        ledger=SpendLedger(tmp_path / "l.jsonl", "p5-distill", 5.0, 25.0),
        max_attempts=1,
        reject_fake_backend=True,
    )
    with pytest.raises(TeacherError):
        await t.classify_one("q")
    await t.aclose()


# ---------------------------------------------------------------- fault injection is opt-in


def test_fault_header_ignored_unless_enabled():
    teacher = FakeTeacher()
    router = Router(FakeStudent(conf=0.99), teacher, 0.5)
    with _client(router) as c:
        r = c.post("/v1/classify", json={"text": "q"}, headers={"X-Distill-Fault": "low_confidence"})
        assert r.json()["source"] == "student" and teacher.calls == 0
    with _client(router, fault_injection_enabled=True, env="ci") as c:
        r = c.post("/v1/classify", json={"text": "q"}, headers={"X-Distill-Fault": "low_confidence"})
        assert r.json()["source"] == "teacher" and r.json()["reason"] == "low_confidence"
        r = c.post("/v1/classify", json={"text": "q"}, headers={"X-Distill-Fault": "invalid_output"})
        assert r.json()["source"] == "teacher" and r.json()["reason"] == "invalid_output"
        r = c.post("/v1/classify", json={"text": "q"}, headers={"X-Distill-Fault": "rm -rf,bogus"})
        assert r.json()["source"] == "student"  # unknown fault names are ignored


# ---------------------------------------------------------------- teacher unavailable path (gap 3)


@pytest.mark.parametrize(
    "exc",
    [
        TeacherError("down"),
        httpx.ConnectError("down"),
        BudgetExceeded("cap"),
        DailyQuotaExhausted("TPD", 600.0),
    ],
)
def test_teacher_unavailable_degrades_with_low_confidence_flag(exc, caplog):
    router = Router(FakeStudent(conf=0.2), FakeTeacher(exc=exc), 0.5)
    with _client(router, teacher_unavailable_policy="degrade") as c:
        r = c.post("/v1/classify", json={"text": "q"})
    j = r.json()
    assert r.status_code == 200
    assert j["source"] == "student_degraded" and j["label"] == "card_arrival"
    assert j["low_confidence"] is True and j["teacher_unavailable"] is True
    assert router.stats["teacher_unavailable"] == 1


def test_teacher_unavailable_fail_policy_returns_503_with_retry_after():
    router = Router(FakeStudent(conf=0.2), FakeTeacher(exc=TeacherError("down")), 0.5)
    with _client(router, teacher_unavailable_policy="fail") as c:
        r = c.post("/v1/classify", json={"text": "q"})
    assert r.status_code == 503 and r.headers["retry-after"].isdigit()
    assert r.json()["detail"] == "teacher unavailable"


def test_teacher_unavailable_and_invalid_student_output_is_503():
    router = Router(FakeStudent(label=None), FakeTeacher(exc=TeacherError("down")), 0.5)
    with _client(router) as c:
        r = c.post("/v1/classify", json={"text": "q"})
    assert r.status_code == 503 and "retry-after" in r.headers


def test_teacher_unavailable_is_logged_for_the_alert(capsys):
    router = Router(FakeStudent(conf=0.2), FakeTeacher(exc=BudgetExceeded("cap")), 0.5)
    with _client(router) as c:
        c.post("/v1/classify", json={"text": "q"})
    recs = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    hit = [x for x in recs if x["msg"] == "teacher_unavailable"]
    # the Terraform log-based metrics filter on these fields (infra/terraform/modules/distill/monitoring.tf)
    assert hit and hit[0]["cause"] == "budget_exceeded" and hit[0]["severity"] == "WARNING"


async def test_teacher_deadline_bounds_escalation_latency():
    router = Router(FakeStudent(conf=0.2), FakeTeacher(delay=5.0), 0.5, teacher_deadline_s=0.05)
    r = await router.classify("q")
    assert r.teacher_unavailable and r.source == "student_degraded" and r.latency_ms < 2000


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


async def test_breaker_opens_skips_teacher_then_probes_and_recovers():
    clock = Clock()
    teacher = FakeTeacher(exc=TeacherError("down"))
    router = Router(FakeStudent(conf=0.2), teacher, 0.5, breaker=TeacherBreaker(3, 30.0, clock))
    for _ in range(3):
        await router.classify("q")
    assert teacher.calls == 3 and router.breaker.state == "open"
    for _ in range(5):
        r = await router.classify("q")
        assert r.teacher_unavailable
    assert teacher.calls == 3  # open breaker: no calls, fast degrade
    clock.t += 31
    teacher.exc = None
    r = await router.classify("q")
    assert r.source == "teacher" and router.breaker.state == "closed" and teacher.calls == 4


async def test_daily_quota_opens_breaker_immediately_until_reset_hint():
    clock = Clock()
    teacher = FakeTeacher(exc=DailyQuotaExhausted("TPD", 600.0))
    router = Router(FakeStudent(conf=0.2), teacher, 0.5, breaker=TeacherBreaker(3, 30.0, clock))
    await router.classify("q")
    assert router.breaker.state == "open"
    clock.t += 100
    await router.classify("q")
    assert teacher.calls == 1  # still open: quota resets in 600 s
    clock.t += 600
    assert router.breaker.state == "half_open"


# ---------------------------------------------------------------- runtime kill switch + admin API


def test_admin_api_is_404_without_admin_key():
    router = Router(FakeStudent(), FakeTeacher(), 0.5)
    with _client(router) as c:
        assert c.post("/admin/kill-switch", json={"teacher_enabled": False}).status_code == 404
        assert c.get("/admin/stats").status_code == 404


def test_runtime_kill_switch_stops_teacher_calls():
    teacher = FakeTeacher()
    router = Router(FakeStudent(conf=0.2), teacher, 0.5)
    with _client(router, admin_api_key="adm1n-key") as c:
        bad = c.post("/admin/kill-switch", json={"teacher_enabled": False}, headers={"X-Admin-Key": "x"})
        assert bad.status_code == 401
        h = {"X-Admin-Key": "adm1n-key"}
        assert c.post("/admin/kill-switch", json={"teacher_enabled": False}, headers=h).json() == {
            "teacher_enabled": False
        }
        assert c.get("/readyz").json()["teacher_enabled"] is False
        r = c.post("/v1/classify", json={"text": "q"}).json()
        assert r["source"] == "student_degraded" and r["low_confidence"] is True and teacher.calls == 0
        assert c.get("/admin/stats", headers=h).json()["router"].get("teacher_calls", 0) == 0
        c.post("/admin/kill-switch", json={"teacher_enabled": True}, headers=h)
        assert c.post("/v1/classify", json={"text": "q"}).json()["source"] == "teacher"


def test_api_key_accepted_via_x_api_key_header():
    router = Router(FakeStudent(), FakeTeacher(), 0.5)
    with _client(router, router_api_key="s3cret-key") as c:
        assert (
            c.post("/v1/classify", json={"text": "q"}, headers={"X-API-Key": "s3cret-key"}).status_code == 200
        )
        assert c.post("/v1/classify", json={"text": "q"}, headers={"X-API-Key": "nope"}).status_code == 401


# ---------------------------------------------------------------- security headers / errors / body cap


def test_security_headers_and_no_server_banner():
    with _client(Router(FakeStudent(), None, 0.5), teacher_enabled=False) as c:
        r = c.get("/healthz")
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
    assert r.headers["cache-control"] == "no-store" and "content-security-policy" in r.headers
    assert "server" not in r.headers


def test_unhandled_error_is_generic_500_without_trace():
    class Exploding:
        async def classify(self, query: str) -> StudentResult:
            raise RuntimeError("secret internal detail /app/src/distill/x.py")

    with _client(Router(Exploding(), None, 0.5), teacher_enabled=False) as c:
        r = c.post("/v1/classify", json={"text": "q"})
    assert r.status_code == 500 and r.json() == {"detail": "internal error"}
    assert "secret" not in r.text and "Traceback" not in r.text
    assert r.headers["x-content-type-options"] == "nosniff"


def test_oversized_body_is_413_before_parsing():
    student = FakeStudent()
    with _client(Router(student, None, 0.5), teacher_enabled=False) as c:
        r = c.post(
            "/v1/classify",
            content=json.dumps({"text": "y" * 20000}),
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 413 and student.calls == 0


async def _echo_app(scope, receive, send):
    body = b""
    while True:
        m = await receive()
        body += m.get("body", b"")
        if not m.get("more_body"):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": str(len(body)).encode()})


async def test_body_limit_counts_streamed_chunks_without_content_length():
    app = BodySizeLimit(_echo_app, max_bytes=1024)
    sent: list[dict] = []
    chunks = [{"type": "http.request", "body": b"x" * 600, "more_body": True}] * 2

    async def receive():
        return chunks.pop(0)

    async def send(m):
        sent.append(m)

    await app({"type": "http", "method": "POST", "path": "/v1/classify", "headers": []}, receive, send)
    assert sent[0]["status"] == 413


# ---------------------------------------------------------------- overload: admission control (gap 2)


class Gate:
    """Inner ASGI app that blocks until released, to hold requests in flight."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.entered = 0
        self._cond = asyncio.Condition()

    async def wait_entered(self, n: int) -> None:
        async with self._cond:
            await asyncio.wait_for(self._cond.wait_for(lambda: self.entered >= n), 2)

    async def __call__(self, scope, receive, send):
        async with self._cond:
            self.entered += 1
            self._cond.notify_all()
        await self.release.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def _ac(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_admission_sheds_excess_with_429_and_retry_after():
    gate = Gate()
    guard = AdmissionControl(gate, max_inflight=2, max_queue=0, queue_timeout_s=30.0)
    async with _ac(SecurityHeaders(guard)) as c:
        held = [asyncio.create_task(c.post("/v1/classify")) for _ in range(2)]
        await gate.wait_entered(2)
        # rejected immediately (queue size 0), not after the 30 s queue timeout and not admitted
        r = await asyncio.wait_for(c.post("/v1/classify"), 1.0)
        assert r.status_code == 429 and r.headers["retry-after"] == "1"
        assert r.json()["detail"] == "overloaded, retry later"
        probe = asyncio.create_task(c.get("/healthz"))  # non-/v1 paths are never shed (probes under load)
        await gate.wait_entered(3)
        gate.release.set()
        res = await asyncio.wait_for(asyncio.gather(*held, probe), 2)
        assert [x.status_code for x in res] == [200, 200, 200]
    assert guard.shed == 1 and guard.inflight == 0


async def test_admission_queues_then_times_out():
    gate = Gate()
    guard = AdmissionControl(gate, max_inflight=1, max_queue=1, queue_timeout_s=0.3)
    async with _ac(guard) as c:
        first = asyncio.create_task(c.post("/v1/classify"))
        await gate.wait_entered(1)
        queued = asyncio.create_task(c.post("/v1/classify"))
        await asyncio.sleep(0.01)
        t0 = asyncio.get_running_loop().time()
        full = await asyncio.wait_for(c.post("/v1/classify"), 1.0)  # queue (1) already taken
        assert full.status_code == 429 and asyncio.get_running_loop().time() - t0 < 0.2  # no waiting
        assert (await asyncio.wait_for(queued, 1.0)).status_code == 429  # waited 0.3 s, then shed
        assert gate.entered == 1
        gate.release.set()
        assert (await first).status_code == 200


async def test_admission_queued_request_succeeds_when_slot_frees():
    gate = Gate()
    guard = AdmissionControl(gate, max_inflight=1, max_queue=4, queue_timeout_s=2.0)
    async with _ac(guard) as c:
        tasks = [asyncio.create_task(c.post("/v1/classify")) for _ in range(3)]
        await asyncio.sleep(0.02)
        assert gate.entered == 1 and guard.waiting == 2
        gate.release.set()
        assert [(await t).status_code for t in tasks] == [200, 200, 200]


# ---------------------------------------------------------------- per-client rate limit


async def test_rate_limit_429_with_retry_after_per_client():
    clock = Clock()

    async def ok(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    rl = RateLimit(ok, rps=1.0, burst=3, clock=clock)
    async with _ac(rl) as c:
        a = {"X-API-Key": "client-a"}
        codes = [(await c.post("/v1/classify", headers=a)).status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]
        r = await c.post("/v1/classify", headers=a)
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
        assert (await c.post("/v1/classify", headers={"X-API-Key": "client-b"})).status_code == 200
        assert (await c.get("/healthz", headers=a)).status_code == 200  # only /v1/* is limited
        clock.t += 1.0
        assert (await c.post("/v1/classify", headers=a)).status_code == 200

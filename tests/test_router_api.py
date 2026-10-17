from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from distill.app import create_app
from distill.logs import JsonFormatter, redact
from distill.router import Reason, Router, decide, simulate_routing, tune_threshold
from distill.settings import Settings
from distill.spend import BudgetExceeded
from distill.student import StudentClient, StudentResult, parse_completion
from distill.teacher import Usage


class FakeStudent:
    def __init__(self, result: StudentResult | Exception):
        self.result = result
        self.calls = 0

    async def classify(self, query: str) -> StudentResult:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    async def ready(self) -> bool:
        return True


class FakeTeacher:
    def __init__(self, label: str | None = "card_arrival", exc: Exception | None = None):
        self.label, self.exc, self.calls = label, exc, 0

    async def classify_one(self, query: str) -> tuple[str | None, Usage]:
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.label, Usage(input_tokens=700, output_tokens=40, usd=0.0001)


def sr(label: str | None, conf: float) -> StudentResult:
    return StudentResult(label, label or "garbage", conf, 0.01)


@pytest.mark.parametrize(
    "student,expected",
    [
        (None, Reason.STUDENT_ERROR),
        (sr(None, 0.99), Reason.INVALID_OUTPUT),
        (sr("card_arrival", 0.3), Reason.LOW_CONFIDENCE),
        (sr("card_arrival", 0.9), Reason.ACCEPTED),
    ],
)
def test_decide(student, expected):
    assert decide(student, 0.5) is expected


async def test_confident_student_is_served_without_teacher():
    t = FakeTeacher()
    r = await Router(FakeStudent(sr("top_up_failed", 0.95)), t, 0.5).classify("q")
    assert (r.label, r.source, t.calls) == ("top_up_failed", "student", 0)


async def test_invalid_student_output_escalates_to_teacher():
    t = FakeTeacher("card_arrival")
    r = await Router(FakeStudent(sr(None, 0.99)), t, 0.5).classify("q")
    assert (r.label, r.source, r.reason, t.calls) == ("card_arrival", "teacher", Reason.INVALID_OUTPUT, 1)


async def test_low_confidence_escalates_and_student_outage_escalates():
    t = FakeTeacher("card_arrival")
    r = await Router(FakeStudent(sr("top_up_failed", 0.2)), t, 0.5).classify("q")
    assert r.source == "teacher" and r.reason is Reason.LOW_CONFIDENCE
    r2 = await Router(FakeStudent(httpx.ConnectError("down")), t, 0.5).classify("q")
    assert r2.source == "teacher" and r2.reason is Reason.STUDENT_ERROR


async def test_kill_switch_blocks_all_teacher_calls():
    """Security: TEACHER_ENABLED=false must stop paid calls even on low confidence."""
    t = FakeTeacher()
    r = await Router(FakeStudent(sr("top_up_failed", 0.1)), t, 0.5, teacher_enabled=False).classify("q")
    assert t.calls == 0 and r.source == "student_degraded" and r.label == "top_up_failed"


async def test_budget_exceeded_degrades_instead_of_failing():
    t = FakeTeacher(exc=BudgetExceeded("cap"))
    r = await Router(FakeStudent(sr("top_up_failed", 0.1)), t, 0.5).classify("q")
    assert r.source == "student_degraded" and r.label == "top_up_failed"
    r2 = await Router(FakeStudent(sr(None, 0.9)), t, 0.5).classify("q")
    assert r2.label is None and r2.source == "none"


def test_simulate_and_tune_threshold_on_validation_data():
    gold = ["a", "b", "c", "d"]
    s_lab = ["a", "x", "c", None]
    s_conf = [0.9, 0.4, 0.8, 0.95]
    t_lab = ["a", "b", "c", "d"]
    routed, esc = simulate_routing(s_lab, s_conf, t_lab, 0.5)
    assert routed == ["a", "b", "c", "d"] and esc == 0.5  # low-conf + invalid escalate
    best = tune_threshold(gold, s_lab, s_conf, t_lab, quality_target=1.0, grid=[0.0, 0.5, 0.85, 0.99])
    assert best["threshold"] == 0.5 and best["routed_accuracy"] == 1.0 and best["target_met"] == 1.0


def test_parse_completion_joint_probability():
    data = {
        "choices": [
            {
                "message": {"content": "card_arrival"},
                "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
            }
        ],
        "usage": {"completion_tokens": 2},
    }
    raw, conf, n = parse_completion(data)
    assert raw == "card_arrival" and n == 2
    assert conf == pytest.approx(2.718281828**-0.3, rel=1e-6)
    assert parse_completion({"choices": [{"message": {"content": "x"}}]})[1] == 0.0


@respx.mock
async def test_student_client_maps_raw_text_to_enum_or_none():
    base = "http://vllm.test/v1"
    respx.post(f"{base}/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": " card_arrival"},
                            "logprobs": {"content": [{"logprob": -0.01}]},
                        }
                    ]
                },
            ),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "rm -rf /"}, "logprobs": {"content": [{"logprob": -0.01}]}}
                    ]
                },
            ),
        ]
    )
    c = StudentClient(base, "distill-student")
    ok = await c.classify("where is my card")
    bad = await c.classify("x")
    await c.aclose()
    assert ok.label == "card_arrival" and bad.label is None


# ---------------- API ----------------


def _client(router: Router, **settings: object) -> TestClient:
    s = Settings(_env_file=None, **settings)  # type: ignore[call-arg]
    return TestClient(create_app(settings=s, router=router))


def test_api_classify_and_health():
    router = Router(FakeStudent(sr("card_arrival", 0.9)), FakeTeacher(), 0.5)
    with _client(router) as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        r = c.post("/v1/classify", json={"text": "where is my card"})
        assert (
            r.status_code == 200 and r.json()["label"] == "card_arrival" and r.json()["source"] == "student"
        )


def test_api_rejects_oversized_and_blank_input():
    router = Router(FakeStudent(sr("card_arrival", 0.9)), FakeTeacher(), 0.5)
    with _client(router) as c:
        assert c.post("/v1/classify", json={"text": "x" * 501}).status_code == 422
        assert c.post("/v1/classify", json={"text": "   "}).status_code == 422
        assert c.post("/v1/classify", json={}).status_code == 422


def test_api_requires_bearer_when_key_configured():
    """Security: with ROUTER_API_KEY set, unauthenticated calls never reach the models."""
    student = FakeStudent(sr("card_arrival", 0.9))
    router = Router(student, FakeTeacher(), 0.5)
    with _client(router, router_api_key="s3cret") as c:
        assert c.post("/v1/classify", json={"text": "q"}).status_code == 401
        bad = c.post("/v1/classify", json={"text": "q"}, headers={"Authorization": "Bearer nope"})
        assert bad.status_code == 401
        assert student.calls == 0
        ok = c.post("/v1/classify", json={"text": "q"}, headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200


def test_api_503_when_no_valid_label_available():
    router = Router(FakeStudent(sr(None, 0.9)), None, 0.5)
    with _client(router, teacher_enabled=False) as c:
        assert c.post("/v1/classify", json={"text": "q"}).status_code == 503


def test_api_logs_never_contain_query_text(capsys):
    """Security: customer text (PII) is logged only as a hash + length (JSON logs go to stdout)."""
    router = Router(FakeStudent(sr("card_arrival", 0.9)), FakeTeacher(), 0.5)
    secret_text = "my card 4111 1111 1111 1111 email jane@example.com"
    with _client(router) as c:
        c.post("/v1/classify", json={"text": secret_text})
    out = capsys.readouterr().out
    recs = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    classified = [r for r in recs if r["msg"] == "classified"]
    assert len(classified) == 1
    assert "4111" not in out and "jane@example.com" not in out and secret_text not in out
    assert classified[0]["q_len"] == len(secret_text) and len(classified[0]["q_sha"]) == 12


def test_json_formatter_redacts_extra_fields():
    rec = logging.LogRecord("x", logging.INFO, "f", 1, "user jane@example.com", (), None)
    rec.detail = "card 4111111111111111"
    out = JsonFormatter().format(rec)
    assert "jane@example.com" not in out and "4111111111111111" not in out


def test_redact_masks_pii_and_secrets():
    s = redact("card 4111-1111-1111-1111 mail a.b@c.io key gsk_ABCDEFGHIJKLMNOPQRSTUV Bearer abc.def")
    assert "4111" not in s and "a.b@c.io" not in s and "gsk_ABC" not in s and "abc.def" not in s


def test_readyz_reports_teacher_misconfiguration():
    router = Router(FakeStudent(sr("card_arrival", 0.9)), None, 0.5)
    with _client(router, teacher_enabled=True) as c:
        r = c.get("/readyz")
        assert r.status_code == 503 and r.json()["teacher_configured"] is False
    with _client(router, teacher_enabled=False) as c:
        assert c.get("/readyz").status_code == 200


@respx.mock
async def test_student_client_attaches_cached_gcp_id_token():
    base = "https://vllm-abc.a.run.app/v1"
    meta = respx.get(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity"
    ).mock(return_value=httpx.Response(200, text="tok123"))
    chat = respx.post(f"{base}/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "card_arrival"}}]})
    )
    c = StudentClient(base, "distill-student", gcp_id_token=True)
    await c.classify("a")
    await c.classify("b")
    await c.aclose()
    assert meta.call_count == 1  # cached
    assert meta.calls[0].request.url.params["audience"] == "https://vllm-abc.a.run.app"
    assert meta.calls[0].request.headers["Metadata-Flavor"] == "Google"
    assert chat.calls[0].request.headers["Authorization"] == "Bearer tok123"

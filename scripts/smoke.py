#!/usr/bin/env python3
"""End-to-end smoke test of a running distillroi router. Standard library only (runs on any CI/deploy runner).

    python scripts/smoke.py --base-url http://127.0.0.1:8115 --json-out results/smoke_local.json

Credentials come from the environment (never from argv, so they don't land in shell history / CI logs):
  SMOKE_API_KEY    app-level key (sent as X-API-Key)             -> enables the auth checks
  SMOKE_ID_TOKEN   Google ID token for an IAM-gated Cloud Run URL (sent as Authorization: Bearer)
  SMOKE_ADMIN_KEY  admin key (X-Admin-Key)                       -> enables the runtime kill-switch check

Checks that need fault injection (X-Distill-Fault, only honoured when the target reports fault_injection=true,
which is impossible with ENV=prod) or the admin key are reported as SKIP, never as PASS. Exit code 1 on any FAIL.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LABELS = frozenset(
    json.loads((ROOT / "src" / "distill" / "banking77_labels.json").read_text(encoding="utf-8"))
)
CORE_QUERY = "My new card still has not arrived, where is it?"
REQUIRED_SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "cache-control": "no-store",
}
LEAK_MARKERS = ("Traceback", 'File "', "site-packages", "uvicorn", "starlette")


@dataclass
class Resp:
    status: int
    headers: dict[str, str]
    body: bytes
    ms: float

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None


@dataclass
class Check:
    name: str
    status: str  # PASS | FAIL | SKIP
    detail: str = ""
    ms: float = 0.0


@dataclass
class Smoke:
    base: str
    api_key: str | None
    id_token: str | None
    admin_key: str | None
    timeout: float = 30.0
    checks: list[Check] = field(default_factory=list)
    target: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- http
    def req(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        *,
        auth: bool = True,
    ) -> Resp:
        h: dict[str, str] = {"user-agent": "distillroi-smoke/1"}
        if body is not None:
            h["content-type"] = "application/json"
        if auth and self.api_key:
            h["x-api-key"] = self.api_key
        # The IAM token is what gets a request through Cloud Run at all. With auth=False it is kept only when an
        # app key exists (so the app-level check is what gets exercised); otherwise IAM rejection is tested.
        if self.id_token and (auth or self.api_key):
            h["authorization"] = f"Bearer {self.id_token}"
        h.update(headers or {})
        r = urllib.request.Request(self.base + path, data=body, method=method, headers=h)  # noqa: S310
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(r, timeout=self.timeout) as resp:  # noqa: S310  # nosec B310
                data = resp.read()
                st, hd = resp.status, {k.lower(): v for k, v in resp.headers.items()}
        except urllib.error.HTTPError as e:
            data = e.read()
            st, hd = e.code, {k.lower(): v for k, v in e.headers.items()}
        return Resp(st, hd, data, (time.perf_counter() - t0) * 1000)

    def classify(self, text: str, fault: str | None = None, **kw: Any) -> Resp:
        hdr = {"x-distill-fault": fault} if fault else {}
        return self.req("POST", "/v1/classify", json.dumps({"text": text}).encode(), hdr, **kw)

    # ---------------------------------------------------------------- bookkeeping
    def run(self, name: str, fn: Any) -> None:
        t0 = time.perf_counter()
        try:
            res = fn()
            status, detail = res if isinstance(res, tuple) else ("PASS", str(res or ""))
        except AssertionError as e:
            status, detail = "FAIL", str(e) or "assertion failed"
        except (OSError, ValueError, KeyError, TypeError) as e:
            status, detail = "FAIL", f"{type(e).__name__}: {e}"
        c = Check(name, status, detail[:300], round((time.perf_counter() - t0) * 1000, 1))
        self.checks.append(c)
        print(f"{c.status:4}  {c.name:34} {c.ms:8.1f} ms  {c.detail}", flush=True)

    # ---------------------------------------------------------------- checks
    def liveness(self) -> str:
        r = self.req("GET", "/healthz")
        assert r.status == 200 and r.json() == {"status": "ok"}, f"healthz {r.status} {r.body[:100]!r}"
        return "200 ok"

    def readiness(self) -> str:
        r = self.req("GET", "/readyz")
        j = r.json() or {}
        self.target = j
        assert r.status == 200, f"readyz {r.status}: {j}"
        assert j.get("student") is True, f"student not reachable: {j}"
        assert j.get("teacher_configured") is True, f"teacher misconfigured: {j}"
        assert j.get("spend_store_ok") is True, f"spend store not reachable: {j}"
        return (
            f"student up, teacher_enabled={j.get('teacher_enabled')}, store={j.get('spend_store')}, "
            f"env={j.get('env')}, faults={j.get('fault_injection')}"
        )

    def security_headers(self) -> str:
        r = self.req("GET", "/healthz")
        for k, v in REQUIRED_SECURITY_HEADERS.items():
            assert r.headers.get(k) == v, f"missing/wrong {k}: {r.headers.get(k)!r}"
        assert "content-security-policy" in r.headers, "no CSP"
        assert "uvicorn" not in r.headers.get("server", "").lower(), "server header leaks uvicorn"
        return "nosniff, DENY, no-store, CSP, no server banner"

    def no_stack_traces(self) -> str:
        bad = self.req("POST", "/v1/classify", b'{"text": "unterminated')
        assert bad.status in (400, 422), f"malformed JSON -> {bad.status}"
        nf = self.req("GET", "/definitely-not-a-route")
        assert nf.status == 404, f"unknown route -> {nf.status}"
        docs = self.req("GET", "/docs")
        assert docs.status == 404, f"/docs exposed ({docs.status})"
        for r in (bad, nf, docs):
            text = r.body.decode("utf-8", "replace")
            assert not any(m in text for m in LEAK_MARKERS), f"internal detail leaked: {text[:120]}"
        return f"malformed->{bad.status}, unknown->404, /docs->404, no internals in bodies"

    def auth_rejected(self) -> Any:
        if not self.api_key and not self.id_token:
            return ("SKIP", "no SMOKE_API_KEY / SMOKE_ID_TOKEN: target auth not exercised")
        r = self.classify(CORE_QUERY, auth=False)
        assert r.status in (401, 403), f"no credentials -> {r.status} (expected 401/403)"
        detail = f"no credentials -> {r.status}"
        if self.api_key:
            w = self.req(
                "POST",
                "/v1/classify",
                json.dumps({"text": CORE_QUERY}).encode(),
                {"x-api-key": "wrong-" + "k" * 12},
            )
            assert w.status == 401, f"wrong key -> {w.status} (expected 401)"
            detail += ", wrong key -> 401"
        return detail

    def auth_accepted(self) -> Any:
        if not self.api_key and not self.id_token:
            return ("SKIP", "no credentials configured")
        r = self.classify(CORE_QUERY)
        assert r.status == 200, f"valid credentials -> {r.status} {r.body[:120]!r}"
        return "valid credentials -> 200"

    def validation(self) -> str:
        cases = {
            "blank": b'{"text": "   "}',
            "missing": b"{}",
            "wrong_type": b'{"text": 42}',
            "empty": b'{"text": ""}',
        }
        got = {}
        for k, body in cases.items():
            got[k] = self.req("POST", "/v1/classify", body).status
        assert all(v == 422 for v in got.values()), f"expected 422 for all: {got}"
        return "blank/missing/wrong-type/empty -> 422"

    def oversize(self) -> str:
        long = self.classify("x" * 501)
        assert long.status == 422, f"501-char text -> {long.status}"
        big = self.req("POST", "/v1/classify", json.dumps({"text": "y" * 65536}).encode())
        assert big.status == 413, f"64 KiB body -> {big.status}"
        return "501 chars -> 422, 64 KiB body -> 413"

    def core_classify(self) -> str:
        r = self.classify(CORE_QUERY)
        j = r.json() or {}
        assert r.status == 200, f"{r.status} {r.body[:160]!r}"
        assert j.get("label") in LABELS, f"label {j.get('label')!r} not in banking77"
        conf = j.get("student_confidence")
        assert isinstance(conf, int | float) and 0.0 <= conf <= 1.0, f"confidence {conf!r}"
        assert j.get("source") in ("student", "teacher", "student_degraded"), f"source {j.get('source')}"
        for k in ("reason", "latency_ms", "low_confidence", "teacher_unavailable"):
            assert k in j, f"missing field {k}"
        return f"label={j['label']} conf={conf:.3f} source={j['source']} ({r.ms:.0f} ms)"

    def _needs_faults(self, teacher: bool = True) -> str | None:
        if not self.target.get("fault_injection"):
            return "target has fault injection off (expected in prod)"
        if teacher and not self.target.get("teacher_enabled"):
            return "teacher disabled on target"
        return None

    def escalate(self, fault: str) -> Any:
        why = self._needs_faults()
        if why:
            return ("SKIP", why)
        r = self.classify(CORE_QUERY, fault=fault)
        j = r.json() or {}
        assert r.status == 200, f"{r.status} {r.body[:160]!r}"
        assert j.get("reason") == fault, f"reason {j.get('reason')} != {fault}"
        assert j.get("source") == "teacher", f"not escalated to teacher: source={j.get('source')} {j}"
        assert j.get("label") in LABELS, f"teacher label {j.get('label')!r} not in banking77"
        assert j.get("low_confidence") is False
        return f"{fault} -> teacher label={j['label']} ({r.ms:.0f} ms)"

    def teacher_unavailable(self) -> Any:
        why = self._needs_faults()
        if why:
            return ("SKIP", why)
        policy = self.target.get("teacher_unavailable_policy", "degrade")
        r = self.classify(CORE_QUERY, fault="low_confidence,teacher_unavailable")
        j = r.json() or {}
        if policy == "fail":
            assert r.status == 503, f"policy=fail expected 503, got {r.status}"
            assert r.headers.get("retry-after", "").isdigit(), "503 without Retry-After"
            return "policy=fail -> 503 + Retry-After"
        assert r.status == 200, f"policy=degrade expected 200, got {r.status} {r.body[:120]!r}"
        assert j.get("teacher_unavailable") is True and j.get("low_confidence") is True, f"flags: {j}"
        assert j.get("source") == "student_degraded" and j.get("label") in LABELS, f"{j}"
        return f"policy=degrade -> 200 student answer flagged low_confidence (label={j['label']})"

    def kill_switch(self) -> Any:
        if not self.target.get("fault_injection"):
            return ("SKIP", "needs fault injection to force an escalation")
        if not self.admin_key:
            if self.target.get("teacher_enabled") is False:
                return self._student_only("teacher disabled by env")
            return ("SKIP", "no SMOKE_ADMIN_KEY: runtime kill switch not exercised")
        ah = {"x-admin-key": self.admin_key}
        calls0 = self._teacher_calls(ah)
        off = self.req("POST", "/admin/kill-switch", b'{"teacher_enabled": false}', ah)
        assert off.status == 200 and (off.json() or {}).get("teacher_enabled") is False, (
            f"off -> {off.status}"
        )
        try:
            rz = self.req("GET", "/readyz").json() or {}
            assert rz.get("teacher_enabled") is False, f"readyz still reports teacher on: {rz}"
            detail = self._student_only("runtime kill switch")
            calls1 = self._teacher_calls(ah)
            assert calls1 == calls0, f"teacher was called while disabled ({calls0} -> {calls1})"
        finally:
            on = self.req("POST", "/admin/kill-switch", b'{"teacher_enabled": true}', ah)
        assert on.status == 200 and (on.json() or {}).get("teacher_enabled") is True, "could not re-enable"
        return f"{detail}; 0 teacher calls while off; re-enabled"

    def _student_only(self, why: str) -> str:
        r = self.classify(CORE_QUERY, fault="low_confidence")
        j = r.json() or {}
        assert r.status == 200, f"student-only path -> {r.status} {r.body[:120]!r}"
        assert j.get("source") == "student_degraded" and j.get("low_confidence") is True, f"{j}"
        assert j.get("label") in LABELS
        ok = self.classify(CORE_QUERY).json() or {}
        assert ok.get("source") == "student", f"confident query not served by student: {ok}"
        return f"{why}: low-confidence -> student_degraded flagged, confident -> student"

    def _teacher_calls(self, ah: dict[str, str]) -> int:
        r = self.req("GET", "/admin/stats", headers=ah)
        assert r.status == 200, f"/admin/stats -> {r.status}"
        return int(((r.json() or {}).get("router") or {}).get("teacher_calls", 0))

    def rate_limit(self, burst_max: int) -> str:
        def one(_: int) -> Resp:
            return self.req("POST", "/v1/classify", b"{}")  # cheap: rejected before any model call

        seen: dict[int, int] = {}
        hit: Resp | None = None
        with cf.ThreadPoolExecutor(max_workers=16) as ex:
            sent = 0
            while sent < burst_max and hit is None:
                batch = list(ex.map(one, range(min(32, burst_max - sent))))
                sent += len(batch)
                for r in batch:
                    seen[r.status] = seen.get(r.status, 0) + 1
                    if r.status == 429 and hit is None:
                        hit = r
        assert hit is not None, f"no 429 after {sent} requests: {seen}"
        ra = hit.headers.get("retry-after", "")
        assert ra.isdigit() and int(ra) >= 1, f"429 without a valid Retry-After ({ra!r})"
        return f"429 after {sent} requests, Retry-After={ra}s, statuses={seen}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--burst-max", type=int, default=600, help="max requests sent to trip the rate limiter")
    ap.add_argument("--skip-rate-limit", action="store_true", help="e.g. against prod during real traffic")
    ap.add_argument("--label", default="", help="free-text label stored in the JSON summary")
    a = ap.parse_args(argv)

    s = Smoke(
        a.base_url.rstrip("/"),
        os.environ.get("SMOKE_API_KEY") or None,
        os.environ.get("SMOKE_ID_TOKEN") or None,
        os.environ.get("SMOKE_ADMIN_KEY") or None,
        a.timeout,
    )
    t0 = time.time()
    print(f"smoke: {s.base}")
    s.run("liveness /healthz", s.liveness)
    s.run("readiness /readyz (deps checked)", s.readiness)
    s.run("security headers", s.security_headers)
    s.run("errors carry no stack traces", s.no_stack_traces)
    s.run("auth rejected without/wrong key", s.auth_rejected)
    s.run("auth accepted with key", s.auth_accepted)
    s.run("input validation -> 422", s.validation)
    s.run("oversize input rejected", s.oversize)
    s.run("classify: valid label + confidence", s.core_classify)
    s.run("low confidence escalates", lambda: s.escalate("low_confidence"))
    s.run("invalid student output escalates", lambda: s.escalate("invalid_output"))
    s.run("teacher unavailable path", s.teacher_unavailable)
    s.run("kill switch: student-only path", s.kill_switch)
    if a.skip_rate_limit:
        s.checks.append(Check("rate limit -> 429 + Retry-After", "SKIP", "--skip-rate-limit"))
        print("SKIP  rate limit -> 429 + Retry-After (--skip-rate-limit)")
    else:
        s.run("rate limit -> 429 + Retry-After", lambda: s.rate_limit(a.burst_max))

    counts = {k: sum(c.status == k for c in s.checks) for k in ("PASS", "FAIL", "SKIP")}
    summary = {
        "base_url": s.base,
        "label": a.label,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "duration_s": round(time.time() - t0, 2),
        "passed": counts["PASS"],
        "failed": counts["FAIL"],
        "skipped": counts["SKIP"],
        "total": len(s.checks),
        "ok": counts["FAIL"] == 0,
        "target": {
            k: s.target.get(k)
            for k in (
                "env",
                "version",
                "teacher_enabled",
                "spend_store",
                "fault_injection",
                "teacher_unavailable_policy",
            )
        },
        "checks": [asdict(c) for c in s.checks],
    }
    print(
        f"\n{'PASS' if summary['ok'] else 'FAIL'}: {counts['PASS']} passed, {counts['FAIL']} failed, "
        f"{counts['SKIP']} skipped of {len(s.checks)} in {summary['duration_s']} s"
    )
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

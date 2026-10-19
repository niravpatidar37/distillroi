"""Confidence router: student first, escalate to the teacher on low confidence or invalid output.

``decide`` is a pure function (unit-tested); ``Router`` wires the two clients, the kill switch, the spend cap
and the teacher circuit breaker together. The escalation threshold is tuned on the VALIDATION split
(scripts/router_eval.py), never on test.

Teacher-unavailable path: when an escalation cannot reach the teacher (breaker open, HTTP/5xx after retries,
daily quota, spend cap, deadline), ``RoutedResult.teacher_unavailable`` is set and the student's answer (if it
is a valid label) is returned with ``source="student_degraded"``; the API then applies
TEACHER_UNAVAILABLE_POLICY (degrade -> 200 flagged low_confidence, fail -> 503 + Retry-After). Each occurrence
logs ``teacher_unavailable``, which the Terraform log-based metric + alert policy watch.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import httpx

from distill.spend import BudgetExceeded
from distill.student import StudentResult
from distill.teacher import DailyQuotaExhausted, TeacherError, Usage

log = logging.getLogger("distill.router")

# Fault names accepted in the X-Distill-Fault header (only when FAULT_INJECTION_ENABLED and ENV != prod).
FAULTS = frozenset({"low_confidence", "invalid_output", "teacher_unavailable"})


class Reason(StrEnum):
    ACCEPTED = "accepted"
    LOW_CONFIDENCE = "low_confidence"
    INVALID_OUTPUT = "invalid_output"
    STUDENT_ERROR = "student_error"


def decide(student: StudentResult | None, threshold: float) -> Reason:
    if student is None:
        return Reason.STUDENT_ERROR
    if student.label is None:
        return Reason.INVALID_OUTPUT
    if student.confidence < threshold:
        return Reason.LOW_CONFIDENCE
    return Reason.ACCEPTED


@dataclass
class RoutedResult:
    label: str | None
    source: str  # "student" | "teacher" | "student_degraded" | "none"
    reason: Reason
    student_label: str | None
    student_confidence: float | None
    latency_ms: float
    teacher_usd: float = 0.0
    teacher_unavailable: bool = False  # escalation wanted but the teacher could not answer

    @property
    def low_confidence(self) -> bool:
        """True when the served label did not pass the student's confidence/validity gate nor the teacher."""
        return self.source == "student_degraded"


class StudentLike(Protocol):
    async def classify(self, query: str) -> StudentResult: ...


class TeacherLike(Protocol):
    async def classify_one(self, query: str) -> tuple[str | None, Usage]: ...


class TeacherBreaker:
    """Consecutive-failure circuit breaker: a dead teacher costs one fast failure, not a timeout per request.

    closed -> (``failures`` consecutive errors) -> open for ``cooldown_s`` (or the provider's quota reset
    hint, capped at 1 h) -> half-open: one probe is let through; success closes, failure re-opens.
    """

    def __init__(
        self, failures: int = 3, cooldown_s: float = 30.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.failures = failures
        self.cooldown_s = cooldown_s
        self.clock = clock
        self._consecutive = 0
        self._open_until = 0.0
        self._probe_inflight = False

    @property
    def state(self) -> str:
        if self._consecutive < self.failures:
            return "closed"
        return "half_open" if self.clock() >= self._open_until else "open"

    def allow(self) -> bool:
        st = self.state
        if st == "closed":
            return True
        if st == "half_open" and not self._probe_inflight:
            self._probe_inflight = True
            return True
        return False

    def success(self) -> None:
        self._consecutive = 0
        self._probe_inflight = False

    def abandon(self) -> None:
        """A probe was cancelled (client went away): let the next request probe instead."""
        self._probe_inflight = False

    def failure(self, retry_in_s: float | None = None, *, open_now: bool = False) -> None:
        self._probe_inflight = False
        self._consecutive = max(self._consecutive + 1, self.failures if open_now else 0)
        if self._consecutive >= self.failures:
            hold = self.cooldown_s if retry_in_s is None else min(3600.0, max(self.cooldown_s, retry_in_s))
            self._open_until = self.clock() + hold


def _unavailable_cause(e: BaseException) -> str:
    if isinstance(e, BudgetExceeded):
        return "budget_exceeded"
    if isinstance(e, DailyQuotaExhausted):
        return "quota_exhausted"
    if isinstance(e, TimeoutError):
        return "deadline"
    return "error"


class Router:
    def __init__(
        self,
        student: StudentLike,
        teacher: TeacherLike | None,
        threshold: float,
        *,
        teacher_enabled: bool = True,
        teacher_max_concurrency: int = 4,
        teacher_deadline_s: float = 12.0,
        breaker: TeacherBreaker | None = None,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.threshold = threshold
        self._teacher_enabled = teacher_enabled
        self._teacher_sem = asyncio.Semaphore(teacher_max_concurrency)
        self.teacher_deadline_s = teacher_deadline_s
        self.breaker = breaker or TeacherBreaker()
        self.stats: Counter[str] = Counter()

    @property
    def teacher_enabled(self) -> bool:
        return self._teacher_enabled and self.teacher is not None

    def set_teacher_enabled(self, enabled: bool) -> None:
        """Runtime kill switch (per instance). TEACHER_ENABLED in the environment is the durable switch."""
        self._teacher_enabled = enabled
        log.warning("kill_switch", extra={"teacher_enabled": enabled})

    async def _call_teacher(self, query: str, faults: frozenset[str]) -> tuple[str | None, Usage]:
        if "teacher_unavailable" in faults:
            raise TeacherError("fault injected: teacher_unavailable")
        if self.teacher is None:
            raise TeacherError("teacher not configured")
        if not self.breaker.allow():
            raise TeacherError("teacher circuit breaker open")
        try:
            async with asyncio.timeout(self.teacher_deadline_s), self._teacher_sem:
                self.stats["teacher_calls"] += 1
                out = await self.teacher.classify_one(query)
        except DailyQuotaExhausted as e:
            self.breaker.failure(e.retry_in_s, open_now=True)  # no point retrying before the quota resets
            raise
        except (BudgetExceeded, TeacherError, httpx.HTTPError, TimeoutError):
            self.breaker.failure()
            raise
        except asyncio.CancelledError:
            self.breaker.abandon()
            raise
        self.breaker.success()
        return out

    async def classify(self, query: str, faults: frozenset[str] = frozenset()) -> RoutedResult:
        t0 = time.perf_counter()
        self.stats["requests"] += 1
        s: StudentResult | None
        try:
            s = await self.student.classify(query)
        except (httpx.HTTPError, ValueError) as e:
            log.warning("student_error", extra={"error": type(e).__name__})
            s = None
        if s is not None and faults:
            if "invalid_output" in faults:
                s = dataclasses.replace(s, label=None, raw="<fault:invalid_output>")
            if "low_confidence" in faults:
                s = dataclasses.replace(s, confidence=0.0)
        reason = decide(s, self.threshold)
        self.stats[f"reason_{reason.value}"] += 1
        s_label = s.label if s else None
        s_conf = s.confidence if s else None

        def done(label: str | None, source: str, usd: float = 0.0, unavailable: bool = False) -> RoutedResult:
            self.stats[f"source_{source}"] += 1
            return RoutedResult(
                label, source, reason, s_label, s_conf, (time.perf_counter() - t0) * 1000, usd, unavailable
            )

        def fallback(usd: float = 0.0, unavailable: bool = False) -> RoutedResult:
            return done(s_label, "student_degraded" if s_label else "none", usd, unavailable)

        if reason is Reason.ACCEPTED:
            return done(s_label, "student")
        if not self.teacher_enabled:
            # kill switch on: serve the student's answer if it is schema-valid (flagged low_confidence)
            return fallback()
        self.stats["escalations"] += 1
        try:
            t_label, usage = await self._call_teacher(query, faults)
        except (BudgetExceeded, TeacherError, httpx.HTTPError, TimeoutError) as e:
            self.stats["teacher_unavailable"] += 1
            log.warning(
                "teacher_unavailable",
                extra={
                    "error": type(e).__name__,
                    "cause": _unavailable_cause(e),
                    "breaker": self.breaker.state,
                },
            )
            return fallback(unavailable=True)
        if t_label is None:
            return fallback(usage.usd)
        return done(t_label, "teacher", usage.usd)


def simulate_routing(
    student_labels: Sequence[str | None],
    student_conf: Sequence[float],
    teacher_labels: Sequence[str | None],
    threshold: float,
) -> tuple[list[str | None], float]:
    """Offline replay of ``decide`` over precomputed predictions -> (routed labels, escalation rate)."""
    out: list[str | None] = []
    esc = 0
    for sl, sc, tl in zip(student_labels, student_conf, teacher_labels, strict=True):
        r = decide(StudentResult(sl, sl or "", sc, 0.0), threshold)
        if r is Reason.ACCEPTED:
            out.append(sl)
        else:
            esc += 1
            out.append(tl if tl is not None else sl)
    return out, (esc / len(out) if out else 0.0)


def tune_threshold(
    gold: Sequence[str],
    student_labels: Sequence[str | None],
    student_conf: Sequence[float],
    teacher_labels: Sequence[str | None],
    quality_target: float = 0.99,
    grid: Sequence[float] | None = None,
) -> dict[str, float]:
    """Lowest-escalation threshold whose routed accuracy reaches ``quality_target`` x teacher accuracy.

    Falls back to the most accurate threshold if the target is unreachable. Run on VALIDATION data only.
    """
    from distill.metrics import accuracy

    grid = list(grid) if grid is not None else [i / 100 for i in range(0, 100)] + [0.995, 0.999]
    teacher_acc = accuracy(gold, teacher_labels)
    best_ok: dict[str, float] | None = None
    best_any: dict[str, float] | None = None
    for t in grid:
        routed, esc = simulate_routing(student_labels, student_conf, teacher_labels, t)
        acc = accuracy(gold, routed)
        row = {
            "threshold": t,
            "routed_accuracy": acc,
            "escalation_rate": esc,
            "teacher_accuracy": teacher_acc,
        }
        if best_any is None or acc > best_any["routed_accuracy"]:
            best_any = row
        if acc >= quality_target * teacher_acc and (best_ok is None or esc < best_ok["escalation_rate"]):
            best_ok = row
    chosen = best_ok or best_any
    if chosen is None:
        raise ValueError("empty threshold grid")
    return {**chosen, "target_met": float(best_ok is not None), "quality_target": quality_target}

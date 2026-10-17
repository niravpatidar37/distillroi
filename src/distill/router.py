"""Confidence router: student first, escalate to the teacher on low confidence or invalid output.

``decide`` is a pure function (unit-tested); ``Router`` wires the two clients, the kill switch and the
spend cap together. The escalation threshold is tuned on the VALIDATION split (scripts/router_eval.py),
never on test.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import httpx

from distill.spend import BudgetExceeded
from distill.student import StudentResult
from distill.teacher import TeacherError, Usage

log = logging.getLogger("distill.router")


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


class StudentLike(Protocol):
    async def classify(self, query: str) -> StudentResult: ...


class TeacherLike(Protocol):
    async def classify_one(self, query: str) -> tuple[str | None, Usage]: ...


class Router:
    def __init__(
        self,
        student: StudentLike,
        teacher: TeacherLike | None,
        threshold: float,
        *,
        teacher_enabled: bool = True,
        teacher_max_concurrency: int = 4,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.threshold = threshold
        self.teacher_enabled = teacher_enabled and teacher is not None
        self._teacher_sem = asyncio.Semaphore(teacher_max_concurrency)

    async def classify(self, query: str) -> RoutedResult:
        t0 = time.perf_counter()
        s: StudentResult | None
        try:
            s = await self.student.classify(query)
        except (httpx.HTTPError, ValueError) as e:
            log.warning("student_error", extra={"error": type(e).__name__})
            s = None
        reason = decide(s, self.threshold)
        s_label = s.label if s else None
        s_conf = s.confidence if s else None

        def done(label: str | None, source: str, usd: float = 0.0) -> RoutedResult:
            return RoutedResult(
                label, source, reason, s_label, s_conf, (time.perf_counter() - t0) * 1000, usd
            )

        if reason is Reason.ACCEPTED:
            return done(s_label, "student")
        if not self.teacher_enabled or self.teacher is None:
            # kill switch on: serve the student's answer if it is at least schema-valid
            return done(s_label, "student_degraded" if s_label else "none")
        try:
            async with self._teacher_sem:
                t_label, usage = await self.teacher.classify_one(query)
        except (BudgetExceeded, TeacherError, httpx.HTTPError) as e:
            log.warning("teacher_unavailable", extra={"error": type(e).__name__})
            return done(s_label, "student_degraded" if s_label else "none")
        if t_label is None:
            return done(s_label, "student_degraded" if s_label else "none", usage.usd)
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

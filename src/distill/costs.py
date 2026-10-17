"""Unit economics: teacher API cost, self-hosted GPU cost (idle time included) and the break-even volume.

All functions are pure so the maths is unit-tested in CI. Inputs come from pricing.yaml (labelled estimates)
and from measured benchmark files; nothing here is a hard-coded result.

Model
-----
Teacher (API):     cost/day = V * c_t                     c_t = in_tok * p_in + out_tok * p_out
Student (GPU):     cost/day = N(V) * H * 24               H = GPU-instance $/hour, billed whether busy or idle
                   N(V) = max(1, ceil(V / (R * 86400 * u)))
                   R = measured sustainable req/s, u = target utilisation
Routed (student + escalations):  cost/day = N(V) * H * 24 + e * V * c_t   (e = escalation rate)
Break-even (one always-on instance, valid while V* <= capacity):
                   V* = 24 * H / ((1 - e) * c_t)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TeacherPrice:
    usd_per_m_input: float
    usd_per_m_output: float

    def cost(self, input_tokens: float, output_tokens: float) -> float:
        return (input_tokens * self.usd_per_m_input + output_tokens * self.usd_per_m_output) / 1e6


@dataclass(frozen=True)
class GpuPrice:
    gpu_usd_per_s: float
    vcpu: float
    vcpu_usd_per_s: float
    mem_gib: float
    mem_usd_per_gib_s: float

    @property
    def usd_per_hour(self) -> float:
        per_s = self.gpu_usd_per_s + self.vcpu * self.vcpu_usd_per_s + self.mem_gib * self.mem_usd_per_gib_s
        return per_s * 3600.0


def load_pricing(path: str | Path) -> tuple[TeacherPrice, GpuPrice, dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    t, g = raw["teacher"], raw["gpu"]
    teacher = TeacherPrice(float(t["usd_per_m_input"]), float(t["usd_per_m_output"]))
    gpu = GpuPrice(
        float(g["gpu_usd_per_s"]),
        float(g["vcpu"]),
        float(g["vcpu_usd_per_s"]),
        float(g["mem_gib"]),
        float(g["mem_usd_per_gib_s"]),
    )
    return teacher, gpu, raw


def instances_needed(requests_per_day: float, capacity_rps: float, target_util: float = 0.6) -> int:
    if capacity_rps <= 0 or not 0 < target_util <= 1:
        raise ValueError("capacity_rps must be > 0 and 0 < target_util <= 1")
    per_instance_day = capacity_rps * 86400.0 * target_util
    return max(1, math.ceil(requests_per_day / per_instance_day))


def student_cost_per_day(
    requests_per_day: float, gpu_usd_per_hour: float, capacity_rps: float, target_util: float = 0.6
) -> float:
    """Always-on fleet sized for the volume; idle hours are paid for (that is the point)."""
    return instances_needed(requests_per_day, capacity_rps, target_util) * gpu_usd_per_hour * 24.0


def cost_per_1k(cost_per_day: float, requests_per_day: float) -> float:
    if requests_per_day <= 0:
        raise ValueError("requests_per_day must be > 0")
    return cost_per_day / requests_per_day * 1000.0


def routed_cost_per_day(
    requests_per_day: float,
    gpu_usd_per_hour: float,
    capacity_rps: float,
    teacher_cost_per_request: float,
    escalation_rate: float,
    target_util: float = 0.6,
) -> float:
    if not 0.0 <= escalation_rate <= 1.0:
        raise ValueError("escalation_rate must be within [0, 1]")
    gpu = student_cost_per_day(requests_per_day, gpu_usd_per_hour, capacity_rps, target_util)
    return gpu + escalation_rate * requests_per_day * teacher_cost_per_request


def break_even_requests_per_day(
    gpu_usd_per_hour: float, teacher_cost_per_request: float, escalation_rate: float = 0.0
) -> float:
    """Daily volume above which one always-on student instance (+ escalations) beats the teacher API."""
    if teacher_cost_per_request <= 0:
        raise ValueError("teacher_cost_per_request must be > 0")
    if not 0.0 <= escalation_rate < 1.0:
        raise ValueError("escalation_rate must be within [0, 1)")
    return 24.0 * gpu_usd_per_hour / ((1.0 - escalation_rate) * teacher_cost_per_request)


def volume_table(
    volumes: list[float],
    gpu_usd_per_hour: float,
    capacity_rps: float,
    teacher_cost_per_request: float,
    escalation_rate: float,
    target_util: float = 0.6,
) -> list[dict[str, float]]:
    rows = []
    for v in volumes:
        api = v * teacher_cost_per_request
        student = student_cost_per_day(v, gpu_usd_per_hour, capacity_rps, target_util)
        routed = routed_cost_per_day(
            v, gpu_usd_per_hour, capacity_rps, teacher_cost_per_request, escalation_rate, target_util
        )
        rows.append(
            {
                "requests_per_day": v,
                "teacher_api_usd_per_day": api,
                "student_only_usd_per_day": student,
                "routed_usd_per_day": routed,
                "instances": float(instances_needed(v, capacity_rps, target_util)),
                "teacher_usd_per_1k": cost_per_1k(api, v),
                "routed_usd_per_1k": cost_per_1k(routed, v),
            }
        )
    return rows

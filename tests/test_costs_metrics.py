from __future__ import annotations

import math

import pytest

from distill.costs import (
    GpuPrice,
    TeacherPrice,
    break_even_requests_per_day,
    cost_per_1k,
    instances_needed,
    load_pricing,
    routed_cost_per_day,
    student_cost_per_day,
    volume_table,
)
from distill.metrics import accuracy, macro_f1, percentile
from distill.settings import Settings


def test_teacher_price_per_token():
    p = TeacherPrice(0.15, 0.60)
    assert p.cost(1_000_000, 0) == pytest.approx(0.15)
    assert p.cost(1000, 500) == pytest.approx(0.00015 + 0.0003)


def test_gpu_hourly_includes_cpu_and_memory():
    g = GpuPrice(0.0001867, 4, 0.000018, 16, 0.000002)
    assert g.usd_per_hour == pytest.approx((0.0001867 + 4 * 0.000018 + 16 * 0.000002) * 3600)


def test_shipped_pricing_file_loads_and_is_labelled_estimate():
    s = Settings()
    t, g, raw = load_pricing(s.pricing_file)
    assert t.usd_per_m_input > 0 and t.usd_per_m_output > 0 and g.usd_per_hour > 0
    assert raw["teacher"]["source"].startswith("https://")
    assert raw["gpu"]["source"].startswith("https://")


def test_idle_time_is_paid_for():
    # one always-on instance costs the same at 1 request/day as at 10k/day
    assert student_cost_per_day(1, 1.0, 50) == student_cost_per_day(10_000, 1.0, 50) == 24.0
    # so cost per 1k requests explodes at low volume
    assert cost_per_1k(24.0, 1) == pytest.approx(24_000.0)


def test_instances_scale_with_volume_beyond_capacity():
    cap_day = 10 * 86400 * 0.5
    assert instances_needed(cap_day, 10, 0.5) == 1
    assert instances_needed(cap_day + 1, 10, 0.5) == 2
    with pytest.raises(ValueError):
        instances_needed(1, 0, 0.5)


def test_break_even_formula_and_crossing():
    h, c = 1.0, 0.0002  # $1/h GPU, $0.0002 per teacher request
    v = break_even_requests_per_day(h, c)
    assert v == pytest.approx(24 / 0.0002)
    # at the break-even volume (within one instance's capacity) both sides cost the same
    assert routed_cost_per_day(v, h, capacity_rps=1000, teacher_cost_per_request=c, escalation_rate=0) == (
        pytest.approx(v * c)
    )
    # escalations push break-even up: V* = 24H / ((1-e) c)
    assert break_even_requests_per_day(h, c, 0.2) == pytest.approx(v / 0.8)
    with pytest.raises(ValueError):
        break_even_requests_per_day(h, c, 1.0)
    with pytest.raises(ValueError):
        break_even_requests_per_day(h, 0.0)


def test_volume_table_rows_consistent():
    rows = volume_table([1000, 1_000_000], 1.0, 50, 0.0002, 0.1)
    assert rows[0]["teacher_api_usd_per_day"] == pytest.approx(0.2)
    assert rows[0]["routed_usd_per_day"] == pytest.approx(24 + 0.1 * 1000 * 0.0002)
    assert rows[1]["routed_usd_per_1k"] < rows[1]["teacher_usd_per_1k"]


def test_accuracy_and_macro_f1_match_hand_computation():
    gold = ["a", "a", "b", "c"]
    pred = ["a", "b", "b", None]
    assert accuracy(gold, pred) == 0.5
    # a: tp1 fn1 fp0 -> 2/3 ; b: tp1 fp1 -> 2/3 ; c: fn1 -> 0
    assert macro_f1(gold, pred) == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    with pytest.raises(ValueError):
        accuracy(["a"], [])


def test_percentile_linear_interpolation():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert percentile(xs, 50) == pytest.approx(2.5)
    assert percentile(xs, 95) == pytest.approx(3.85)
    assert math.isclose(percentile([7.0], 99), 7.0)

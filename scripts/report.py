"""Assemble the three-axis benchmark (quality, cost per 1K, latency), the break-even volume and the plots.

Reads only measured result files; writes results/benchmark.json, results/benchmark.md and docs/img/*.png.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distill.costs import (  # noqa: E402
    break_even_requests_per_day,
    instances_needed,
    load_pricing,
    routed_cost_per_day,
    student_cost_per_day,
    volume_table,
)
from distill.settings import Settings  # noqa: E402

TARGET_UTIL = 0.6  # plan instances at 60% of the measured knee throughput (headroom for bursts)
IMG = ROOT / "docs/img"


def load(p: str) -> Any:
    path = ROOT / p
    return json.loads(path.read_text()) if path.exists() else None


def lvl(levels: list[dict[str, Any]], c: int) -> dict[str, Any] | None:
    return next((x for x in levels if x["concurrency"] == c), None)


def main() -> None:
    _teacher_price, gpu, pricing_raw = load_pricing(Settings().pricing_file)
    H = gpu.usd_per_hour
    t_sum = load("results/teacher_test_summary.json")
    t_online = load("results/teacher_online_bench.json")
    served = load("results/served_metrics.json")
    sweep = load("results/vllm_sweep.json")
    router = load("results/router_eval.json")
    live = load("results/router_live.json")
    served_cfg = load("results/served_config.json") or {}
    splits = load("results/splits_summary.json")

    c_t = float(t_online["usd_per_request_at_paid_rate"])
    knee = sweep["knee"]
    R = float(knee["throughput_rps"])
    esc = float(router["test_770"]["escalation_rate"])
    be_student = break_even_requests_per_day(H, c_t, 0.0)
    be_routed = break_even_requests_per_day(H, c_t, esc)
    cap_day = R * 86400 * TARGET_UTIL

    def student_per_1k(v: float) -> float:
        return student_cost_per_day(v, H, R, TARGET_UTIL) / v * 1000

    def routed_per_1k(v: float) -> float:
        return routed_cost_per_day(v, H, R, c_t, esc, TARGET_UTIL) / v * 1000

    vols = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    t1 = lvl(t_online["levels"], 1)
    t4 = lvl(t_online["levels"], 4)
    s1 = lvl(sweep["levels"], 1)
    s4 = lvl(sweep["levels"], 4)
    tq = router["test_770"]
    bench: dict[str, Any] = {
        "test_set": "banking77 test split, stratified 10/class = 770 queries, gold labels, untouched until Phase 4",
        "student": served_cfg,
        "train_examples": splits,
        "quality": {
            "teacher": {"accuracy": tq["teacher_accuracy"], "macro_f1": tq["teacher_macro_f1"]},
            "student": {"accuracy": tq["student_accuracy"], "macro_f1": tq["student_macro_f1"]},
            "routed": {"accuracy": tq["routed_accuracy"], "macro_f1": tq["routed_macro_f1"]},
            "student_pct_of_teacher_accuracy": tq["student_accuracy"] / tq["teacher_accuracy"],
            "routed_pct_of_teacher_accuracy": tq["routed_accuracy"] / tq["teacher_accuracy"],
            "student_full_test_3080": served.get("test_full") if served else None,
            "teacher_batched_ceiling_run": t_sum,
        },
        "cost": {
            "teacher_usd_per_request": c_t,
            "teacher_usd_per_1k": c_t * 1000,
            "teacher_mean_tokens": {
                k: t_online[k] for k in ("mean_input_tokens", "mean_output_tokens", "mean_reasoning_tokens")
            },
            "gpu_usd_per_hour": H,
            "gpu_price_source": pricing_raw["gpu"]["source"],
            "gpu_price_label": "estimate - verify against provider pricing page",
            "measured_knee_throughput_rps": R,
            "planning_utilisation": TARGET_UTIL,
            "requests_per_day_per_instance": cap_day,
            "student_usd_per_1k_fully_utilised_lower_bound": H / (R * 3600) * 1000,
            "student_usd_per_1k_by_volume": {str(v): student_per_1k(v) for v in vols},
            "routed_usd_per_1k_by_volume": {str(v): routed_per_1k(v) for v in vols},
            "volume_table": volume_table([float(v) for v in vols], H, R, c_t, esc, TARGET_UTIL),
        },
        "latency_ms": {
            "teacher_c1": t1,
            "teacher_c4": t4,
            "student_c1": s1,
            "student_c4": s4,
            "routed_live": live["latency_ms_all"] if live else None,
        },
        "break_even": {
            "formula": "V* = 24 * H / ((1 - e) * c_t); valid while V* <= one instance's daily capacity",
            "H_gpu_usd_per_hour": H,
            "c_t_teacher_usd_per_request": c_t,
            "student_only_requests_per_day": be_student,
            "routed_requests_per_day": be_routed,
            "escalation_rate_e": esc,
            "fits_one_instance": be_routed <= cap_day,
            "instances_at_break_even": instances_needed(be_routed, R, TARGET_UTIL),
        },
        "router": {
            "threshold": router["threshold"],
            "escalation_rate_test": esc,
            "live": {k: v for k, v in live.items() if k != "rows"} if live else None,
        },
        "hardware_note": (
            "Throughput/latency measured on an RTX 3060 Laptop (6 GB) under Docker Desktop/WSL2; the cost model "
            "prices that throughput at Cloud Run L4 rates (L4 is a faster, 24 GB card, so this is conservative)."
        ),
    }
    (ROOT / "results/benchmark.json").write_text(json.dumps(bench, indent=2) + "\n")

    def pct(x: float) -> str:
        return f"{100 * x:.1f}%"

    q = bench["quality"]
    md = [
        "| System | Accuracy (test, n=770) | Macro-F1 | $ / 1K requests | p50 / p95 latency @ c=1 | p50 / p95 @ c=4 |",
        "|---|---|---|---|---|---|",
        f"| Teacher: gpt-oss-120b on Groq (1 query/request) | {pct(q['teacher']['accuracy'])} | "
        f"{q['teacher']['macro_f1']:.3f} | ${c_t * 1000:.3f} (paid list price) | "
        f"{t1['p50_ms']:.0f} / {t1['p95_ms']:.0f} ms | {t4['p50_ms']:.0f} / {t4['p95_ms']:.0f} ms |",
        f"| Student: {served_cfg.get('name', '?')} on vLLM | {pct(q['student']['accuracy'])} "
        f"({pct(q['student_pct_of_teacher_accuracy'])} of teacher) | {q['student']['macro_f1']:.3f} | "
        f"${student_per_1k(1_000_000):.3f} at 1M/day; ${student_per_1k(100_000):.2f} at 100K/day "
        f"(idle included) | {s1['p50_ms']:.0f} / {s1['p95_ms']:.0f} ms | {s4['p50_ms']:.0f} / {s4['p95_ms']:.0f} ms |",
        f"| Routed: student -> teacher if conf < {router['threshold']} | {pct(q['routed']['accuracy'])} "
        f"({pct(q['routed_pct_of_teacher_accuracy'])} of teacher) | {q['routed']['macro_f1']:.3f} | "
        f"${routed_per_1k(1_000_000):.3f} at 1M/day (escalation {pct(esc)}) | "
        + (
            f"live: {live['latency_ms_all']['p50']:.0f} / {live['latency_ms_all']['p95']:.0f} ms "
            f"(c={live['concurrency']}) |"
            if live
            else "not yet measured |"
        )
        + " |",
    ]
    (ROOT / "results/benchmark.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(json.dumps(bench["break_even"], indent=2))
    plots(bench, sweep, router)


def plots(bench: dict[str, Any], sweep: dict[str, Any], router: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    IMG.mkdir(parents=True, exist_ok=True)
    # vLLM sweep
    lv = sweep["levels"]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    cs = [x["concurrency"] for x in lv]
    ax1.plot(cs, [x["throughput_rps"] for x in lv], "o-", color="tab:blue", label="throughput (req/s)")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("concurrent requests")
    ax1.set_ylabel("req/s", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(cs, [x["p50_ms"] for x in lv], "s--", color="tab:orange", label="p50 ms")
    ax2.plot(cs, [x["p95_ms"] for x in lv], "^--", color="tab:red", label="p95 ms")
    ax2.set_ylabel("latency (ms)")
    ax1.axvline(sweep["knee"]["concurrency"], color="grey", ls=":", label="knee")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    ax1.set_title("vLLM student: throughput vs latency (RTX 3060 Laptop)")
    fig.tight_layout()
    fig.savefig(IMG / "vllm_sweep.png", dpi=120)
    plt.close(fig)

    # break-even
    c = bench["cost"]
    import numpy as np

    vols = np.logspace(3, 7, 200)
    H, c_t, R = c["gpu_usd_per_hour"], c["teacher_usd_per_request"], c["measured_knee_throughput_rps"]
    e = bench["break_even"]["escalation_rate_e"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(vols, vols * c_t, label="teacher API")
    ax.plot(
        vols,
        [routed_cost_per_day(v, H, R, c_t, e, TARGET_UTIL) for v in vols],
        label="self-hosted student + router",
    )
    ax.axvline(bench["break_even"]["routed_requests_per_day"], color="grey", ls=":", label="break-even")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("requests per day")
    ax.set_ylabel("USD per day")
    ax.legend()
    ax.set_title("Break-even: API vs self-hosted (idle GPU time included)")
    fig.tight_layout()
    fig.savefig(IMG / "break_even.png", dpi=120)
    plt.close(fig)

    # router threshold curve (val)
    cv = router["curve_val"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        [x["threshold"] for x in cv],
        [x["accuracy"] for x in cv],
        "o-",
        label="routed accuracy (val, vs gold)",
    )
    ax.plot(
        [x["threshold"] for x in cv], [x["escalation_rate"] for x in cv], "s-", label="escalation rate (val)"
    )
    ax.axvline(router["threshold"], color="grey", ls=":", label="chosen threshold")
    ax.set_xlabel("student confidence threshold")
    ax.legend()
    ax.set_title("Router threshold tuned on validation")
    fig.tight_layout()
    fig.savefig(IMG / "router_threshold.png", dpi=120)
    plt.close(fig)

    # training curves
    runs = sorted(p for p in (ROOT / "results/train").glob("*/steps.csv"))
    if runs:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        for p in runs:
            name = p.parent.name
            with p.open() as fh:
                rows = list(csv.DictReader(fh))
            a1.plot(
                [float(r["epoch"]) for r in rows], [float(r["loss"]) for r in rows], label=name, alpha=0.8
            )
            ep = json.loads((p.parent / "summary.json").read_text())["epochs"]
            a2.plot(
                [x["epoch"] for x in ep],
                [x["val_acc_vs_teacher"] for x in ep],
                "o-",
                label=f"{name} vs teacher",
            )
            a2.plot(
                [x["epoch"] for x in ep], [x["val_acc_vs_gold"] for x in ep], "x--", label=f"{name} vs gold"
            )
        a1.set_yscale("log")
        a1.set_xlabel("epoch")
        a1.set_ylabel("train loss")
        a1.legend()
        a2.set_xlabel("epoch")
        a2.set_ylabel("val accuracy")
        a2.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(IMG / "training_curves.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()

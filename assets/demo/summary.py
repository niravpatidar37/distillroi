"""Print the three-axis result table and recompute the break-even volume from the committed results.

    python assets/demo/summary.py

Reads results/benchmark.json (the Phase-4 benchmark) and recomputes V* with distill.costs, then checks
it against the committed number, so the line printed is the same formula the README uses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from distill.costs import break_even_requests_per_day  # noqa: E402

B, DIM, R, COPPER, CYAN = "\x1b[1m", "\x1b[2m", "\x1b[0m", "\x1b[33m", "\x1b[36m"


def main() -> int:
    bench = json.loads((ROOT / "results/benchmark.json").read_text(encoding="utf-8"))
    q, c, lat, be = bench["quality"], bench["cost"], bench["latency_ms"], bench["break_even"]
    t_acc = q["teacher"]["accuracy"]
    per_1k = c["routed_usd_per_1k_by_volume"]["1000000"], c["student_usd_per_1k_by_volume"]["1000000"]
    rows = [
        (
            "teacher  gpt-oss-120b (Groq)",
            t_acc,
            c["teacher_usd_per_1k"],
            f"{lat['teacher_c1']['p50_ms']:.0f} / {lat['teacher_c1']['p95_ms']:.0f} ms",
        ),
        (
            "student  1.5B q15_r16 (vLLM)",
            q["student"]["accuracy"],
            per_1k[1],
            f"{lat['student_c1']['p50_ms']:.0f} / {lat['student_c1']['p95_ms']:.0f} ms",
        ),
        (
            f"routed   conf < {bench['router']['threshold']} -> teacher",
            q["routed"]["accuracy"],
            per_1k[0],
            f"{lat['routed_live']['p50']:.0f} / {lat['routed_live']['p95']:.0f} ms*",
        ),
    ]
    print(f"{B}banking77 test, n=770 (results/benchmark.json){R}")
    print(f"{DIM}{'system':<36}{'accuracy':>9}{'of teacher':>12}{'$/1K @1M/day':>14}{'p50 / p95':>16}{R}")
    for name, acc, usd, p in rows:
        print(f"{name:<36}{acc:>9.1%}{acc / t_acc:>12.1%}{'$' + format(usd, '.3f'):>14}{p:>16}")
    print(f"{DIM}latency at c=1; *routed = live end-to-end at c=2 incl. real escalations{R}")
    print()
    h, ct, e = be["H_gpu_usd_per_hour"], be["c_t_teacher_usd_per_request"], be["escalation_rate_e"]
    v = break_even_requests_per_day(h, ct, e)
    assert abs(v - be["routed_requests_per_day"]) < 1.0, "recomputed break-even differs from results"
    print(f"{B}break-even{R}  V* = 24*H / ((1-e)*c_t)")
    print(f"  H = ${h:.4f}/GPU-hour (Cloud Run L4, estimate)   c_t = ${ct:.6f}/request   e = {e:.3f}")
    print(
        f"  {COPPER}{B}V* = {v:,.0f} requests/day{R}{COPPER}: above it the routed student is cheaper "
        f"than the API{R}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

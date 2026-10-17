"""Teacher latency + per-request cost in ONLINE mode (one query per request, as the router calls it).

  python scripts/bench_teacher.py --n 24 --concurrency 1 --concurrency 4

Latency = wall time of the successful HTTP attempt (client-side rate-limit waits are excluded and reported
separately as 429 retries in the Groq free tier are an artefact of the free quota, not of the paid API).
Output: results/teacher_online_bench.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distill.costs import load_pricing  # noqa: E402
from distill.data import read_jsonl  # noqa: E402
from distill.metrics import accuracy, percentile  # noqa: E402
from distill.settings import PROJECT, get_settings, load_local_env  # noqa: E402
from distill.spend import SpendLedger  # noqa: E402
from distill.teacher import DailyQuotaExhausted, GroqTeacher, Usage  # noqa: E402


async def main_async(n: int, levels: list[int]) -> None:
    load_local_env()
    s = get_settings()
    assert s.groq_api_key is not None
    price, _, _ = load_pricing(s.pricing_file)
    teacher = GroqTeacher(
        api_key=s.groq_api_key.get_secret_value(),
        base_url=s.groq_base_url,
        model=s.teacher_model,
        price=price,
        ledger=SpendLedger(s.llm_spend_ledger, PROJECT, s.project_spend_cap_usd, s.llm_spend_cap_usd),
        reasoning_effort=s.teacher_reasoning_effort,
        tokens_per_minute=7500,
    )
    rows = read_jsonl(ROOT / "data/processed/test_770.jsonl")
    out: dict[str, object] = {
        "mode": "online, 1 query/request",
        "model": s.teacher_model,
        "reasoning_effort": s.teacher_reasoning_effort,
        "levels": [],
    }
    all_usage: list[Usage] = []
    waits: list[float] = []
    offset = 0
    for c in levels:
        chunk = rows[offset : offset + n]
        offset += n
        sem = asyncio.Semaphore(c)
        res: list[tuple[str | None, Usage]] = [(None, Usage())] * len(chunk)

        async def one(i: int, text: str, sem: asyncio.Semaphore = sem, res: list = res) -> None:  # type: ignore[type-arg]
            for _ in range(
                40
            ):  # free-tier TPD window: wait for it to slide, then retry (latency = successful call)
                try:
                    async with sem:
                        res[i] = await teacher.classify_one(text)
                    return
                except DailyQuotaExhausted as e:
                    waits.append(e.retry_in_s or 600.0)
                    print(f"TPD full; waiting {(e.retry_in_s or 600.0):.0f}s", flush=True)
                    await asyncio.sleep((e.retry_in_s or 600.0) + 10)
            raise RuntimeError("teacher quota never freed")

        await asyncio.gather(*(one(i, str(r["text"])) for i, r in enumerate(chunk)))
        lat = [u.latency_s * 1000 for _, u in res]
        all_usage += [u for _, u in res]
        level = {
            "concurrency": c,
            "n": len(chunk),
            "p50_ms": round(percentile(lat, 50), 1),
            "p95_ms": round(percentile(lat, 95), 1),
            "acc_vs_gold": accuracy([str(r["gold"]) for r in chunk], [lab for lab, _ in res]),
            "invalid": sum(lab is None for lab, _ in res),
        }
        out["levels"].append(level)  # type: ignore[attr-defined]
        print(json.dumps(level), flush=True)
    await teacher.aclose()
    k = len(all_usage)
    out["mean_input_tokens"] = sum(u.input_tokens for u in all_usage) / k
    out["mean_output_tokens"] = sum(u.output_tokens for u in all_usage) / k
    out["mean_reasoning_tokens"] = sum(u.reasoning_tokens for u in all_usage) / k
    out["usd_per_request_at_paid_rate"] = sum(u.usd for u in all_usage) / k
    out["total_usd"] = sum(u.usd for u in all_usage)
    out["free_tier_tpd_waits"] = len(waits)
    (ROOT / "results/teacher_online_bench.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k2: v for k2, v in out.items() if k2 != "levels"}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--concurrency", type=int, action="append")
    a = ap.parse_args()
    asyncio.run(main_async(a.n, a.concurrency or [1, 4]))


if __name__ == "__main__":
    main()

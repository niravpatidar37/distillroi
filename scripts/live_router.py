"""Send real test queries through the running router API (student on vLLM + live teacher escalation).

  python scripts/live_router.py --n 100 --concurrency 4      (router on 127.0.0.1:8115)

Measures the live escalation rate, routed accuracy vs gold, end-to-end latency, and the teacher spend of the run
(read from the shared ledger). Output: results/router_live.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distill.data import read_jsonl  # noqa: E402
from distill.metrics import accuracy, percentile  # noqa: E402
from distill.settings import PROJECT, get_settings, load_local_env  # noqa: E402

URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:8115")


def project_spend(ledger: Path) -> float:
    if not ledger.exists():
        return 0.0
    tot = 0.0
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("project") == PROJECT:
            tot += float(row.get("usd", 0.0))
    return tot


async def main_async(n: int, concurrency: int, offset: int) -> None:
    load_local_env()
    ledger = Path(get_settings().llm_spend_ledger)
    spend0 = project_spend(ledger)
    rows = read_jsonl(ROOT / "data/processed/test_770.jsonl")[offset : offset + n]
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = [{} for _ in rows]
    async with httpx.AsyncClient(base_url=URL, timeout=60) as client:
        ready = await client.get("/readyz")
        assert ready.status_code == 200, ready.text

        async def one(i: int, r: dict) -> None:
            async with sem:
                t0 = time.perf_counter()
                resp = await client.post("/v1/classify", json={"text": r["text"]})
                ms = (time.perf_counter() - t0) * 1000
                body = resp.json() if resp.status_code == 200 else {}
                results[i] = {
                    "id": r["id"],
                    "gold": r["gold"],
                    "status": resp.status_code,
                    "ms": ms,
                    "label": body.get("label"),
                    "source": body.get("source"),
                    "reason": body.get("reason"),
                    "conf": body.get("student_confidence"),
                }

        t0 = time.perf_counter()
        await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))
        wall = time.perf_counter() - t0
    by_source: dict[str, list[float]] = {}
    for x in results:
        by_source.setdefault(str(x["source"]), []).append(x["ms"])
    spend = project_spend(ledger) - spend0
    out = {
        "n": len(results),
        "concurrency": concurrency,
        "wall_s": round(wall, 2),
        "escalation_rate": sum(x["source"] == "teacher" for x in results) / len(results),
        "degraded": sum(x["source"] in ("student_degraded", "none") for x in results),
        "routed_accuracy": accuracy([x["gold"] for x in results], [x["label"] for x in results]),
        "student_only_accuracy_on_accepted": accuracy(
            [x["gold"] for x in results if x["source"] == "student"],
            [x["label"] for x in results if x["source"] == "student"],
        )
        if any(x["source"] == "student" for x in results)
        else None,
        "latency_ms": {
            src: {"n": len(v), "p50": round(percentile(v, 50), 1), "p95": round(percentile(v, 95), 1)}
            for src, v in by_source.items()
        },
        "latency_ms_all": {
            "p50": round(percentile([x["ms"] for x in results], 50), 1),
            "p95": round(percentile([x["ms"] for x in results], 95), 1),
        },
        "teacher_usd_this_run": spend,
        "teacher_usd_per_1k_routed_requests": spend / len(results) * 1000,
        "rows": results,
    }
    (ROOT / "results/router_live.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()
    asyncio.run(main_async(a.n, a.concurrency, a.offset))


if __name__ == "__main__":
    main()

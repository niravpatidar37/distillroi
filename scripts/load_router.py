"""Overload probe for the router: fire N requests at a fixed concurrency and record status codes + latency.

    uv run python scripts/load_router.py --base-url http://127.0.0.1:8115 --concurrency 128 --n 1500 \
        --out results/router_overload_local.json

Uses real banking77 queries (validation split text) so the student does real work. The point is the shape under
overload: admitted requests keep a bounded latency and the excess gets a fast 429 + Retry-After instead of every
request slowing down (the c>64 collapse seen in results/vllm_sweep.json).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1)


async def run(base: str, n: int, conc: int, key: str | None) -> dict[str, object]:
    texts = [
        json.loads(x)["text"] for x in (ROOT / "data/processed/val.jsonl").read_text("utf-8").splitlines()
    ]
    headers = {"X-API-Key": key} if key else {}
    codes: Counter[int] = Counter()
    lat: dict[int, list[float]] = {}
    retry_after: Counter[str] = Counter()
    sem = asyncio.Semaphore(conc)
    limits = httpx.Limits(max_connections=conc, max_keepalive_connections=conc)
    async with httpx.AsyncClient(base_url=base, timeout=60, limits=limits) as c:

        async def one(i: int) -> None:
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await c.post("/v1/classify", json={"text": texts[i % len(texts)]}, headers=headers)
                    code = r.status_code
                    if code == 429:
                        retry_after[r.headers.get("retry-after", "missing")] += 1
                except httpx.HTTPError:
                    code = -1
                codes[code] += 1
                lat.setdefault(code, []).append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(n)))
        wall = time.perf_counter() - t0
    ok = lat.get(200, [])
    return {
        "concurrency": conc,
        "requests": n,
        "wall_s": round(wall, 2),
        "status_counts": {str(k): v for k, v in sorted(codes.items())},
        "ok_per_s": round(len(ok) / wall, 1),
        "ok_p50_ms": pct(ok, 0.50),
        "ok_p95_ms": pct(ok, 0.95),
        "ok_p99_ms": pct(ok, 0.99),
        "ok_mean_ms": round(statistics.fmean(ok), 1) if ok else None,
        "rejected_429_p95_ms": pct(lat.get(429, []), 0.95),
        "retry_after_values": dict(retry_after),
        "errors": codes.get(-1, 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8115")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--concurrency", type=int, action="append", required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    rows = []
    for c in a.concurrency:
        row = asyncio.run(run(a.base_url, a.n, c, os.environ.get("SMOKE_API_KEY")))
        print(json.dumps(row))
        rows.append(row)
    if a.out:
        a.out.write_text(json.dumps({"note": a.note, "runs": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Benchmark the vLLM-served student: predictions for val/test (router tuning + served quality) and a
throughput/latency sweep over concurrency to find the knee.

  python scripts/bench_vllm.py predict --split val --split test_770 --split test_full
  python scripts/bench_vllm.py sweep --levels 1,2,4,8,16,32,64,128

Outputs: results/preds/served_<split>.jsonl, results/served_metrics.json, results/vllm_sweep.json
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

from distill.data import read_jsonl, write_jsonl  # noqa: E402
from distill.metrics import accuracy, macro_f1, percentile  # noqa: E402
from distill.student import StudentClient, StudentResult  # noqa: E402

BASE = os.environ.get("STUDENT_BASE_URL", "http://127.0.0.1:8105/v1")
MODEL = os.environ.get("STUDENT_MODEL", "distill-student")
SPLITS = {
    "val": ROOT / "data/processed/val.jsonl",
    "test_770": ROOT / "data/processed/test_770.jsonl",
    "test_full": ROOT / "data/processed/test_full.jsonl",
}


async def run_load(
    client: StudentClient, texts: list[str], concurrency: int
) -> tuple[list[StudentResult | None], float]:
    sem = asyncio.Semaphore(concurrency)
    results: list[StudentResult | None] = [None] * len(texts)

    async def one(i: int, t: str) -> None:
        async with sem:
            try:
                results[i] = await client.classify(t)
            except httpx.HTTPError:
                results[i] = None

    t0 = time.perf_counter()
    await asyncio.gather(*(one(i, t) for i, t in enumerate(texts)))
    return results, time.perf_counter() - t0


async def predict(splits: list[str]) -> None:
    client = StudentClient(
        BASE,
        MODEL,
        timeout_s=60,
        client=httpx.AsyncClient(base_url=BASE, timeout=60, limits=httpx.Limits(max_connections=256)),
    )
    metrics_path = ROOT / "results/served_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    for split in splits:
        rows = read_jsonl(SPLITS[split])
        res, wall = await run_load(client, [str(r["text"]) for r in rows], 32)
        write_jsonl(
            ROOT / f"results/preds/served_{split}.jsonl",
            (
                {
                    "id": r["id"],
                    "gold": r["gold"],
                    "teacher": r.get("teacher"),
                    "student": x.label if x else None,
                    "conf": x.confidence if x else 0.0,
                    "raw": (x.raw[:60] if x else None),
                }
                for r, x in zip(rows, res, strict=True)
            ),
        )
        preds = [x.label if x else None for x in res]
        gold = [str(r["gold"]) for r in rows]
        m = {
            "n": len(rows),
            "acc_vs_gold": accuracy(gold, preds),
            "macro_f1_vs_gold": macro_f1(gold, preds),
            "invalid_rate": sum(p is None for p in preds) / len(preds),
            "errors": sum(x is None for x in res),
            "wall_s_at_c32": round(wall, 2),
        }
        if rows[0].get("teacher") is not None:
            m["acc_vs_teacher"] = accuracy([str(r["teacher"]) for r in rows], preds)
        metrics[split] = m
        print(split, json.dumps(m), flush=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    await client.aclose()


async def sweep(levels: list[int], min_requests: int) -> None:
    # Load-generation text: real banking77 training-split queries that were never teacher-labelled or trained on
    # (pool rows after index 5000), so the sweep never touches the test set.
    texts = [str(r["text"]) for r in read_jsonl(ROOT / "data/processed/pool.jsonl")[5000:]]
    client = StudentClient(
        BASE,
        MODEL,
        timeout_s=120,
        client=httpx.AsyncClient(base_url=BASE, timeout=120, limits=httpx.Limits(max_connections=512)),
    )
    await run_load(client, texts[:64], 16)  # warm-up (CUDA graphs, prefix cache)
    rows = []
    for c in levels:
        n = max(min_requests, 8 * c)
        batch = [texts[i % len(texts)] for i in range(n)]
        res, wall = await run_load(client, batch, c)
        lat = [x.latency_s * 1000 for x in res if x is not None]
        row = {
            "concurrency": c,
            "requests": n,
            "errors": sum(x is None for x in res),
            "wall_s": round(wall, 3),
            "throughput_rps": round(len(lat) / wall, 2),
            "p50_ms": round(percentile(lat, 50), 1),
            "p95_ms": round(percentile(lat, 95), 1),
            "p99_ms": round(percentile(lat, 99), 1),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    await client.aclose()
    base_p95 = rows[0]["p95_ms"]
    # Knee: the highest concurrency whose p95 stays within 2x of the single-stream p95.
    within = [r for r in rows if r["p95_ms"] <= 2 * base_p95 and r["errors"] == 0]
    knee = max(within, key=lambda r: r["concurrency"]) if within else rows[0]
    out = {
        "endpoint": "vLLM OpenAI-compatible, student",
        "knee_rule": "max concurrency with p95 <= 2 x p95(c=1)",
        "knee": knee,
        "max_throughput_rps": max(r["throughput_rps"] for r in rows),
        "levels": rows,
        "hardware": os.environ.get("BENCH_HARDWARE", "NVIDIA RTX 3060 Laptop 6 GB (Docker Desktop WSL2)"),
    }
    (ROOT / "results/vllm_sweep.json").write_text(json.dumps(out, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--split", action="append", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--levels", default="1,2,4,8,16,32,64,128")
    s.add_argument("--min-requests", type=int, default=200)
    a = ap.parse_args()
    if a.cmd == "predict":
        asyncio.run(predict(a.split))
    else:
        asyncio.run(sweep([int(x) for x in a.levels.split(",")], a.min_requests))


if __name__ == "__main__":
    main()

"""Run the teacher over a split. Resumable: each batch is cached by content hash; rerun to continue.

  python scripts/run_teacher.py --split test     # Phase 1: teacher accuracy on the frozen test subset
  python scripts/run_teacher.py --split pool --limit 5000 --wait-hours 12   # Phase 2: label training inputs

Groq's free tier has a rolling 24 h tokens-per-day (TPD) window. With --wait-hours > 0 the run sleeps on a TPD
429 for the time Groq names in "try again in ..." and continues; otherwise it stops cleanly.
The output file is rewritten after every batch (a partial run is always usable). BudgetExceeded stops.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distill.costs import load_pricing  # noqa: E402
from distill.data import read_jsonl, write_jsonl  # noqa: E402
from distill.metrics import accuracy, macro_f1  # noqa: E402
from distill.settings import PROJECT, get_settings, load_local_env  # noqa: E402
from distill.spend import BudgetExceeded, SpendLedger  # noqa: E402
from distill.teacher import BatchResult, DailyQuotaExhausted, GroqTeacher, TeacherError  # noqa: E402

SPLITS = {
    "test": (ROOT / "data/processed/test_770.jsonl", ROOT / "results/teacher_test_770.jsonl"),
    "pool": (ROOT / "data/processed/pool.jsonl", ROOT / "data/processed/teacher_labeled_pool.jsonl"),
}


def _summary(split: str, stats: dict[str, Any], out: list[dict[str, Any]], t0: float) -> None:
    stats["elapsed_s"] = round(time.time() - t0, 1)
    if split == "test" and out:
        gold = [str(r["gold"]) for r in out]
        pred = [r["teacher"] if isinstance(r["teacher"], str) else None for r in out]
        stats["accuracy_vs_gold"] = accuracy(gold, pred)
        stats["macro_f1_vs_gold"] = macro_f1(gold, pred)
    (ROOT / f"results/teacher_{split}_summary.json").write_text(json.dumps(stats, indent=2) + "\n")


async def run(split: str, limit: int | None, batch_size: int, wait_hours: float) -> None:
    load_local_env()
    s = get_settings()
    if not s.groq_api_key:
        raise SystemExit("GROQ_API_KEY missing")
    price, _, _ = load_pricing(s.pricing_file)
    ledger = SpendLedger(s.llm_spend_ledger, PROJECT, s.project_spend_cap_usd, s.llm_spend_cap_usd)
    teacher = GroqTeacher(
        api_key=s.groq_api_key.get_secret_value(),
        base_url=s.groq_base_url,
        model=s.teacher_model,
        price=price,
        ledger=ledger,
        cache_dir=ROOT / "data/cache/teacher",
        reasoning_effort=s.teacher_reasoning_effort,
        tokens_per_minute=int(os.environ.get("TEACHER_TPM", "7500")),
    )
    src, dst = SPLITS[split]
    rows = read_jsonl(src)[: limit or None]
    out: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "split": split,
        "batch_size": batch_size,
        "batches": 0,
        "cached_batches": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "usd": 0.0,
        "items": 0,
        "valid": 0,
        "failed": 0,
        "quota_waits": 0,
        "quota_wait_s": 0.0,
        "stopped": None,
    }
    t0 = time.time()
    deadline = t0 + wait_hours * 3600
    start = 0
    try:
        while start < len(rows):
            chunk = rows[start : start + batch_size]
            res: BatchResult
            try:
                res = await teacher.classify_batch([str(r["text"]) for r in chunk])
            except DailyQuotaExhausted as e:
                wait = min((e.retry_in_s or 600.0) + 15.0, 1800.0)
                if time.time() + wait > deadline:
                    stats["stopped"] = f"daily quota: {e}"
                    break
                stats["quota_waits"] += 1
                stats["quota_wait_s"] += wait
                stamp = time.strftime("%H:%M:%S")
                print(f"[{stamp}] TPD window full at item {start}; sleeping {wait:.0f}s", flush=True)
                await asyncio.sleep(wait)
                continue  # retry the same batch
            except TeacherError as e:
                print(f"batch {start} failed: {e}", flush=True)
                start += batch_size
                continue
            stats["batches"] += 1
            stats["cached_batches"] += int(res.usage.cached)
            if not res.usage.cached:
                for k in ("input_tokens", "output_tokens", "reasoning_tokens", "usd"):
                    stats[k] += getattr(res.usage, k)
            out.extend({**r, "teacher": res.labels.get(i)} for i, r in enumerate(chunk, start=1))
            stats["items"] += len(chunk)
            stats["valid"] += len(res.labels)
            stats["failed"] += len(res.failed)
            write_jsonl(dst, out)
            _summary(split, stats, out, t0)
            if stats["batches"] % 5 == 0:
                print(json.dumps({k: stats[k] for k in ("batches", "items", "valid", "usd")}), flush=True)
            start += batch_size
    except BudgetExceeded as e:
        stats["stopped"] = f"budget: {e}"
    finally:
        await teacher.aclose()
    write_jsonl(dst, out)
    _summary(split, stats, out, t0)
    print(json.dumps(stats, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=sorted(SPLITS), required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--wait-hours", type=float, default=0.0)
    a = ap.parse_args()
    asyncio.run(run(a.split, a.limit, a.batch_size, a.wait_hours))


if __name__ == "__main__":
    main()

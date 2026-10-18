"""Send real banking77 test queries to the running router API and show each routing decision.

    python assets/demo/ask.py assets/demo/queries.txt      (or query ids as arguments)

Queries come from data/processed/test_770.jsonl (gold labels). The router (127.0.0.1:8115, override with
ROUTER_URL) asks the vLLM student first and returns its label, confidence and routing reason. For the
recorded demo the router runs with TEACHER_ENABLED=false (no Groq quota used), so an escalation is shown
as the router's decision plus the teacher's CACHED answer for that query from results/teacher_test_770.jsonl
(the Phase-1 teacher run), clearly labelled as cached.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:8115")
THRESHOLD = float(json.loads((ROOT / "results/router_eval.json").read_text(encoding="utf-8"))["threshold"])

B, DIM, R = "\x1b[1m", "\x1b[2m", "\x1b[0m"
GREEN, RED, COPPER, CYAN = "\x1b[32m", "\x1b[31m", "\x1b[33m", "\x1b[36m"


def load(path: Path) -> dict[str, dict[str, str]]:
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {r["id"]: r for r in rows}


def mark(label: str | None, gold: str) -> str:
    return f"{GREEN}✓{R}" if label == gold else f"{RED}✗{R}"


def main(args: list[str]) -> int:
    ids: list[str] = []
    for a in args:  # query ids, or a .txt file with one id per line (# comments allowed)
        if a.endswith(".txt"):
            lines = (ln.split("#")[0].strip() for ln in (ROOT / a).read_text(encoding="utf-8").splitlines())
            ids += [ln for ln in lines if ln]
        else:
            ids.append(a)
    test = load(ROOT / "data/processed/test_770.jsonl")
    teacher = load(ROOT / "results/teacher_test_770.jsonl")
    accepted = escalated = 0
    with httpx.Client(base_url=URL, timeout=30) as c:
        for qid in ids:
            q = test[qid]
            t0 = time.perf_counter()
            resp = c.post("/v1/classify", json={"text": q["text"]})
            wall = (time.perf_counter() - t0) * 1000
            print(f'{B}"{q["text"].strip()}"{R}  {DIM}gold: {q["gold"]}{R}')
            if resp.status_code == 200:
                body = resp.json()
                conf = body["student_confidence"]
                student = body["label"]
                cmp = ">=" if body["reason"] == "accepted" else "<"
                ok = mark(student, q["gold"])
                print(f"  student  {student:<28} conf {conf:.3f} {cmp} {THRESHOLD:.2f}  {ok}")
                body["router"] = f"source={body['source']} (teacher off for this recording)"
            else:
                detail = resp.json().get("detail", "")
                body = {"reason": "invalid_output", "router": f"HTTP {resp.status_code}: {detail}"}
                print(f"  student  {RED}(output is not one of the 77 labels){R}")
            if body["reason"] == "accepted":
                accepted += 1
                print(f"  {GREEN}-> ACCEPT{R}   answered by the 1.5B student in {wall:.0f} ms")
            else:
                escalated += 1
                t_label = teacher[qid]["teacher"]
                print(f"  {COPPER}-> ESCALATE ({body['reason']}){R}  {DIM}{body['router']}{R}")
                print(
                    f"  teacher  {t_label:<28} {DIM}cached, results/teacher_test_770.jsonl{R}"
                    f"  {mark(t_label, q['gold'])}"
                )
            print()
    n = accepted + escalated
    print(
        f"{CYAN}{n} queries: {accepted} answered by the student, {escalated} escalated "
        f"({escalated / n:.0%} here; 28.1% on all 770 test queries){R}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

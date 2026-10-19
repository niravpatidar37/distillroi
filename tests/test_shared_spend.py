"""Shared (cross-instance) teacher spend cap: Firestore budget leases with compare-and-swap.

The in-memory Firestore below implements only what FirestoreDoc uses (GET document, :commit with an
updateTime / exists precondition) and yields to the event loop on every request, so concurrent instances
really interleave between read and write. Error codes match the emulator (400 FAILED_PRECONDITION /
409 ALREADY_EXISTS).
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import random
import uuid

import httpx
import pytest
import respx

from distill.costs import TeacherPrice
from distill.shared_spend import MICROS, FirestoreDoc, LeaseBudget, to_micros
from distill.spend import BudgetExceeded, SpendLedger
from distill.teacher import GroqTeacher


class MemFirestore:
    def __init__(self, *, honour_preconditions: bool = True, seed: int = 0) -> None:
        self.docs: dict[str, dict] = {}
        self.honour = honour_preconditions
        self.rng = random.Random(seed)
        self._ver = itertools.count(1)
        self.commits = 0

    async def handler(self, request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(self.rng.random() * 0.002)  # interleave concurrent clients
        path = request.url.path.removeprefix("/v1/")
        if request.method == "GET":
            doc = self.docs.get(path)
            if doc is None:
                return httpx.Response(404, json={"error": {"code": 404, "status": "NOT_FOUND"}})
            return httpx.Response(200, json=doc)
        assert path.endswith("/documents:commit")
        (w,) = json.loads(request.content)["writes"]
        name = w["update"]["name"]
        pre = w.get("currentDocument") or {}
        cur = self.docs.get(name)
        if self.honour:
            if pre.get("exists") is False and cur is not None:
                return httpx.Response(409, json={"error": {"code": 409, "status": "ALREADY_EXISTS"}})
            if "updateTime" in pre and (cur is None or cur["updateTime"] != pre["updateTime"]):
                return httpx.Response(400, json={"error": {"code": 400, "status": "FAILED_PRECONDITION"}})
        # check-and-write is atomic (no await in between), like Firestore's commit
        self.commits += 1
        self.docs[name] = {"name": name, "fields": w["update"]["fields"], "updateTime": f"t{next(self._ver)}"}
        return httpx.Response(200, json={"writeResults": [{}]})

    def granted_micros(self) -> int:
        (doc,) = self.docs.values()
        return int(doc["fields"]["granted_micros"]["integerValue"])


def _doc(fs: MemFirestore | None = None, transport: httpx.AsyncBaseTransport | None = None) -> FirestoreDoc:
    t = transport or httpx.MockTransport(fs.handler)  # type: ignore[union-attr]
    return FirestoreDoc(
        "demo",
        "(default)",
        "spend_caps",
        "p5",
        emulator_host="fs.test",
        client=httpx.AsyncClient(transport=t),
    )


async def _drain(budget: LeaseBudget, rng: random.Random, spent: list[int]) -> None:
    """One instance: reserve a worst-case estimate, 'call', settle the (smaller) actual, until refused."""
    for _ in range(5000):  # bounded so a broken cap fails the assertion instead of hanging
        est = rng.uniform(0.0002, 0.004)
        try:
            await budget.reserve(est)
        except BudgetExceeded:
            return
        await asyncio.sleep(0)
        act = est * rng.uniform(0.2, 1.0)
        await budget.settle(est, act)
        spent.append(to_micros(act))


async def _run_instances(fs: MemFirestore, n: int, cap: float, lease: float, seed: int) -> list[int]:
    spent: list[int] = []
    budgets = [LeaseBudget(_doc(fs), cap, lease) for _ in range(n)]
    rngs = [random.Random(seed * 100 + i) for i in range(n)]
    await asyncio.gather(*(_drain(b, r, spent) for b, r in zip(budgets, rngs, strict=True)))
    for b in budgets:
        await b.aclose()
    return spent


@pytest.mark.parametrize("seed", range(6))
async def test_shared_cap_holds_across_concurrent_instances(seed):
    """Guarantee: total actual spend over ALL instances <= the shared cap (oracle: sum of settled actuals)."""
    cap, lease, n = 0.25, 0.02, 6
    fs = MemFirestore(seed=seed)
    spent = await _run_instances(fs, n, cap, lease, seed)
    assert sum(spent) <= to_micros(cap)
    assert fs.granted_micros() <= to_micros(cap)
    # liveness: the cap is not uselessly strict (stranded budget <= one lease + one estimate per instance)
    assert sum(spent) >= to_micros(cap) * 0.2 * 0.5
    assert len(spent) > 20


async def test_lost_updates_are_detected_when_preconditions_are_ignored():
    """Teeth: with the CAS precondition dropped (mutant), concurrent leases over-grant past the cap."""
    cap, lease, n = 0.05, 0.02, 8
    fs = MemFirestore(honour_preconditions=False, seed=1)
    budgets = [LeaseBudget(_doc(fs), cap, lease) for _ in range(n)]
    await asyncio.gather(*(b.reserve(0.001) for b in budgets), return_exceptions=True)
    leased = sum(b.snapshot()["leased_usd"] for b in budgets)
    assert leased * MICROS > to_micros(cap)  # 8 x 0.02 handed out against a 0.05 cap


async def test_single_instance_cannot_exceed_cap_and_refuses_before_spending():
    fs = MemFirestore()
    b = LeaseBudget(_doc(fs), cap_usd=0.01, lease_usd=0.004)
    for _ in range(3):
        await b.reserve(0.003)
        await b.settle(0.003, 0.003)
    with pytest.raises(BudgetExceeded):
        await b.reserve(0.003)  # 0.009 spent, 0.003 more > 0.01
    assert fs.granted_micros() <= to_micros(0.01)


async def test_store_unreachable_fails_closed():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("firestore down", request=request)

    b = LeaseBudget(_doc(transport=httpx.MockTransport(boom)), cap_usd=5.0)
    with pytest.raises(BudgetExceeded):
        await b.reserve(0.001)
    assert await b.ready() is False


async def test_unused_lease_is_returned_on_shutdown():
    fs = MemFirestore()
    b = LeaseBudget(_doc(fs), cap_usd=1.0, lease_usd=0.1)
    await b.reserve(0.01)
    await b.settle(0.01, 0.004)
    assert fs.granted_micros() == to_micros(0.1)
    await b.aclose()
    (doc,) = fs.docs.values()
    assert fs.granted_micros() == to_micros(0.004)  # only what was spent stays granted
    assert int(doc["fields"]["spent_micros"]["integerValue"]) == to_micros(0.004)


@respx.mock
async def test_teacher_checks_shared_cap_before_any_network_call(tmp_path):
    fs = MemFirestore()
    shared = LeaseBudget(_doc(fs), cap_usd=0.0001)  # smaller than one worst-case estimate
    route = respx.post("https://groq.test/v1/chat/completions").mock(return_value=httpx.Response(500))
    t = GroqTeacher(
        api_key="k",
        base_url="https://groq.test/v1",
        model="openai/gpt-oss-120b",
        price=TeacherPrice(0.15, 0.60),
        ledger=SpendLedger(tmp_path / "l.jsonl", "p5-distill", 5.0, 25.0),
        shared_budget=shared,
        max_attempts=1,
    )
    with pytest.raises(BudgetExceeded):
        await t.classify_one("where is my card")
    assert route.call_count == 0
    await t.aclose()


@respx.mock
async def test_teacher_settles_actual_cost_into_shared_budget(tmp_path):
    fs = MemFirestore()
    shared = LeaseBudget(_doc(fs), cap_usd=1.0, lease_usd=0.01)
    respx.post("https://groq.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"1": "card_arrival"}'}}],
                "usage": {"prompt_tokens": 700, "completion_tokens": 40},
            },
        )
    )
    t = GroqTeacher(
        api_key="k",
        base_url="https://groq.test/v1",
        model="openai/gpt-oss-120b",
        price=TeacherPrice(0.15, 0.60),
        ledger=SpendLedger(tmp_path / "l.jsonl", "p5-distill", 5.0, 25.0),
        shared_budget=shared,
        max_attempts=1,
    )
    label, usage = await t.classify_one("where is my card")
    snap = shared.snapshot()
    assert label == "card_arrival"
    assert snap["reserved_usd"] == 0.0 and snap["spent_usd"] == pytest.approx(usage.usd, abs=1e-6)
    await t.aclose()


# ---------------------------------------------------------------- real Firestore emulator (CI smoke, local)

EMU = os.environ.get("FIRESTORE_EMULATOR_HOST")


@pytest.mark.skipif(not EMU, reason="needs FIRESTORE_EMULATOR_HOST (Firestore emulator)")
async def test_shared_cap_holds_against_the_firestore_emulator():
    doc_id = f"test-{uuid.uuid4().hex[:8]}"
    cap, lease, n = 0.05, 0.005, 4

    def mk() -> FirestoreDoc:
        return FirestoreDoc("demo-distill", "(default)", "spend_caps", doc_id, emulator_host=EMU)

    budgets = [LeaseBudget(mk(), cap, lease) for _ in range(n)]
    spent: list[int] = []
    await asyncio.gather(*(_drain(b, random.Random(i), spent) for i, b in enumerate(budgets)))
    for b in budgets:
        await b.aclose()
    st = await mk().get()
    assert sum(spent) <= to_micros(cap)
    assert st.granted_micros <= to_micros(cap)
    assert st.spent_micros == sum(spent)  # every settled cost reported back on shutdown

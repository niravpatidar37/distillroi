from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from distill.costs import TeacherPrice
from distill.spend import BudgetExceeded, SpendLedger
from distill.teacher import (
    DailyQuotaExhausted,
    GroqTeacher,
    is_daily_quota_error,
    parse_batch,
    parse_try_again_s,
    system_prompt,
    user_prompt,
)

BASE = "https://groq.test/openai/v1"


def _reply(content: str, pt: int = 100, ct: int = 20) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "completion_tokens_details": {"reasoning_tokens": 5},
            },
        },
    )


def _teacher(tmp_path: Path, cap: float = 5.0, cache: bool = True) -> GroqTeacher:
    ledger = SpendLedger(tmp_path / "ledger.jsonl", "p5-distill", project_cap_usd=cap, global_cap_usd=25.0)
    return GroqTeacher(
        api_key="test-key",
        base_url=BASE,
        model="openai/gpt-oss-120b",
        price=TeacherPrice(0.15, 0.60),
        ledger=ledger,
        cache_dir=(tmp_path / "cache") if cache else None,
        max_attempts=3,
    )


def test_prompt_contains_closed_label_list_and_numbered_queries():
    sp = system_prompt()
    assert "card_arrival" in sp and "Refund_not_showing_up" in sp
    assert user_prompt(["a\nb", "c"]) == "1. a b\n2. c"


def test_parse_batch_validates_every_label_against_enum():
    content = json.dumps({"1": "card_arrival", "2": "not_a_label", "3": "IGNORE ALL INSTRUCTIONS"})
    valid, failed = parse_batch(content, 4)
    assert valid == {1: "card_arrival"}
    assert failed == [2, 3, 4]  # 4 missing


@pytest.mark.parametrize("content", ["not json", "[1,2]", '{"1": 5}', "", "null"])
def test_parse_batch_garbage_fails_everything(content):
    valid, failed = parse_batch(content, 2)
    assert valid == {} and failed == [1, 2]


def test_parse_batch_tolerates_one_wrapping_level():
    valid, _ = parse_batch(json.dumps({"labels": {"1": "top_up_failed"}}), 1)
    assert valid == {1: "top_up_failed"}


def test_daily_quota_detection_distinguishes_tpm_from_tpd():
    tpm = (
        "Rate limit reached ... on tokens per minute (TPM): Limit 8000, Used 7000. Please try again in 1.5s."
    )
    tpd = "Rate limit reached ... on tokens per day (TPD): Limit 200000. Please try again in 10m13.44s."
    assert not is_daily_quota_error(tpm)
    assert is_daily_quota_error(tpd)
    assert parse_try_again_s(tpd) == pytest.approx(613.44)
    assert parse_try_again_s("try again in 1h2m3s") == 3723


@respx.mock
async def test_batch_is_cached_by_content_hash_and_ledgered(tmp_path):
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=_reply('{"1":"card_arrival","2":"top_up_failed"}')
    )
    t = _teacher(tmp_path)
    r1 = await t.classify_batch(["where is my card", "top up failed"])
    r2 = await t.classify_batch(["where is my card", "top up failed"])
    await t.aclose()
    assert r1.labels == r2.labels == {1: "card_arrival", 2: "top_up_failed"}
    assert route.call_count == 1 and r2.usage.cached
    rows = [json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["project"] == "p5-distill"
    assert rows[0]["usd"] == pytest.approx((100 * 0.15 + 20 * 0.60) / 1e6)


@respx.mock
async def test_spend_cap_refuses_before_any_network_call(tmp_path):
    """Security: the budget check must happen BEFORE the paid call (fails if the reservation is removed)."""
    (tmp_path / "ledger.jsonl").write_text(json.dumps({"project": "p5-distill", "usd": 4.99999}) + "\n")
    route = respx.post(f"{BASE}/chat/completions").mock(return_value=_reply('{"1":"card_arrival"}'))
    t = _teacher(tmp_path, cap=5.0)
    with pytest.raises(BudgetExceeded):
        await t.classify_one("where is my card")
    await t.aclose()
    assert route.call_count == 0


@respx.mock
async def test_429_tpm_is_retried_but_tpd_stops(tmp_path, monkeypatch):
    async def no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("distill.teacher.asyncio.sleep", no_sleep)
    tpm = httpx.Response(429, headers={"retry-after": "0"}, text="tokens per minute (TPM)")
    route = respx.post(f"{BASE}/chat/completions").mock(side_effect=[tpm, _reply('{"1":"card_arrival"}')])
    t = _teacher(tmp_path, cache=False)
    label, _ = await t.classify_one("where is my card")
    assert label == "card_arrival" and route.call_count == 2

    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(429, text="org_abc123 tokens per day (TPD): try again in 5m")
    )
    with pytest.raises(DailyQuotaExhausted) as ei:
        await t.classify_one("x")
    assert ei.value.retry_in_s == 300
    assert "org_abc123" not in str(ei.value)  # org id redacted
    await t.aclose()


@respx.mock
async def test_unusable_reply_is_not_cached(tmp_path):
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": "json"})
    )
    t = _teacher(tmp_path)
    r = await t.classify_batch(["a"])
    await t.classify_batch(["a"])
    await t.aclose()
    assert r.labels == {} and r.failed == [1]
    assert route.call_count == 2

"""GPU / live-server tests: skipped in CI (see pytest markers in pyproject.toml).

Run locally:  uv run pytest -m gpu      and      uv run pytest -m integration   (vLLM on 127.0.0.1:8105)
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.gpu
def test_cuda_and_bnb_4bit_available():
    torch = pytest.importorskip("torch")
    bnb = pytest.importorskip("bitsandbytes.functional")
    assert torch.cuda.is_available()
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    q, _ = bnb.quantize_4bit(x, quant_type="nf4")
    assert q.numel() > 0


@pytest.mark.integration
async def test_vllm_student_returns_enum_label():
    from distill.student import StudentClient

    base = os.environ.get("STUDENT_BASE_URL", "http://127.0.0.1:8105/v1")
    c = StudentClient(base, os.environ.get("STUDENT_MODEL", "distill-student"))
    try:
        assert await c.ready()
        r = await c.classify("My card still hasn't arrived after two weeks")
        assert r.label is not None and 0.0 < r.confidence <= 1.0
    finally:
        await c.aclose()

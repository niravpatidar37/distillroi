"""FAKE LLM SERVER - TEST/SMOKE USE ONLY. Never deploy. Refuses to start when ENV=prod.

A deterministic stand-in for the two OpenAI-compatible backends the router talks to, so the CI smoke stack
runs with no GPU and no API key:

* student (vLLM): ``POST /v1/chat/completions`` with ``model=distill-student`` -> a banking77 label chosen by
  keyword rules, with vLLM-style token logprobs (confident for a keyword hit, low confidence otherwise); ``GET
  /health`` like vLLM.
* teacher (Groq): the same route with any other model -> ``{"1": "<label>"}`` JSON plus a usage block.

Every response carries ``x-distill-fake: 1``. The router refuses such replies when ENV=prod (student.py,
teacher.py), Settings refuses http backends in prod, and the image stage that contains this file is never the
default build target (Dockerfile ``fake-llm`` stage). Three independent gates; tests cover each.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from distill.labels import canonical_label

FAKE_HEADER = {"x-distill-fake": "1"}
STUDENT_MODEL = os.environ.get("FAKE_STUDENT_MODEL", "distill-student")

# keyword -> label (checked in order; every label is validated against the closed enum at import time)
RULES: list[tuple[str, str]] = [
    (r"\bcard\b.*\b(arriv|come|deliver)", "card_arrival"),
    (r"\b(lost|stolen)\b", "lost_or_stolen_card"),
    (r"\btop[ -]?up\b.*\bfail", "top_up_failed"),
    (r"\brefund\b", "request_refund"),
    (r"\bexchange rate\b", "exchange_rate"),
    (r"\bpin\b", "change_pin"),
    (r"\bcharged twice\b", "transaction_charged_twice"),
    (r"\bclose my account\b|\bterminate\b", "terminate_account"),
]
for _, _lab in RULES:
    assert canonical_label(_lab) == _lab, _lab
DEFAULT_LABEL = "card_arrival"


def refuse_in_prod() -> None:
    if os.environ.get("ENV", "").strip().lower() == "prod":
        raise SystemExit("fake LLM server refuses to run with ENV=prod")


def classify(text: str) -> tuple[str, bool]:
    t = text.lower()
    for pat, lab in RULES:
        if re.search(pat, t):
            return lab, True
    return DEFAULT_LABEL, False


def _last_user(body: dict[str, Any]) -> str:
    msgs = body.get("messages") or []
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


def create_app() -> FastAPI:
    refuse_in_prod()
    app = FastAPI(title="FAKE-LLM (test only)", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "fake": True}, headers=FAKE_HEADER)

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        data = [{"id": STUDENT_MODEL, "object": "model"}, {"id": "fake-teacher", "object": "model"}]
        return JSONResponse({"object": "list", "data": data}, headers=FAKE_HEADER)

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> JSONResponse:
        body = await request.json()
        user = _last_user(body)
        if body.get("model") == STUDENT_MODEL:
            label, hit = classify(user)
            lp = -0.01 if hit else -1.5  # joint prob 0.99 vs 0.22 -> below any sensible threshold
            payload: dict[str, Any] = {
                "id": "fake-student",
                "object": "chat.completion",
                "model": STUDENT_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": label},
                        "logprobs": {"content": [{"token": label, "logprob": lp}]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 3, "total_tokens": 43},
            }
            return JSONResponse(payload, headers=FAKE_HEADER)
        # teacher: numbered queries "1. text" -> {"1": label, ...}
        items = re.findall(r"^(\d+)\.\s*(.*)$", user, flags=re.MULTILINE)
        out = {num: classify(text)[0] for num, text in items}
        content = json.dumps(out)
        payload = {
            "id": "fake-teacher",
            "object": "chat.completion",
            "model": body.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
            "usage": {
                "prompt_tokens": 700,
                "completion_tokens": 30,
                "completion_tokens_details": {"reasoning_tokens": 10},
            },
        }
        return JSONResponse(payload, headers=FAKE_HEADER)

    return app


def main() -> None:
    refuse_in_prod()
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("HOST", "0.0.0.0"),  # noqa: S104 (container port; compose binds 127.0.0.1)  # nosec B104
        port=int(os.environ.get("PORT", "8000")),
        log_level="warning",
        server_header=False,
    )


if __name__ == "__main__":
    main()

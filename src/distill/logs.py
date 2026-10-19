"""Structured JSON logging with PII redaction. Query text is never logged; only its hash and length."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import time
from typing import Any

_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD]"),
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), "[IBAN]"),
    (re.compile(r"\+?\d[\d ()-]{7,}\d"), "[PHONE]"),
    (re.compile(r"(?i)\b(bearer|api[_-]?key|token)\b\s*[:=]?\s*\S+"), r"\1 [SECRET]"),
    (re.compile(r"gsk_[A-Za-z0-9]{20,}"), "[SECRET]"),
]

_STD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


def redact(text: str) -> str:
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text


def query_fingerprint(text: str) -> dict[str, Any]:
    return {"q_sha": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], "q_len": len(text)}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "severity": record.levelname,  # Cloud Logging maps this field to the entry severity
            "logger": record.name,
            "msg": redact(record.getMessage()),
        }
        for k, v in record.__dict__.items():
            if k not in _STD and not k.startswith("_"):
                payload[k] = redact(v) if isinstance(v, str) else v
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(level)

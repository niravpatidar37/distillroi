"""Configuration from environment variables only (12-factor). Secrets are SecretStr and never logged.

Production gates (enforced at startup by ``Settings`` validation, see ``_prod_gates``):
* ``ENV=prod`` refuses fault injection, the Firestore emulator and any plain-http model backend, so the
  smoke stack's fake LLM server (http, test-only) can never serve production traffic.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_ENV_FILE = "F:/.secrets/portfolio.env"
PROJECT = "p5-distill"


def load_local_env() -> None:
    """Load the local secrets file for scripts on the dev box. Containers get env vars injected instead."""
    from dotenv import load_dotenv

    path = Path(os.environ.get("PORTFOLIO_ENV_FILE", DEFAULT_ENV_FILE))
    if path.is_file():
        load_dotenv(path, override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    # deployment environment: dev (laptop) | ci (smoke stack) | staging | prod
    env: Literal["dev", "ci", "staging", "prod"] = "dev"

    # teacher (Groq, OpenAI-compatible)
    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    teacher_model: str = "openai/gpt-oss-120b"
    teacher_reasoning_effort: str = "low"
    teacher_timeout_s: float = 30.0
    teacher_max_concurrency: int = Field(default=4, ge=1, le=64)
    # Kill switch: TEACHER_ENABLED=false stops every paid call; the router then serves student-only.
    teacher_enabled: bool = True
    # Teacher down / out of quota / over budget: "degrade" serves the student's answer flagged
    # low_confidence=true; "fail" returns 503 + Retry-After so the caller can retry or fall back itself.
    teacher_unavailable_policy: Literal["degrade", "fail"] = "degrade"
    teacher_deadline_s: float = Field(default=12.0, gt=0)  # whole escalation incl. retries
    teacher_breaker_failures: int = Field(default=3, ge=1)  # consecutive failures that open the breaker
    teacher_breaker_cooldown_s: float = Field(default=30.0, gt=0)

    # student (vLLM OpenAI-compatible server)
    student_base_url: str = "http://127.0.0.1:8105/v1"
    student_model: str = "distill-student"
    student_timeout_s: float = 10.0
    # On Cloud Run the vLLM service is IAM-gated: attach a Google ID token from the metadata server.
    student_gcp_id_token: bool = False
    router_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    router_api_key: SecretStr | None = None  # optional app-level key (X-API-Key or Authorization: Bearer)
    admin_api_key: SecretStr | None = None  # enables /admin/* (runtime kill switch, stats); unset -> 404

    # overload protection (per instance)
    max_inflight: int = Field(default=64, ge=1)  # concurrent /v1/* requests being processed
    max_queue: int = Field(default=64, ge=0)  # requests allowed to wait for a slot
    queue_timeout_s: float = Field(default=2.0, gt=0)  # max wait for a slot before 429
    rate_limit_rps: float = Field(default=20.0, ge=0)  # per-client token refill; 0 disables
    rate_limit_burst: int = Field(default=40, ge=1)
    max_body_bytes: int = Field(default=8192, ge=1024)

    # Fault injection (X-Distill-Fault header) for smoke tests of the escalation paths. Never in prod.
    fault_injection_enabled: bool = False

    # spend cap: local JSONL ledger (portfolio-wide on the dev box; per instance on Cloud Run) ...
    llm_spend_ledger: str = "F:/.llm-spend/ledger.jsonl"
    llm_spend_cap_usd: float = 25.0
    project_spend_cap_usd: float = 7.0
    pricing_file: str = str(Path(__file__).resolve().parent / "pricing.yaml")
    # ... plus an optional cap SHARED by every instance (Firestore budget leases; see shared_spend.py)
    spend_store: Literal["file", "firestore"] = "file"
    shared_spend_cap_usd: float = Field(default=7.0, ge=0)
    spend_lease_usd: float = Field(default=0.05, gt=0)
    firestore_project: str | None = None
    firestore_database: str = "(default)"
    firestore_emulator_host: str | None = None  # standard emulator env var; refused in prod

    log_level: str = "INFO"

    @field_validator(
        "groq_api_key",
        "router_api_key",
        "admin_api_key",
        "firestore_project",
        "firestore_emulator_host",
        mode="before",
    )
    @classmethod
    def _blank_is_unset(cls, v: object) -> object:
        # compose passes `KEY: ${KEY:-}` as an empty string; an empty key must mean "not configured"
        return None if isinstance(v, str) and not v.strip() else v

    @model_validator(mode="after")
    def _prod_gates(self) -> Settings:
        if self.env == "prod":
            if self.fault_injection_enabled:
                raise ValueError("FAULT_INJECTION_ENABLED must be false when ENV=prod")
            if self.firestore_emulator_host:
                raise ValueError("FIRESTORE_EMULATOR_HOST must not be set when ENV=prod")
            for name in ("student_base_url", "groq_base_url"):
                if not str(getattr(self, name)).startswith("https://"):
                    raise ValueError(f"{name.upper()} must be https when ENV=prod (test servers are http)")
        if self.spend_store == "firestore" and not self.firestore_project:
            raise ValueError("SPEND_STORE=firestore needs FIRESTORE_PROJECT")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

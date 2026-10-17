"""Configuration from environment variables only (12-factor). Secrets are SecretStr and never logged."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
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

    # teacher (Groq, OpenAI-compatible)
    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    teacher_model: str = "openai/gpt-oss-120b"
    teacher_reasoning_effort: str = "low"
    teacher_timeout_s: float = 30.0
    teacher_max_concurrency: int = Field(default=4, ge=1, le=64)
    # Kill switch: TEACHER_ENABLED=false stops every paid call; the router then serves student-only.
    teacher_enabled: bool = True

    # student (vLLM OpenAI-compatible server)
    student_base_url: str = "http://127.0.0.1:8105/v1"
    student_model: str = "distill-student"
    student_timeout_s: float = 10.0
    # On Cloud Run the vLLM service is IAM-gated: attach a Google ID token from the metadata server.
    student_gcp_id_token: bool = False
    router_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    router_api_key: SecretStr | None = None  # optional bearer token for the router API

    # spend cap (shared ledger across the portfolio)
    llm_spend_ledger: str = "F:/.llm-spend/ledger.jsonl"
    llm_spend_cap_usd: float = 25.0
    project_spend_cap_usd: float = 7.0
    pricing_file: str = str(Path(__file__).resolve().parent / "pricing.yaml")

    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

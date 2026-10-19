# syntax=docker/dockerfile:1
# Router API image (CPU only). The student model is served by a separate vLLM container/service.
# Base images are pinned by digest (multi-arch index); Dependabot (docker ecosystem) proposes bumps.
FROM ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 AS uv

FROM python:3.12.15-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
# runtime deps only: no dev group, no `train` extra (torch etc. never enter this image)
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12.15-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS base
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --no-create-home app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER app

# ---------------------------------------------------------------------------------------------------------------
# TEST-ONLY stage: deterministic fake of the vLLM + Groq APIs for the CI smoke stack (no GPU, no key).
# Build it explicitly with `--target fake-llm`; it refuses to start with ENV=prod. Never pushed or deployed.
FROM base AS fake-llm
COPY --chown=app:app smoke/fake_llm.py /app/fake_llm.py
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=10 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]
CMD ["python", "/app/fake_llm.py"]

# ---------------------------------------------------------------------------------------------------------------
# Production router (default target: last stage). Contains no fake/test code.
FROM base AS runtime
ENV PORT=8080 LLM_SPEND_LEDGER=/tmp/ledger.jsonl
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','8080'), timeout=2).status == 200 else 1)"]
CMD ["python", "-m", "distill.app"]

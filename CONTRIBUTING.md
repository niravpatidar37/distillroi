# Contributing

Issues and pull requests are welcome.

- **Setup:** Python 3.12 via uv: `uv sync` (add `--extra train` for the GPU training stack).
- **Before a PR:** `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, `uv run pytest -q`.
  CI also runs bandit, pip-audit, gitleaks, a Docker build with trivy, and Terraform fmt/validate/tflint/checkov.
- **Numbers:** any result in the README must come from a committed file in `results/` produced by a script in
  `scripts/`. Do not edit numbers by hand; a negative result is fine.
- **Never commit** secrets (`.env*`), model weights (`artifacts/`, `*.safetensors`), caches or W&B runs.
- **Security issues:** report privately, see [SECURITY.md](SECURITY.md).
- Use conventional commit messages (`feat:`, `fix:`, `docs:`, ...).

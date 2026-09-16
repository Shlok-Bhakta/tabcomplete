#!/usr/bin/env bash
# Full local smoke: lint + tests + harness validation. CPU-only.
set -euo pipefail
uv run ruff check .
uv run pytest -q
uv run python scripts/colab_smoke.py
git diff --check
echo "smoke OK"

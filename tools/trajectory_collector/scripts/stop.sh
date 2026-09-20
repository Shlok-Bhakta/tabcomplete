#!/usr/bin/env bash
# Stop the trajectory collector (idempotent).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker CLI not found" >&2
  exit 1
fi

docker compose down --remove-orphans || true
echo "collector stopped"

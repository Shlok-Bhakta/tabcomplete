#!/usr/bin/env bash
# Collector status: compose ps + /healthz + `ss -ltnp` bind check (Tailnet-only).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$ROOT_DIR/.env"
  set +a
fi

BIND_ADDR="${TABCOMPLETE_TAILSCALE_BIND_ADDR:-unknown}"
PORT="${TABCOMPLETE_COLLECTOR_PORT:-8787}"

echo "== compose ps =="
docker compose ps || true
echo
echo "== health =="
if curl -fsS --max-time 3 "http://${BIND_ADDR}:${PORT}/healthz" 2>&1; then
  echo
  echo "health: OK (${BIND_ADDR}:${PORT})"
else
  echo
  echo "health: FAILED (${BIND_ADDR}:${PORT})" >&2
fi
echo
echo "== listeners (ss -ltnp, port ${PORT}) =="
if command -v ss >/dev/null 2>&1; then
  ss -ltnp 2>/dev/null | grep -E ":${PORT}\\b" || echo "(no listener on port ${PORT})"
  echo
  if ss -ltnp 2>/dev/null | grep -E "0\\.0\\.0\\.0:${PORT}\\b|:::.*:${PORT}\\b" >/dev/null; then
    echo "WARNING: collector appears published on 0.0.0.0/:: (expected Tailnet-only ${BIND_ADDR})" >&2
  else
    echo "bind check: no 0.0.0.0 publish detected (expected bind ${BIND_ADDR}:${PORT})"
  fi
else
  echo "(ss not available; skipping bind check)"
fi

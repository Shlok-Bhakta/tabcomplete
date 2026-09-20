#!/usr/bin/env bash
# Resolve the Tailnet IPv4 address for this host.
# Usage: resolve-tailscale.sh [hostname]
# Prints the IPv4 address on stdout. Honors TABCOMPLETE_TAILSCALE_HOST if set.
set -euo pipefail

HOST_ARG="${1:-${TABCOMPLETE_TAILSCALE_HOST:-}}"

if ! command -v tailscale >/dev/null 2>&1; then
  echo "error: tailscale CLI not found" >&2
  exit 1
fi

# Prefer the local node's own Tailnet address (stable, no MagicDNS lookup).
if [ -z "$HOST_ARG" ]; then
  exec tailscale ip -4
fi

# Resolve a peer/hostname via `tailscale ip`.
if IP=$(tailscale ip -4 "$HOST_ARG" 2>/dev/null); then
  printf '%s\n' "$IP"
  exit 0
fi

echo "error: unable to resolve Tailscale IPv4 for host '$HOST_ARG'" >&2
exit 1

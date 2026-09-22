# Deployment inventory

Inspected 2026-09-22. Addresses, disk UUID, credentials, and machine-local paths
are intentionally omitted here. Record those in the uncommitted deployment
manifest and restricted environment files.

| Host | Verified environment | Role |
|---|---|---|
| Kiwi | Ubuntu 24.04.1, amd64, 4 CPUs, about 14.6 GiB RAM, Docker Engine and Compose 2.29.7 | One SigNoz stack, external telemetry storage, gateway, official MCP |
| Crabcake | Ubuntu 24.04.4, amd64, 8 CPUs, rootless Podman behind Docker-compatible CLI | Development/evaluation, bounded local collector WAL, fixed reverse-proxy alias |
| MacBook | Reachable through the existing SSH mesh | Independent tailnet client access verification |

Kiwi's intended disk is the existing approximately 5 TB ST5000LM000-2AN1
external drive with an ext4 filesystem. The old `/Volumes/KiwiData` path does not
exist on this Linux host. The verified mount had about 4.2 TiB available before
deployment. UUID and mount/source identity were checked before creating the
application subtree; no disk was formatted, repartitioned, or erased.

Selected unused tailnet ports: gateway 9090, authenticated SigNoz 8080, OTLP gRPC
4317, OTLP HTTP 4318, MCP 8000. Crabcake uses tailnet 9090 for the alias and
localhost 4317/4318 for its local collector. Databases have no published host
ports. Actual Docker mount inspection found and corrected Keeper's unused
image-declared anonymous volume. Current persistent mounts are explicit binds.

Preserved Kiwi services include the playground on 38765 and its internal llama
server, Samba, spatial graffiti, Obsidian sync, Planista, CouchDB, printer
service, TML, volunteer application, Convex, Dockge, MinIO, Nextcloud, PostgreSQL,
and Immich. Their pre-existing LAN/public bindings and tunnel containers were
not changed. Crabcake's trajectory collector remains on 8787. Its existing
Tailscale Serve 443 route remains untouched; the monitoring alias uses direct
tailnet binding instead. No second SigNoz instance runs on crabcake.

Kiwi uses the existing Docker service conventions. Its user manager reported
linger disabled, so the periodic storage health check uses the existing cron
facility instead of assuming a user systemd service survives logout. Crabcake
uses user systemd units for its collector and alias. Container log rotation and
memory limits apply only to the new observability components.

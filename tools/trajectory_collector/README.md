# Trajectory Collector

Local-only editor telemetry collector. LazyVim (Neovim) streams typed edit
/ cursor / prediction events over the tailnet to a Bun+TypeScript server
backed by SQLite. Stored trajectories are raw replayable facts for future
offline dataset construction — see `protocol.md` for the wire contract.

## Architecture

```text
LazyVim plugin (client)
  │  POST /v1/events/batch, /v1/blobs/*, /v1/repository/snapshot
  │  over Tailscale, http://crabcake:8787
  ▼
Docker container (Bun + TypeScript server, port 8787)
  │  bind 100.100.163.102 (tailnet-only)
  ▼
SQLite WAL (/data/collector.sqlite in container)
  = /mnt/ssd/collector-data/collector.sqlite on the host
```

- **Client**: Neovim/LazyVim plugin (owned separately, under `nvim/`).
  Batches events, spools to disk when offline, content-addresses large
  payloads by sha256.
- **Transport**: Tailscale only. The server binds the tailnet IP; there is
  no listener on LAN or public interfaces.
- **Server**: Bun + TypeScript in Docker (compose + Dockerfile owned
  separately). Implements `protocol.md` v1 verbatim.
- **Storage**: single SQLite file in WAL mode on SSD-backed host path.

## Why raw replayable trajectories

This collector deliberately does **not** build training examples, RL pairs,
or labels:

1. Label semantics (what counts as accept? how long is AFTERMATH? what is a
   weak vs strong accept?) are research decisions that change per
   experiment. Baking today's opinion into the collection layer would rot
   every stored trajectory when the opinion changes.
2. Raw `edit` deltas + `buffer_snapshot` anchors + `context_hash` values let
   any future pairing (STATE / PROPOSED / FEEDBACK / AFTERMATH) be computed
   offline, deterministically, more than once.
3. The training-side Python event protocol (`src/tinycomplete/protocol/`)
   is a separate modeling concern; the collector's job is lossless capture.

## Server + Docker setup

Prerequisites: Tailscale up on the host, `crabcake` reachable
(`tailscale status` shows it), Podman running with the `docker` shim
(`docker` commands are served by Podman on this machine), SSD mounted at
`/mnt/ssd`.

```bash
# 1. Create the data directory (persists the DB outside the container)
mkdir -p /mnt/ssd/collector-data

# 2. Start the collector (compose file owned separately)
docker compose up -d collector

# 3. Verify
curl http://crabcake:8787/healthz
curl http://crabcake:8787/v1/stats
```

Expected: `{"status":"ok","protocol_version":1,...}` from `/healthz`.
The container maps `/mnt/ssd/collector-data:/data` so the database file
survives container rebuilds at `/mnt/ssd/collector-data/collector.sqlite`.

## Tailscale-only design, no app auth — read this warning

The server has **no authentication by design**: anyone who can reach the
port can append events and read stats. This is acceptable ONLY because the
server binds exclusively to the tailnet address (`100.100.163.102:8787`)
and the tailnet is private.

> **WARNING: do not expose this server publicly.** Do not add a LAN
> listener, port-forward, reverse proxy, or public DNS name in front of it
> without adding authentication first. If public exposure is ever needed,
> front it with Tailscale Serve/funnel identity headers or a bearer token
> checked on every `/v1/*` route — that work is explicitly out of scope
> for v1.

## Environment variables

| Variable | Used by | Default | Meaning |
|----------|---------|---------|---------|
| `TABCOMPLETE_TAILSCALE_HOST` | client, scripts | `crabcake` | tailnet hostname of the server |
| `TABCOMPLETE_BIND_ADDR` | server | `100.100.163.102` | IP the server binds; keep tailnet-only |
| `TABCOMPLETE_COLLECTOR_PORT` | server, client | `8787` | TCP port |
| `TABCOMPLETE_COLLECTOR_DB` | server | `/data/collector.sqlite` | SQLite path *inside the container* |
| `TABCOMPLETE_COLLECTOR_URL` | client | `http://crabcake:8787` | full base URL the client posts to |

Client plugin config lives in the editor side (`nvim/`); machine-local
overrides go in `tools/trajectory_collector/.env` (git-ignored, never
committed). `.env.example` (at repo root, owned separately) stays
committable as the template.

## IP resolution

- Preferred: set `TABCOMPLETE_COLLECTOR_URL=http://crabcake:8787` (tailnet
  MagicDNS name; follows the node if its IP changes).
- Fallback: `http://100.100.163.102:8787` (numeric bind IP).
- Diagnose: `tailscale status | grep crabcake`, `tailscale ping crabcake`,
  then `curl http://crabcake:8787/healthz` from the client machine.

## LazyVim setup + commands

The plugin lives under `nvim/` (owned separately). Typical setup:

```lua
-- LazyVim plugin spec (illustrative; authoritative config is in nvim/)
{
  "tabcomplete/collector.nvim",
  config = function()
    require("collector").setup({
      url = vim.env.TABCOMPLETE_COLLECTOR_URL or "http://crabcake:8787",
    })
  end,
}
```

Commands (provided by the plugin):

| Command | Effect |
|---------|--------|
| `:TabCompleteCollectorStart` | open session, snapshot repo, begin streaming |
| `:TabCompleteCollectorStop` | flush spool, close session |
| `:TabCompleteCollectorStatus` | session id, queued events, last flush, server stats |
| `:TabCompleteCollectorFlush` | force-flush the spool now |
| `:TabCompleteCollectorSnapshot` | send a `repository/snapshot` manifest on demand |

Session start sends `POST /v1/session/start` followed by a
`POST /v1/repository/snapshot` manifest; editor exit flushes and sends
`POST /v1/session/end`.

## SQLite location

- Host: `/mnt/ssd/collector-data/collector.sqlite` (plus `-wal`/`-shm`
  sidecars while running).
- Container: `/data/collector.sqlite` (`TABCOMPLETE_COLLECTOR_DB`).
- WAL mode is on; readers never block writers. Query a **copy**, never the
  live file: `sqlite3 /tmp/c.sqlite "select ..."` after
  `sqlite3 /mnt/ssd/collector-data/collector.sqlite ".backup /tmp/c.sqlite"`.

## Dedup strategy

Two independent keys, both enforced server-side:

1. `event_id` — client-generated UUID per event. Re-posts (retries,
   spool replays) with a seen `event_id` are dropped and counted in
   `events_duplicate_dropped`.
2. `(session_id, sequence_number)` — sequence numbers are dense per
   session. A colliding pair with a DIFFERENT `event_id` is a client bug
   and returns `409 SEQUENCE_CONFLICT`; with the SAME `event_id` it is a
   retry and is dropped silently.

Blob dedup is by sha256: identical bytes stored once; re-uploads return
`"duplicate": true` and bump `blob_dedup_hits`.

## Repository scan behavior

On session start (and on `:TabCompleteCollectorSnapshot`) the client walks
the repo and posts a manifest to `POST /v1/repository/snapshot`. File
contents travel as blobs; the manifest carries paths + hashes only.

- **Exclusion list** (never scanned, never uploaded): `.git/`,
  `.jj/`, `node_modules/`, `target/`, `dist/`, `build/`, `.venv/`,
  `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`,
  `checkpoints/`, `outputs/`, `secrets/`, `*.sqlite*`, `.env` files,
  `collector-data/`, spool directories.
- **1 MiB cap**: files larger than 1 MiB are listed with their size and
  `"truncated": true`; their contents are NOT uploaded.
- **Binary policy**: a file whose first 8 KiB contains a NUL byte is
  treated as binary — listed with `"binary_skipped": true`, contents never
  uploaded. No extension allow/deny list; the NUL sniff decides.

## Spool behavior (client-side offline queue)

When the server is unreachable, the plugin appends batches to a local
disk spool and retries with backoff.

- Location: plugin data dir (e.g. `~/.local/share/nvim/collector-spool/`);
  never inside the repo.
- **256 MiB cap**: when the spool exceeds 256 MiB, the OLDEST unsent
  batches are dropped first (with a counter surfaced by
  `:TabCompleteCollectorStatus`) so a long offline stretch cannot fill the
  disk. Drops are lossy by policy — the next successful flush begins with
  a fresh `buffer_snapshot` per open file to re-anchor replay.
- Durability: fsync per batch; a client crash loses at most the in-memory
  batch. After any restart the client opens a NEW session (sequence
  numbers are never resurrected across processes).

## Troubleshooting

| Symptom | Check |
|---------|-------|
| `curl http://crabcake:8787/healthz` hangs | `tailscale status` on both ends; `tailscale ping crabcake`; is the container running? (`docker ps`) |
| `404 UNKNOWN_SESSION` on batch | plugin restarted without re-starting the session — run `:TabCompleteCollectorStart` |
| `409 SEQUENCE_CONFLICT` | two writers sharing one session, or sequence resurrection after restart — start a new session |
| `413 BODY_TOO_LARGE` | batch over 10 MiB — shrink batches; large buffers must go via blobs |
| `422 REPLAY_INVALID` | `buffer_snapshot` references a blob never uploaded — check blob upload errors in plugin logs |
| DB locked / slow stats | query a `.backup` copy, not the live file; confirm disk space on `/mnt/ssd` |
| Spool growing unbounded | server down for a while; cap enforced at 256 MiB with oldest-first drops — check status counter |

## Backup

SQLite backup (online-safe, WAL-aware):

```bash
sqlite3 /mnt/ssd/collector-data/collector.sqlite ".backup /mnt/ssd/collector-data/collector-backup-$(date +%F).sqlite"
```

Keep the whole `/mnt/ssd/collector-data` directory (DB + backups) on the
SSD volume; include it in host backup jobs. Container rebuilds only need
the volume re-mounted — no dump/restore step.

## Uninstall

1. Stop the container: `docker compose down` (add `-v` only if you intend
   to drop named volumes; the DB lives on the host bind-mount and is NOT
   removed by this).
2. Disable the Neovim plugin (remove the spec from your LazyVim config).
3. Delete data (irreversible): `rm -rf /mnt/ssd/collector-data` and the
   client spool dir. This destroys all collected trajectories.

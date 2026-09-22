# TabComplete observability

Kiwi runs one SigNoz stack. Crabcake runs a bounded forwarding collector and a
small reverse proxy to Kiwi's Bun monitoring gateway. Only the gateway is
anonymous. Native SigNoz administration requires its normal login.

See [deployment evidence and remaining acceptance](VERIFICATION.md) for the actual
smoke, recovery, storage, and agent-query results.

Machine addresses, volume UUID, endpoints, and credentials belong in restricted
deployment configuration, not this repository. `deployment-manifest.example.json`
describes the non-secret inventory format.

## Versions and supported installation

`casting.yaml` is the maintained installation definition. Foundry **0.2.17**
validates and renders it into `pours/deployment`; never hand-edit those outputs.
All upstream images are digest-pinned in the casting and lock files:

| Component | Version |
|---|---|
| SigNoz, migrator, SigNoz collector | 0.142.1 |
| Official SigNoz MCP server | 0.14.0 |
| ClickHouse and Keeper | 25.12.5 |
| PostgreSQL metastore | 16, digest-pinned |
| Bun gateway | 1.3.14 |
| Crabcake OpenTelemetry contrib collector | 0.161.0 |
| Python OpenTelemetry API/SDK/exporters | 1.44.0 |
| Python semantic conventions | 0.65b0, locked transitively |

The inspected Kiwi is Linux amd64, not macOS or Apple Silicon. This rendering
targets that verified architecture. Reinspect the host and complete image set
before using another architecture. No whole-stack emulation is configured.

Sources: [supported Docker installation](https://signoz.io/docs/install/docker/),
[Foundry](https://github.com/SigNoz/foundry),
[official MCP server](https://github.com/SigNoz/signoz-mcp-server),
[persistent queue semantics](https://github.com/open-telemetry/opentelemetry-collector/tree/v0.161.0/exporter/exporterhelper).

## Safe deployment

Read `AGENTS.md`, inspect branches/worktrees, then inventory the actual hosts.
Check `uname`, Docker context/runtime, containers and ports, Tailscale, RAM,
`findmnt`/`lsblk`, and existing service conventions. Do not assume old mount paths
still identify the external drive. Do not create a missing mount directory.

Create `.env` and `.admin.env` with `scripts/init_local_config.py --help`. They
contain secrets and must remain mode 0600. Its storage arguments must come from
live host inspection. Prepare only the application subtree:

```sh
python3 scripts/storage_guard.py prepare --root "$TABCOMPLETE_OBS_DATA_ROOT" \
  --expected-source "$TABCOMPLETE_OBS_VOLUME_SOURCE" \
  --expected-uuid "$TABCOMPLETE_OBS_VOLUME_UUID"
bash scripts/render.sh
bash scripts/deploy.sh
python3 scripts/bootstrap_signoz.py --directory "$DEPLOYMENT_DIRECTORY" --url "$SIGNOZ_ADMIN_URL"
python3 scripts/provision.py --directory "$DEPLOYMENT_DIRECTORY" --url "$SIGNOZ_ADMIN_URL"
```

Run these from the deployed `infra/observability` directory. `deploy.sh` checks
the real filesystem, UUID, writable mount, free space, and identity marker before
rendering, pulling, building, or starting. Required growing bind mounts disable
host-directory creation. The collector WAL directory must be writable by its
image UID 10001; restrict any ownership adjustment to that application directory.
Keep PostgreSQL's image-managed ownership intact.

All growing backend state lives below the configured external root:
`clickhouse`, `keeper`, `keeper-image-state`, `metastore`, `collector-wal`,
`model-artifacts`, `gateway-state`, and `backups`. Keeper's image-declared second
data path also has an explicit bind, even though its configured state uses
`keeper`. Verify **actual** mounts with `docker inspect`, including image-created
volumes. Never move Docker's global data root or modify unrelated services.

Host publication binds the Tailscale IPv4 address. Databases stay on the private
Compose network. No Funnel, public tunnel, or public collector is configured.
Existing host-level tunnels and services are outside this deployment and remain
untouched. If direct tailnet binding is unavailable on a future host, inspect and
preserve existing Tailscale Serve routes before using a localhost-backed alias.

## Bootstrap and identity

Bootstrap uses supported registration, session, service-account, and role APIs.
The gateway and MCP receive only the `signoz-viewer` service key. The local
maintenance script reads the separate administrator file to provision dashboards
and retention. Viewer dashboard writes were verified to return HTTP 403.
Never print environment files, run unredacted `docker inspect`, or publish
`docker compose config` output that interpolates secrets.

Crabcake's alias uses `gateway/src/alias.ts` and a machine-local `alias.env`.
`scripts/install-alias.sh` installs its user service without touching Tailscale
Serve. It forwards only GET requests to one fixed gateway. It has no database or
credentials. The collector installer reads a restricted `collector.env` with
`TABCOMPLETE_KIWI_OTLP_HTTP_ENDPOINT` and preserves the WAL on restart.

## Instrument a workload

Instrumentation defaults off. Enable it for a process, not by changing scientific
configuration:

```sh
export TABCOMPLETE_OBSERVABILITY_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
uv run python scripts/evaluate_code_benchmark.py --help
```

Full content capture is separately opt-in with
`TABCOMPLETE_OBSERVABILITY_CAPTURE_CONTENT=1`. Use it only for public/synthetic
benchmarks or explicitly authorized personal projects. Do not enable it globally
for unrelated private or employer repositories. Configure
`TABCOMPLETE_ARTIFACT_UPLOAD_URL` and `TABCOMPLETE_ARTIFACT_UPLOAD_TOKEN_FILE` in
restricted local configuration. Credentials never go into spans, browser code,
URLs, or Git.

Hooks cover shared Transformers and HTTP generation, the threaded prediction
runner, causal and next-edit evaluation, long-context scoring/generation,
campaign subprocesses, training scalar logs and validation, checkpoint I/O,
pytest, and playground provider boundaries. No automatic GenAI wrapper is
installed on top of these manual hooks. Existing providers here are synchronous
and non-streaming; first-token latency is absent, not total request latency.

Run IDs persist in local metadata. Attempts are new per process. Thread tasks
get distinct copied contexts. Owned campaign subprocesses inherit only additional
correlation fields, without removing their existing scientific environment.
External model providers do not receive internal baggage or run metadata.
Run start/heartbeat/summary records are short completed spans; missing heartbeats
mean stale or unknown, never proof of failure.

Metric attributes are enforced by `metrics.PERMITTED_METRIC_ATTRIBUTES` and SDK
views. UUIDs, case/request/trace IDs, prompts, paths, full revisions, and exception
messages are trace/log attributes, never metric labels. Duration and observed
first-output histograms use milliseconds, with buckets from 1 ms to 600,000 ms.
Token counts carry `{token}` units. Unknown usage, first-output timing, and cost
stay unknown. Legacy scientific provider fallback values remain unchanged;
telemetry separately records whether usage was actually reported.

Quality failure does not mark an otherwise successful model transport as an
error. Scientific result files remain authoritative. The gateway deduplicates
terminal case IDs or trace/span IDs and separates suite hashes and protocols.
Its bounded recent-event window is explicitly partial when more pages exist.
Percentiles are calculated from individual observed durations, not averaged p95s.

## Artifact and offline limits

Payloads use SHA-256 of stored UTF-8 bytes and gzip compression. Redaction changes
the hash and status. Whitespace is preserved. The default maximum is 8 MiB per
payload. Oversized or unauthorized capture is explicitly omitted. The gateway
checks hashes, enforces a 20 GiB cap and 100 GiB free-space floor, and serves text
with restrictive headers. Uploads are asynchronous and outside inference timing.
Failed transfers leave a bounded local CAS for recovery; a captured local hash is
not proof that transfer succeeded. Verify retrieval before claiming full capture.
With the same restricted upload environment, retry pending files using
`uv run python scripts/observability.py sync-artifacts .tabcomplete-observability/artifacts`.
The command reports accepted/failed counts and returns nonzero for failed delivery.
It preserves the local copies; the default local CAS cap is also 20 GiB.

The collector uses persistent byte-sized sending queues, 128 MiB per exporter
signal queue, retry/backoff, memory limits, and batches of at most 2,048 items.
Storage files have bookkeeping overhead beyond the serialized queue bound.
Delivery is not exactly once. Queue-full and enqueue/export failures must be
investigated, not hidden by scientific totals derived from span counts.

For workers without an already-authorized reachable collector:

```sh
export TABCOMPLETE_OBSERVABILITY_ENABLED=1
export TABCOMPLETE_OBSERVABILITY_MODE=offline
export TABCOMPLETE_OBSERVABILITY_OFFLINE_BUNDLE=outputs/telemetry.jsonl
# Run the normally authorized workload, then collect its outputs.
uv run python scripts/observability.py import-offline outputs/telemetry.jsonl
```

Offline bundles default to 256 MiB. They retain observed span IDs, timestamps,
parents, links, events, status, and attributes. Import marks records historical
and never replays live metric counters. The ledger is committed only after
accepted export; a crash between export and ledger commit can still replay IDs.
Crabcake's uncommitted `~/.config/tabcomplete-observability/client.json` points
offline imports to the SSH/stdin importer on Kiwi. Its canonical ledger lives in
external `gateway-state/imports.sqlite`, covered by metadata backups. Temporary
uploads also use that drive and are removed after successful import. Install
the Python 3.11 importer with `scripts/install-importer.sh SSH_HOST DIRECTORY`
from a development host with uv 0.12.3. Without a remote client configuration,
`--ledger` selects an explicit local test ledger. Do not put Tailscale keys in
notebooks or datasets.

## Dashboard and agents

`dashboards/runs-and-models.json` compiles into native dashboard schema v6 through
the pinned release's v2 API. Provisioning uses a stable dashboard name, updates
the same ID, and validates every panel query. Native administration stays
authenticated. The Bun page refreshes about every five seconds and uses bounded
read-only operations, escaped text, Host/Origin checks, and relative drill-downs.
Event timestamps and ingestion time are distinct; unavailable ingestion time is
reported as unknown.

```sh
export TABCOMPLETE_MONITOR_URL=http://TAILNET_GATEWAY:9090
uv run python scripts/observability.py status
uv run python scripts/observability.py runs --since 24h --json
uv run python scripts/observability.py run RUN_ID --json
uv run python scripts/observability.py failures --run RUN_ID --json
uv run python scripts/observability.py trace TRACE_ID --json
uv run python scripts/observability.py request REQUEST_ID --json
uv run python scripts/observability.py compare RUN_A RUN_B --json
uv run python scripts/observability.py doctor
```

All query commands support `--since`, `--limit`, and `--cursor`. JSON schema
version is 1. Exit codes are 0 success, 2 argument/configuration error, 3 backend
failure, and 4 missing detail. Comparisons flag incompatible and unknown suite,
protocol, device, model, context, and decoding values. Provisioning is local-only:
`observability.py provision --directory DIR --signoz-url URL`.

The official MCP endpoint is `/mcp` on its configured private port. The local
Codex configuration adds `tabcomplete-signoz` without replacing other servers
and allows only query/discovery tools. Credentials stay in Kiwi's MCP process.
See [Codex MCP configuration](https://developers.openai.com/codex/mcp).

## Maintenance and recovery

Retention is explicit: logs 7 days, traces 14 days, metrics 90 days, copied
artifacts 30 days. Repeated artifact use renews its copy's retention timestamp.
Edit `config/retention.json` and run provisioning to change database retention.
Historical scientific manifests are preserved separately from copied payloads.
`scripts/install_health_cron.py DIR` installs a once-a-minute verified external
drive check while preserving unrelated cron entries. Stale/missing health is
shown honestly. The 100 GiB total-data high-water warning is not a deletion rule.
Container logs rotate at 10 MiB with three files. Never delete live ClickHouse
files or prune research checkpoints to recover observability space.

`scripts/backup.sh` uses PostgreSQL's consistent `pg_dump` and copies maintained
definitions plus essential manifests into the external backup directory.
`scripts/restore.sh BACKUP tabcomplete_restore_NAME` restores into a **new**
isolated database and refuses to overwrite an existing database. Promotion is a
separate maintenance operation: stop SigNoz, set its DSN to the restored database
through the maintained casting, rerender, and verify before resuming service.
The running production metastore is never replaced by the restore test.

For full telemetry backups, use ClickHouse's supported BACKUP/RESTORE mechanism,
or stop this complete stack for a cold copy of ClickHouse, Keeper, and metastore
into a verified external backup directory. Do not copy live database files and
claim a consistent backup. The included routine backs up metadata and essential
manifests, not the entire trace/metric history.

## Verification

```sh
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install matplotlib
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync mypy src scripts/observability.py
cd infra/observability/gateway
bun install --frozen-lockfile
bun test
bunx tsc --noEmit --module preserve --target esnext --moduleResolution bundler --allowJs --skipLibCheck src/*.ts tests/*.ts
```

`scripts/observability_smoke.py` runs three deterministic executable cases and
an optional request to an existing local model. `observability_verify.py` checks
the authoritative smoke results against SigNoz through the gateway, every
captured payload hash, and an actual official MCP query. `observability_outage.py`
stops only this deployment's ingester, emits 20 owned test events, restarts in a
finally block, and verifies replay. Never use paid APIs or launch training for
these checks. `observability_overhead.py` measures rather than promises overhead.

A verified deployment also requires restart persistence, exact mount identity,
tailnet access from two clients, a negative LAN binding check, native panel
rendering, skill validation, and preservation of existing services. Container
health alone is insufficient. Record unavailable verification separately from
passed checks.

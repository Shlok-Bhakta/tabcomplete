# Deployment evidence — 2026-09-22

The backend and private gateway are deployed, not merely rendered. **Full visual
acceptance remains blocked:** this session's collaborative browser reports no
preview automation host. Native panel queries and gateway summaries passed;
neither is represented here as proof of browser rendering.

Actual addresses, volume UUID, credentials, and host paths are excluded from Git.
Kiwi's restricted `deployment-manifest.json` records inspected image IDs, mounts,
ports, and volume identity. The delivery message supplies the private URLs.

## Verified deployment

- Supported Foundry 0.2.17 rendering, inspected before startup. Maintained patches
  pin the complete upstream image set by digest. Kiwi is Linux amd64.
- Seven running application containers; five with configured health checks are
  healthy. Ingestion and official MCP were checked with actual telemetry/query
  operations, not inferred healthy from container state.
- All published application ports bind Kiwi's Tailscale IPv4 address. Databases
  have no host publications. The Bun alias on crabcake has no database or key.
- Both private pages returned HTTP 200 from crabcake and an independent MacBook
  tailnet client. Kiwi's LAN-address gateway connection failed (HTTP 000).
- Existing playground returned HTTP 200. The existing local model served actual
  smoke and overhead requests. No paid call, training campaign, or GPU quota used.
- Every active Docker data mount was inspected; no anonymous/named volumes remain
  attached to this stack. Keeper's image-declared second data path was explicitly
  bound after inspection identified it. Temporary directories are bounded tmpfs;
  container logs rotate at 10 MiB × 3.
- External storage is the existing approximately 5 TB ST5000LM000-2AN1 ext4 drive,
  not the absent legacy macOS path. About 4.2 TiB remained free. Guards verify
  source, UUID, marker, writability, and free space. Disposable missing-marker and
  wrong-volume tests passed without unmounting the live disk.
- Retention API confirmed logs 7 days, traces 336 hours, metrics 2160 hours.
  Artifact copies expire after 30 days, capped at 20 GiB; pressure stops copying.

## Smoke, reconciliation, and agent access

Final deterministic smoke run:
`run-5c4041c3-60a7-4699-a596-cf8269c01093`.

| Authoritative result | Dashboard summary using real SigNoz rows |
|---|---|
| Three executable cases | Three evaluated cases |
| Parse passes | 2/3 |
| Compile passes | 2/3 |
| Functional passes | 1/3 |
| Quality failures | Two, plus separately recorded synthetic provider timeout |

Real local-model request:
`request-d66c35e8-e13b-4e0f-8929-a276b4513690`.
Its trace is `b7fcc77b9588b3112b239f64d7df68e7`.
Six unique payloads were retrieved and SHA-256 verified; the real model's full
output matched the authoritative smoke result exactly. Repeated fixture input
was stored once. Explicit artifact resync accepted all six existing hashes.

The official MCP `signoz_execute_builder_query` returned this known run.
CLI status, runs/pagination, run, failures, trace, request, compare, and doctor
all returned schema version 1 with exit code 0. A viewer dashboard-write attempt
returned HTTP 403. The local Codex MCP registration preserves other servers and
allows query/discovery tools only.

Native dashboard `TabComplete — Runs and Models` retained ID
`01a0caad-3460-7b11-958c-fc86b0a0162e` through repeated provisioning. All six
panel queries passed against SigNoz 0.142.1, schema v6. A reconciliation test
caught and fixed null quality fields incorrectly increasing observed-case counts;
the regression is covered by a Bun test.

## Recovery and historical records

- Full container recreation preserved both smoke runs, all payloads, and the
  provisioned dashboard. The earlier run is
  `run-17056f74-f2f0-4ecc-949d-3a2c68a0c3d7`.
- Temporary Kiwi-ingester outage, final run
  `run-6599a645-4de1-4363-b12c-1f3b3e97caf1`: 20 expected events, 20 unique replayed
  events, 20 raw rows. Fixture work continued and took 0.074887 seconds while the
  ingester was stopped. The local collector retried using its persistent queue.
  This observation is not an exactly-once transport guarantee.
- Existing completed research C5 summary imported as
  `historical-8b1ba8e92a085a5ea3073e42`. Reimport returned duplicate=true and zero
  imported spans. The canonical ledger and preserved compact manifest are on
  Kiwi's external drive. No campaign was rerun. Unknown original execution
  timestamps are labeled explicitly; import time is not a training stage timing.
- PostgreSQL consistent metadata backup completed. Restore into the new isolated
  `tabcomplete_restore_verify_r1` database passed without replacing the running
  metastore. Routine backup does not claim to include all ClickHouse telemetry;
  consistent full-history backup options are documented separately.

## Tests and performance

Final results: **209 Python tests passed**, **16 Bun tests passed** (36 assertions),
Ruff passed, mypy passed across 60 source files, TypeScript passed, and the skill
validator passed. The suite includes
disabled behavior, thread/subprocess and owned-HTTP context, success/failure/
timeout/cancellation/retry, unknown usage, unchanged HTTP request bodies/results,
redaction, CAS dedup/whitespace/retry, metric labels and rank accounting, offline
ledger rollback/dedup/IDs, outage isolation, gateway bounds/Host/Origin/path safety,
idempotent dashboard construction, and disposable storage guards.

The repository suite, Ruff, Python type checks, Bun tests, TypeScript checks,
and skill validator passed. An opt-in instrumented pytest session also passed
and exported through the local collector. CPU Torch and matplotlib were installed
for verification; no CUDA assumption or training run was used.

Actual ClickHouse metric-series metadata was audited: project metric dimensions
were `backend` and `token_type`; the backend added `le` for histogram buckets and
`__temporality__`. Resource keys were service name/version, deployment environment,
and host name. No run, case, request, trace, prompt, or path labels were found.
The final metadata backup was restored again into the new isolated
`tabcomplete_restore_verify_r2` database after temporary-storage hardening.

Measured deterministic boundary overhead across 1,000 iterations:
disabled 15.45 µs, enabled 110.84 µs, added 95.38 µs per fixture. Three actual local
requests in each mode had medians 335.35 ms off and 330.55 ms on. This small,
sequential, cache/load-sensitive sample does **not** establish zero overhead;
content capture was off for that latency comparison.

## Operational limits and remaining acceptance

- Browser rendering and interaction still need an available preview host.
- Gateway views are bounded event windows and explicitly show partial data;
  scientific result files remain the benchmark denominator.
- Backend ingestion timestamps, unobserved first-token latency, absent usage,
  missing collector series, and costs remain unknown rather than invented.
- Content capture is opt-in per authorized workload. Failed uploads preserve
  bounded local copies; `sync-artifacts` retries them. A local hash alone is not
  proof of remote availability.
- Storage warnings are visible on the page; no external paging destination was
  configured. No GPU/training-runtime validation was performed or implied.

The skill is committed at `.agents/skills/tabcomplete-observability/SKILL.md`,
referenced from AGENTS.md, and installed in the host's established skill directory.

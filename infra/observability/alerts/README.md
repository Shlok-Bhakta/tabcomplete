# Local monitoring warnings

No external notification destination is configured. Warnings are shown on the
private main page; this does not claim pager delivery.

- Storage identity/mount check failure: unhealthy or stale storage status.
- Health record older than 90 seconds: stale, never silently healthy.
- Total application data at the high-water threshold: warning (default 100 GiB,
  configurable with `TABCOMPLETE_OBS_HIGH_WATER_BYTES`).
- Artifact cap 20 GiB or free disk below 100 GiB: optional copying refused.
- Run heartbeat absent for 45 seconds: stale/unknown, not failed.
- Queue size/capacity and export/enqueue/refusal metrics: shown when observed;
  missing series remain unknown rather than reporting zero drops.

The definitions are implemented in `scripts/storage_guard.py`,
`gateway/src/server.ts`, and `gateway/public/summary.js`. Database retention is
maintained in `config/retention.json` and applied through `scripts/provision.py`.

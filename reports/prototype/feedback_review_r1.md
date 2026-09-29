# Feedback review provenance, 2026-09-29 UTC

The crabcake automatic prototype still serves the adapted q25-coder Q4_K_M
model. This change closes a feedback provenance gap: the old plugin always
marked prediction requests `human_verified=false`, so the existing exporter
could never qualify a real decision for future preference work. The editor now
offers an optional `:TabCompleteReviewLast` confirmation after an explicit
acceptance or rejection. Ordinary typing, automatic suggestion display, and
`<M-l>` acceptance do not require it. Training remains disabled.

## Evidence path

The new `prediction_reviewed` event lives in the existing ordered collector
stream and names the exact resolution event, prediction, outcome, and session.
The event is emitted only after explicit confirmation, and only for the last
unreviewed, nonsynthetic explicit decision in the same session. Cancellation
and repeated confirmation do not emit it. A scripted keymap can still mimic
editor input, so the event is a local reviewer assertion, not cryptographic
proof of a physical human. All scripted live smoke events were marked
synthetic.

Schema migration 3 adds `review_event_id` and `review_status` to the existing
rebuildable prediction projection. Raw events remain authoritative. The
exporter requires a nonsynthetic request and review, a visible focused display,
correct request→display→decision→review order, an exact outcome-event link,
and no gap between decision and review. Missing, duplicated, conflicting,
synthetic, or malformed confirmations stay unverified. No older request flag
alone can qualify a pair.

The installed collector was upgraded before the Neovim client source because
an old server rejects unknown event types and would spool a whole mixed batch.
The collector still binds only `100.100.163.102:8787`; inference still binds
only `127.0.0.1:19093`. No new database or model service was created.

## Migration and deployment

Before deployment, a SQLite backup API copy of the live WAL database was
written to
`/mnt/ssd/collector-data/backups/collector-before-feedback-review-20260929T051244Z.sqlite`.
It passed `PRAGMA integrity_check=ok` with 4,833 events and 70 projections at
schema 2. A disposable copy migrated to schema 3, retained those counts, and
reapplying migration made no changes. The active Neovim configuration was
copied to
`/home/crabcake/.config/nvim/lua/plugins/tabcomplete-trajectory.lua.backup-feedback-20260929T051244Z`.
The old collector image is tagged
`localhost/tabcomplete-trajectory-collector:rollback-20260929T051244Z`.

The new image ID is
`b53fc7236837f451051e658fb12ed7b197dafe072b9040d957696b6b60a405bc`.
After restart it was healthy, the active database passed integrity check,
schema 3 was applied, and its historical 70 projections remained. The stable
LazyVim path stayed
`/home/crabcake/Projects/tabcomplete-automatic-feedback-r1/tools/trajectory_collector/nvim`;
that clean branch was fast-forwarded to commit `c0e9f3d`. A fresh Neovim
startup reported `mode=automatic`, experimental opt-in true, quality validated
false, personalization training false, acceptance key `<M-l>`, and the review
command present.

The selected GGUF remains Q4_K_M SHA-256
`a45bc50aa7c74a03e7bdcade90315b052b31675a97631ed5c96df953160fd626`.
The running CPU-only `llama-server` process and its launch arguments were not
changed. This continuation did not measure new latency or model quality.

## End-to-end checks

An in-memory Neovim test exercised the review command, cancellation, exact
outcome linkage, and duplicate suppression. The automatic predictor test
rejected an attempt to review a synthetic accept. A live synthetic protocol
batch created prediction
`8af82b36-6149-4ff9-b886-c100ed2453f7` in session
`b6aa798e-5666-4ca8-8790-edabea3c0886`: the first delivery ingested four
events, retry ingested zero, and the projection remained `ambiguous` because
the review was explicitly synthetic. No human label was fabricated.

A new headless Neovim session loaded the installed plugin and used the actual
selected model. In disposable session
`8ab91346-7f02-4952-a9c5-1a5dbefb31ca`, automatic typing pauses produced
three displayed proposals. The script accepted one, dismissed one by typing,
and typed a match to one; every action was marked synthetic. The existing
collector replay rebuilt `main.py` from ten anchors and seven deltas with zero
unanchored deltas and zero mismatches, byte-equal to the final anchor. This
verifies plumbing, not human acceptance or general model accuracy.

At final inspection the live database had 4,871 events, 74 projections, one
synthetic review event, **zero confirmed human reviews**, and
`PRAGMA integrity_check=ok`. The existing exporter produced 70 historical
proposal records and zero defensible preference pairs before these new smoke
events. Automatic personalization is still disabled.

## Verification

- Python: 241 passed, one skipped in the combined CPU Torch environment.
- Bun collector: 58 passed; TypeScript typecheck passed.
- Collector Lua: 56 passed; automatic state-machine and explicit-review
  headless tests passed separately.
- Ruff passed repository-wide; mypy passed on 63 source files.
- The disposable schema-2 migration and live schema-3 checks passed.

No GPU session, model download, teacher call, or paid service was used.

## Rollback

The collector data migration is additive. To restore the prior server image,
retag `localhost/tabcomplete-trajectory-collector:rollback-20260929T051244Z`
as `localhost/tabcomplete-trajectory-collector:latest`, then run
`podman-compose up -d --no-build --force-recreate collector` from
`/home/crabcake/Projects/tabcomplete-automatic-feedback-r1/tools/trajectory_collector`.
The older server cannot accept `prediction_reviewed`, so also switch the
Neovim plugin source back to commit `dda0218` or refrain from invoking
`:TabCompleteReviewLast`. Keep the database at schema 3 during an image-only
rollback; the older server ignores the additive columns. The pre-migration
SQLite backup is available only for a separate, deliberate full data restore.

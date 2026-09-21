# Trajectory analysis scripts

Bun scripts that act **directly on the collector SQLite database**. Written for
future models/agents: if you answer a question about collected data, run one of
these instead of hand-rolling SQL (hand-rolled queries produced a wrong answer
once — see "lessons" below).

## The read-only rule

- `lib.ts`, `netdiff.ts`, `trajectory.ts`, `replay.ts` are **READ-ONLY**.
  They open the DB with `{ readonly: true }`, print to stdout, and never write
  to the database or the filesystem. Safe to run against production data.
- **WRITE scripts** are explicitly labeled: the filename is called out with a
  `*** WRITE SCRIPT ***` banner in the file header and in stdout. Currently
  the only one is `backup.ts` (filesystem copy, never opens the DB).
- Rule for future write scripts: filename must make the write obvious, the
  script must print a WRITE banner, and the README must say to run `backup.ts`
  first. Read-only scripts must never import from write scripts.

## Scripts

| script | mode | purpose |
|---|---|---|
| `netdiff.ts --file <relpath\|abspath> [--repo ID] [--db PATH]` | READ-ONLY | first anchor → last anchor unified diff + gross-vs-net stats |
| `replay.ts --file <relpath\|abspath> [--repo ID] [--db PATH]` | READ-ONLY | replays anchor + ordered deltas, asserts byte-equality with final anchor (exit 0/1) |
| `trajectory.ts --session <uuid-prefix> \| --live [--db PATH]` | READ-ONLY | session summary: event counts, files touched, time span |
| `examples.ts --file <relpath\|abspath> [--repo ID] [--db PATH] [--history N]` | READ-ONLY | emits JSONL next-edit candidate pairs (state → action), one per anchor window |
| `backup.ts [--db PATH] [--dest DIR]` | **WRITE** | timestamped copy of sqlite + wal + shm; run before any write op |

Defaults: `--db` is `$TABCOMPLETE_COLLECTOR_DB` or
`/mnt/ssd/collector-data/collector.sqlite`.

## Schema quirks these scripts know (so you don't have to)

- `events.file_id` is **NULL for everything the Neovim client sends** (the
  server nulls legacy string file ids). File attribution goes through
  `payload_json.path`, never the `file_id` column.
- Payload paths may be absolute (`/home/…/exercises/…`) or repo-relative;
  `lib.ts` collects all observed spellings per file.
- Blobs are `gzip` or `raw` per the `compression` column (mirrors
  `server/src/blobs.ts`); decode accordingly.
- `edit_delta` rows are zero-based, end-exclusive
  (`start_row/old_end_row/new_end_row`), same triple as `nvim_buf_attach`.

## Lessons encoded here

1. **Never truncate.** An early `LIMIT 10` on anchors silently dropped the true
   final state and produced a confident 4-hunk answer for a 5-region file.
   These scripts fetch ALL anchors/deltas and refuse hash prefixes (full
   64-hex sha256 only) for the same reason.
2. **Replay is the ground truth.** If `replay.ts` exits 0, the capture for
   that file is byte-exact; distrust any summary that disagrees with it.
3. **Known case: phantom zero-text deletions.** One stored event in
   `exercises/07_structs/structs1.rs` (seq 332: a `(10,11,10)` deletion of an
   empty line with empty inserted text) never happened in the buffer —
   changedtick is consecutive and bytecounts are consistent, yet replaying it
   shifts every later row and breaks byte-equality (187/188 deltas verify
   clean around it). A second file (`structs3.rs`) breaks the same way at a
   whitespace retype with no zero-text deletion anywhere nearby — the common
   thread is whitespace fiddling with single-segment consecutive ticks, prime
   suspects being formatter/autopairs edits interleaving with the callback.
   `replay.ts` names phantom suspects on failure. The Neovim plugin guards
   the whole class live (`buffers.lua`: expected-vs-live line-count check on
   every delta, shadow resync + skip on mismatch, covered by
   `empty-line-delete-emits`, `random-ops-shadow-matches-live`, and
   `divergent-shadow-resyncs` headless tests). Note the guard's limit, stated
   in code: post-change, the live buffer cannot corroborate deleted text, so
   count-neutral divergence is still possible — bounded by anchors,
   detectable here. Historical rows predate the guard and stay as-is.

## Typecheck

```bash
cd tools/trajectory_collector/analysis && bunx tsc --noEmit
```

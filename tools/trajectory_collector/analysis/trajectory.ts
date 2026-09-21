#!/usr/bin/env bun
/**
 * trajectory.ts — READ-ONLY. Session/file activity summary: event counts by
 * type, time span, anchors, files touched, gross-vs-net edit stats.
 * Prints to stdout only.
 *
 * Usage:
 *   bun analysis/trajectory.ts --session <uuid-prefix> [--db /path/collector.sqlite]
 *   bun analysis/trajectory.ts --live [--db /path/collector.sqlite]   # latest open session
 */
import { openReadOnly, parseArgs, DEFAULT_DB } from "./lib.ts";

const args = parseArgs(Bun.argv.slice(2), new Set(["db", "session", "live"]));
const dbPath = args.db ?? DEFAULT_DB;
const db = openReadOnly(dbPath);

let sessionId: string;
if ("live" in args) {
  const row = db.query(
    "SELECT session_id FROM sessions WHERE ended_at IS NULL ORDER BY started_at DESC LIMIT 1",
  ).get() as { session_id: string } | null;
  if (!row) {
    console.error("no open session");
    process.exit(2);
  }
  sessionId = row.session_id;
} else if (args.session) {
  const rows = db.query("SELECT session_id, started_at, ended_at, machine_id FROM sessions WHERE session_id LIKE ?").all(
    `${args.session}%`,
  ) as { session_id: string; started_at: string; ended_at: string | null; machine_id: string }[];
  if (rows.length !== 1) {
    console.error(`session prefix ${JSON.stringify(args.session)} matched ${rows.length} sessions`);
    process.exit(2);
  }
  sessionId = rows[0]!.session_id;
} else {
  console.error("pass --session <uuid-prefix> or --live");
  process.exit(2);
}

const meta = db.query(
  "SELECT session_id, machine_id, started_at, ended_at, cwd_name, git_branch_at_start FROM sessions WHERE session_id = ?",
).get(sessionId) as {
  session_id: string;
  machine_id: string;
  started_at: string;
  ended_at: string | null;
  cwd_name: string | null;
  git_branch_at_start: string | null;
};

console.log(`# trajectory (READ-ONLY, db=${dbPath})`);
console.log(`# session: ${meta.session_id}  machine: ${meta.machine_id}`);
console.log(`# started: ${meta.started_at}  ended: ${meta.ended_at ?? "(open)"}  cwd: ${meta.cwd_name ?? "?"}  branch: ${meta.git_branch_at_start ?? "?"}`);
console.log("# events by type:");
const counts = db.query(
  "SELECT event_type, COUNT(*) AS c FROM events WHERE session_id = ? GROUP BY 1 ORDER BY 2 DESC",
).all(sessionId) as { event_type: string; c: number }[];
for (const r of counts) console.log(`#   ${r.event_type}: ${r.c}`);

console.log("# files touched (by payload path):");
const paths = db.query(
  `SELECT json_extract(payload_json, '$.path') AS p, COUNT(*) AS c FROM events
   WHERE session_id = ? AND json_extract(payload_json, '$.path') IS NOT NULL
   GROUP BY 1 ORDER BY 2 DESC LIMIT 20`,
).all(sessionId) as { p: string; c: number }[];
for (const r of paths) console.log(`#   ${r.c}  ${r.p}`);

const span = db.query(
  "SELECT MIN(timestamp_ms) AS lo, MAX(timestamp_ms) AS hi FROM events WHERE session_id = ?",
).get(sessionId) as { lo: number; hi: number };
console.log(`# span: ${new Date(span.lo).toISOString()} -> ${new Date(span.hi).toISOString()} (${((span.hi - span.lo) / 1000).toFixed(1)}s)`);
db.close();

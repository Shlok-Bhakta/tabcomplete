#!/usr/bin/env bun
/**
 * replay.ts — READ-ONLY. Verifies the mandatory reconstruction property:
 * first anchor + ordered edit_deltas must byte-equal the last anchor.
 *
 * Also checks every delta's deleted_text against the running buffer
 * (validates capture correctness, not just the endpoint).
 * Exit 0 on full match, 1 otherwise. Prints to stdout only.
 *
 * Usage:
 *   bun analysis/replay.ts --file example.py --session <session-id> [--db /path/collector.sqlite]
 */
import { blobBytes, fileAnchors, fileDeltas, openReadOnly, parseArgs, resolveFile, DEFAULT_DB } from "./lib.ts";
import type { DeltaEvent } from "./lib.ts";

const args = parseArgs(Bun.argv.slice(2), new Set(["db", "file", "repo", "session"]));
const dbPath = args.db ?? DEFAULT_DB;
const db = openReadOnly(dbPath);

const { file, absPaths } = resolveFile(db, { file: args.file, repo: args.repo });
const anchors = fileAnchors(db, file.relative_path, absPaths).filter(
  (anchor) => !args.session || anchor.session_id === args.session,
);
if (anchors.length === 0) {
  console.error(`no anchors found for ${file.relative_path}`);
  process.exit(2);
}
const sessionDeltas = fileDeltas(db, file.relative_path, absPaths).filter(
  (delta) => !args.session || delta.session_id === args.session,
);
const first = anchors[0]!;
const last = anchors[anchors.length - 1]!;
const afterFirst = (delta: DeltaEvent) =>
  delta.session_id === first.session_id
    ? delta.sequence_number > first.sequence_number
    : delta.timestamp_ms > first.timestamp_ms;
const beforeLast = (delta: DeltaEvent) =>
  delta.session_id === last.session_id
    ? delta.sequence_number < last.sequence_number
    : delta.timestamp_ms < last.timestamp_ms;
const deltas = sessionDeltas.filter((delta) => afterFirst(delta) && beforeLast(delta));
const outside = sessionDeltas.filter((delta) => !afterFirst(delta) || !beforeLast(delta));
const unanchored = outside.length;

const norm = (b: Uint8Array): { lines: string[]; trailingNl: boolean } => {
  let s = new TextDecoder().decode(b);
  let trailingNl = false;
  if (s.endsWith("\n")) {
    trailingNl = true;
    s = s.slice(0, -1);
  }
  return { lines: s === "" ? [] : s.split("\n"), trailingNl };
};

const start = norm(blobBytes(db, anchors[0]!.content_hash));
let lines = [...start.lines];
let failures = 0;
const suspects: DeltaEvent[] = [];
for (const d of deltas) {
  const p = d.payload;
  if (p.deleted_text === "" && p.inserted_text === "" && p.old_end_row > p.new_end_row) {
    // Zero-text line deletion: the known phantom signature (see README
    // lesson 3). Harmless if real, fatal to replay if phantom — remember it
    // so a later failure can point here instead of at the cascade.
    suspects.push(d);
  }
  const actual = lines.slice(p.start_row, p.old_end_row).join("\n");
  // A pure insertion has empty deleted_text; a pure deletion has empty inserted_text.
  const expected = p.deleted_text;
  if (actual !== expected) {
    console.error(
      `delta mismatch @${p.start_row} ts=${d.timestamp_ms}: ` +
        `buffer has ${JSON.stringify(actual.slice(0, 80))}, event claims deleted ${JSON.stringify(expected.slice(0, 80))}`,
    );
    failures++;
    if (failures > 5) {
      console.error("... stopping after 5 mismatches");
      break;
    }
    continue;
  }
  const ins =
    p.inserted_text === ""
      ? new Array<string>(p.new_end_row - p.start_row).fill("")
      : p.inserted_text.split("\n");
  if (ins.length !== p.new_end_row - p.start_row) {
    console.error(
      `delta shape mismatch ts=${d.timestamp_ms} seq=${d.sequence_number}: ` +
        `inserted text splits to ${ins.length} lines but rows say ${p.new_end_row - p.start_row}`,
    );
    failures++;
    if (failures > 5) {
      console.error("... stopping after 5 mismatches");
      break;
    }
    continue;
  }
  lines.splice(p.start_row, p.old_end_row - p.start_row, ...ins);
}

const rebuilt = lines.join("\n") + (start.trailingNl ? "\n" : "");
const target = blobBytes(db, anchors[anchors.length - 1]!.content_hash);
const enc = new TextEncoder().encode(rebuilt);
let match = enc.length === target.length;
if (match) {
  for (let i = 0; i < enc.length; i++) {
    if (enc[i] !== target[i]) {
      match = false;
      break;
    }
  }
}

console.log(`# replay (READ-ONLY, db=${dbPath})`);
console.log(`# file: ${file.relative_path}`);
console.log(`# session: ${args.session ?? "all"}`);
console.log(`# anchors: ${anchors.length}  deltas replayed: ${deltas.length}  unanchored: ${unanchored}  delta mismatches: ${failures}`);
console.log(`# rebuilt bytes: ${enc.length}  target bytes: ${target.length}`);
for (const delta of outside.slice(0, 10)) {
  console.log(`# unanchored delta ts=${delta.timestamp_ms} seq=${delta.sequence_number}`);
}
if (match && failures === 0 && unanchored === 0) {
  console.log("REPLAY-OK: anchor + deltas byte-equal the final anchor");
} else {
  if (suspects.length > 0) {
    console.log(
      `suspect phantom deletions (zero-text line removals that may never have happened; ` +
        `see README lesson 3):`,
    );
    for (const s of suspects) {
      console.log(
        `  ts=${s.timestamp_ms} seq=${s.sequence_number} rows ${s.payload.start_row}->${s.payload.old_end_row}->${s.payload.new_end_row}`,
      );
    }
  }
  console.log("REPLAY-FAIL");
  process.exit(1);
}
db.close();

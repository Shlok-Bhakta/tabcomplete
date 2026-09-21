#!/usr/bin/env bun
/**
 * netdiff.ts — READ-ONLY. Prints the net change for a tracked file:
 * first anchor content -> last anchor content, as a unified diff,
 * plus gross-vs-net stats (delta count, bytes typed vs bytes net).
 *
 * No truncation: fetches ALL anchors. Full hashes only.
 * Never writes to the database or the filesystem (output goes to stdout).
 *
 * Usage:
 *   bun analysis/netdiff.ts --file exercises/07_structs/structs1.rs [--repo repo:rustlings] [--db /path/collector.sqlite]
 */
import { blobText, fileAnchors, fileDeltas, lineOps, openReadOnly, opsToHunks, parseArgs, resolveFile, splitLines, DEFAULT_DB } from "./lib.ts";

const args = parseArgs(Bun.argv.slice(2), new Set(["db", "file", "repo"]));
const dbPath = args.db ?? DEFAULT_DB;
const db = openReadOnly(dbPath);

const { file, absPaths } = resolveFile(db, { file: args.file, repo: args.repo });
const anchors = fileAnchors(db, file.relative_path, absPaths);
if (anchors.length === 0) {
  console.error(`no anchors found for ${file.relative_path}`);
  process.exit(2);
}
const deltas = fileDeltas(db, file.relative_path, absPaths);

const before = blobText(db, anchors[0]!.content_hash);
const after = blobText(db, anchors[anchors.length - 1]!.content_hash);

let grossInserted = 0;
let grossDeleted = 0;
for (const d of deltas) {
  grossInserted += d.payload.inserted_text.length;
  grossDeleted += d.payload.deleted_text.length;
}

console.log(`# netdiff (READ-ONLY, db=${dbPath})`);
console.log(`# file: ${file.relative_path}  repo: ${file.repo_id}`);
console.log(`# anchors: ${anchors.length} (${anchors[0]!.event_type} -> ${anchors[anchors.length - 1]!.event_type})`);
console.log(`# edit_deltas: ${deltas.length}  gross inserted chars: ${grossInserted}  gross deleted chars: ${grossDeleted}`);
const netDelta = after.length - before.length;
console.log(`# net bytes: ${before.length} -> ${after.length} (${netDelta >= 0 ? "+" : ""}${netDelta})`);
console.log(`--- a/${file.relative_path}\t(first anchor ${anchors[0]!.content_hash.slice(0, 12)})`);
console.log(`+++ b/${file.relative_path}\t(last anchor ${anchors[anchors.length - 1]!.content_hash.slice(0, 12)})`);

const hunks = opsToHunks(lineOps(splitLines(before), splitLines(after)));
for (const h of hunks) {
  console.log(`@@ -${h.aStart},${h.aCount} +${h.bStart},${h.bCount} @@`);
  for (const o of h.body) console.log(`${o.t}${o.line}`);
}
console.log(`# hunks: ${hunks.length}`);
db.close();

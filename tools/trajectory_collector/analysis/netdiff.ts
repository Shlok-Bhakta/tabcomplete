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
import { blobText, fileAnchors, fileDeltas, openReadOnly, parseArgs, resolveFile, DEFAULT_DB } from "./lib.ts";

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

function splitLines(s: string): string[] {
  if (s === "") return [];
  const parts = s.split("\n");
  if (parts.length > 0 && parts[parts.length - 1] === "") parts.pop();
  return parts;
}

type Op = { t: " " | "-" | "+"; line: string; aNum: number; bNum: number };

/** Exact LCS op list with 1-based line numbers on both sides. */
function ops(a: string[], b: string[]): Op[] {
  const n = a.length;
  const m = b.length;
  const dp: Uint32Array[] = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i]![j] = a[i] === b[j] ? dp[i + 1]![j + 1]! + 1 : Math.max(dp[i + 1]![j]!, dp[i]![j + 1]!);
    }
  }
  const out: Op[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      out.push({ t: " ", line: a[i]!, aNum: i + 1, bNum: j + 1 });
      i++;
      j++;
    } else if (dp[i + 1]![j]! >= dp[i]![j + 1]!) {
      out.push({ t: "-", line: a[i]!, aNum: i + 1, bNum: -1 });
      i++;
    } else {
      out.push({ t: "+", line: b[j]!, aNum: -1, bNum: j + 1 });
      j++;
    }
  }
  while (i < n) out.push({ t: "-", line: a[i]!, aNum: i++ + 1, bNum: -1 });
  while (j < m) out.push({ t: "+", line: b[j]!, aNum: -1, bNum: j++ + 1 });
  return out;
}

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

const CTX = 3;
const all = ops(splitLines(before), splitLines(after));
// Indices of change ops; group runs separated by <= 2*CTX context lines.
const changeIdx: number[] = [];
for (let k = 0; k < all.length; k++) if (all[k]!.t !== " ") changeIdx.push(k);
let hunkCount = 0;
let g = 0;
while (g < changeIdx.length) {
  let h = g;
  while (h + 1 < changeIdx.length && changeIdx[h + 1]! - changeIdx[h]! <= 2 * CTX + 1) h++;
  const lo = Math.max(0, changeIdx[g]! - CTX);
  const hi = Math.min(all.length, changeIdx[h]! + CTX + 1);
  const body = all.slice(lo, hi);
  const aCount = body.filter((o) => o.t !== "+").length;
  const bCount = body.filter((o) => o.t !== "-").length;
  const firstA = body.find((o) => o.t !== "+")!.aNum;
  const firstB = body.find((o) => o.t !== "-")!.bNum;
  console.log(`@@ -${firstA},${aCount} +${firstB},${bCount} @@`);
  for (const o of body) console.log(`${o.t}${o.line}`);
  hunkCount++;
  g = h + 1;
}
console.log(`# hunks: ${hunkCount}`);
db.close();

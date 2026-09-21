#!/usr/bin/env bun
/**
 * examples.ts — READ-ONLY. Emits candidate next-edit training pairs as JSONL,
 * one record per consecutive-anchor window for a file. This is the coded
 * version of "collapse a noisy keystroke trajectory to (state -> action)":
 * no intuition, same output every run. Prints to stdout only.
 *
 * Each record:
 *   state  = file content at window start + cursor + bounded recent history
 *   action = net change to window end as structured hunks
 *   stats  = gross-vs-net (how much typing the action distills)
 *
 * Policy choices are explicit and flagged POLICY so future training work can
 * change them deliberately: window boundaries (anchors), history bound (20
 * pre-window deltas, raw not coalesced), full-file state (not a cursor
 * window). Nothing here trains a model; it only materialises pairs.
 *
 * Usage:
 *   bun analysis/examples.ts --file exercises/07_structs/structs1.rs [--repo ID] [--db PATH] [--history 20]
 */
import {
  blobText,
  fileAnchors,
  fileDeltas,
  lineOps,
  openReadOnly,
  opsToHunks,
  parseArgs,
  resolveFile,
  splitLines,
  DEFAULT_DB,
} from "./lib.ts";

const args = parseArgs(Bun.argv.slice(2), new Set(["db", "file", "repo", "history"]));
const dbPath = args.db ?? DEFAULT_DB;
const historyBound = args.history !== undefined ? parseInt(args.history, 10) : 20; // POLICY
if (!Number.isInteger(historyBound) || historyBound < 0) throw new Error("--history must be a non-negative integer");
const db = openReadOnly(dbPath);

const { file, absPaths } = resolveFile(db, { file: args.file, repo: args.repo });
const anchors = fileAnchors(db, file.relative_path, absPaths);
if (anchors.length < 2) {
  console.error(`need >= 2 anchors for ${file.relative_path}, have ${anchors.length}`);
  process.exit(2);
}
const deltas = fileDeltas(db, file.relative_path, absPaths);

// POLICY: windows are consecutive anchors (buffer_open/buffer_write).
// Rationale: anchors are the moments a human considered the file worth
// snapshotting (open, save) — the natural unit of "one editing episode".
for (let w = 0; w + 1 < anchors.length; w++) {
  const from = anchors[w]!;
  const to = anchors[w + 1]!;
  const inWindow = deltas.filter((d) => d.timestamp_ms > from.timestamp_ms && d.timestamp_ms <= to.timestamp_ms);
  const beforeWindow = deltas.filter((d) => d.timestamp_ms <= from.timestamp_ms).slice(-historyBound);

  const stateContent = blobText(db, from.content_hash);
  const endContent = blobText(db, to.content_hash);
  const hunks = opsToHunks(lineOps(splitLines(stateContent), splitLines(endContent)));
  if (hunks.length === 0) continue; // nothing changed: no training signal

  // POLICY: state cursor = last known cursor at or before window start.
  const cursorSrc = [...beforeWindow].reverse().find((d) => d.payload.cursor_after);
  const cursor = cursorSrc ? cursorSrc.payload.cursor_after : null;

  let grossIns = 0;
  let grossDel = 0;
  for (const d of inWindow) {
    grossIns += d.payload.inserted_text.length;
    grossDel += d.payload.deleted_text.length;
  }

  const record = {
    protocol: "trajectory-example/v1",
    file: file.relative_path,
    repo_id: file.repo_id,
    window: {
      from: { ts: from.timestamp_ms, anchor: from.event_type, sha256: from.content_hash },
      to: { ts: to.timestamp_ms, anchor: to.event_type, sha256: to.content_hash },
      span_s: +((to.timestamp_ms - from.timestamp_ms) / 1000).toFixed(1),
    },
    state: { content: stateContent, cursor },
    // POLICY: recent history is raw (bounded), not coalesced — coalescing is
    // a training-time decision; the evidence stays intact here.
    recent_history: beforeWindow.map((d) => ({
      ts: d.timestamp_ms,
      start_row: d.payload.start_row,
      old_end_row: d.payload.old_end_row,
      new_end_row: d.payload.new_end_row,
      deleted_text: d.payload.deleted_text,
      inserted_text: d.payload.inserted_text,
      cursor_after: d.payload.cursor_after ?? null,
    })),
    action: {
      hunks: hunks.map((h) => ({
        aStart: h.aStart,
        aCount: h.aCount,
        bStart: h.bStart,
        bCount: h.bCount,
        lines: h.body.map((o) => ({ t: o.t, line: o.line })),
      })),
    },
    stats: {
      gross_deltas: inWindow.length,
      gross_inserted_chars: grossIns,
      gross_deleted_chars: grossDel,
      net_bytes_from: stateContent.length,
      net_bytes_to: endContent.length,
    },
  };
  console.log(JSON.stringify(record));
}
db.close();

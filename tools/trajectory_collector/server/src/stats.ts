import { statSync } from "node:fs";
import type { Database } from "bun:sqlite";
import { resolveDbPath } from "./db";

export interface CollectorStats {
  protocol_version: 1;
  db_path: string;
  db_bytes: number;
  wal_bytes: number;
  blob_count: number;
  blob_original_bytes: number;
  blob_stored_bytes: number;
  compression_ratio: number | null;
  event_count: number;
  session_count: number;
  repository_count: number;
  file_count: number;
  file_version_count: number;
}

function fileBytes(path: string): number {
  try {
    return statSync(path).size;
  } catch {
    return 0;
  }
}

function count(db: Database, table: string): number {
  const row = db.query<{ n: number }, []>(`SELECT COUNT(*) AS n FROM ${table}`).get();
  return row?.n ?? 0;
}

export function collectStats(db: Database, dbPath?: string): CollectorStats {
  const path = dbPath ?? resolveDbPath();
  const dbBytes = fileBytes(path);
  const walBytes = fileBytes(`${path}-wal`);
  const blobCount = count(db, "blobs");
  const blobAgg = db
    .query<{ original: number | null; stored: number | null }, []>(
      "SELECT COALESCE(SUM(original_bytes),0) AS original, COALESCE(SUM(stored_bytes),0) AS stored FROM blobs",
    )
    .get();
  const original = blobAgg?.original ?? 0;
  const stored = blobAgg?.stored ?? 0;
  return {
    protocol_version: 1,
    db_path: path,
    db_bytes: dbBytes,
    wal_bytes: walBytes,
    blob_count: blobCount,
    blob_original_bytes: original,
    blob_stored_bytes: stored,
    compression_ratio: original > 0 ? stored / original : null,
    event_count: count(db, "events"),
    session_count: count(db, "sessions"),
    repository_count: count(db, "repositories"),
    file_count: count(db, "files"),
    file_version_count: count(db, "file_versions"),
  };
}

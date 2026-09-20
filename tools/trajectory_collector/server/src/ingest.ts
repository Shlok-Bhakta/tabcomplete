import type { Database } from "bun:sqlite";
import { sanitizeOriginUrl, type FileVersionSource, type RepositorySnapshotRequest } from "./types";

function nowIso(): string {
  return new Date().toISOString();
}

function nullableString(v: unknown): string | null {
  if (v === undefined || v === null) return null;
  if (typeof v !== "string") return null;
  return v;
}

export interface SnapshotResult {
  repo_id: string;
  files_upserted: number;
  versions_inserted: number;
  versions_skipped_duplicate: number;
}

const VALID_SOURCES = new Set<string>([
  "repo_scan",
  "buffer_open",
  "buffer_write",
  "session_anchor",
  "periodic_anchor",
]);

export function ingestRepositorySnapshot(
  db: Database,
  body: RepositorySnapshotRequest,
  observedAtDefault?: string,
): SnapshotResult {
  const now = observedAtDefault ?? nowIso();
  const repoId = body.repo_id;
  const rootName = nullableString(body.root_name);
  const rootIdentityHash = nullableString(body.root_identity_hash);
  const originUrl = sanitizeOriginUrl(body.origin_url);

  const existing = db
    .query<{ repo_id: string }, [string]>("SELECT repo_id FROM repositories WHERE repo_id = ?")
    .get(repoId);

  if (existing) {
    db.prepare(
      `UPDATE repositories SET
         root_name = COALESCE(?, root_name),
         root_identity_hash = COALESCE(?, root_identity_hash),
         origin_url = COALESCE(?, origin_url),
         last_seen_at = ?
       WHERE repo_id = ?`,
    ).run(rootName, rootIdentityHash, originUrl, now, repoId);
  } else {
    db.prepare(
      `INSERT INTO repositories (repo_id, root_name, root_identity_hash, origin_url, first_seen_at, last_seen_at)
       VALUES (?, ?, ?, ?, ?, ?)`,
    ).run(repoId, rootName, rootIdentityHash, originUrl, now, now);
  }

  const selectFile = db.prepare(
    "SELECT file_id FROM files WHERE repo_id = ? AND relative_path = ?",
  );
  const insertFile = db.prepare(
    `INSERT INTO files (repo_id, relative_path, extension, language, first_seen_at, last_seen_at)
     VALUES (?, ?, ?, ?, ?, ?)`,
  );
  const updateFile = db.prepare(
    `UPDATE files SET extension = COALESCE(?, extension),
                      language = COALESCE(?, language),
                      last_seen_at = ?
     WHERE file_id = ?`,
  );
  const insertVersion = db.prepare(
    `INSERT OR IGNORE INTO file_versions (file_id, sha256, observed_at, source)
     VALUES (?, ?, ?, ?)`,
  );

  let filesUpserted = 0;
  let versionsInserted = 0;
  let versionsSkipped = 0;

  const defaultSource: FileVersionSource = isValidDefaultSource(body.default_source)
    ? body.default_source
    : "repo_scan";

  const txn = db.transaction((files: RepositorySnapshotRequest["files"]) => {
    for (const f of files) {
      const rel = f.relative_path;
      const ext = nullableString(f.extension);
      const lang = nullableString(f.language);
      const sha = f.sha256.toLowerCase();
      const observedAt =
        typeof f.observed_at === "string" && f.observed_at.length > 0 ? f.observed_at : now;
      const source: string =
        typeof f.source === "string" && VALID_SOURCES.has(f.source) ? f.source : defaultSource;

      const row = selectFile.get(repoId, rel) as { file_id: number } | null;
      let fileId: number;
      if (row) {
        fileId = row.file_id;
        updateFile.run(ext, lang, now, fileId);
      } else {
        const info = insertFile.run(repoId, rel, ext, lang, now, now);
        fileId = Number(info.lastInsertRowid);
      }
      filesUpserted += 1;
      const res = insertVersion.run(fileId, sha, observedAt, source);
      if (Number(res.changes) === 0) {
        versionsSkipped += 1;
      } else {
        versionsInserted += 1;
      }
    }
  });

  txn(body.files);

  return {
    repo_id: repoId,
    files_upserted: filesUpserted,
    versions_inserted: versionsInserted,
    versions_skipped_duplicate: versionsSkipped,
  };
}

function isValidDefaultSource(v: unknown): v is FileVersionSource {
  return typeof v === "string" && VALID_SOURCES.has(v);
}

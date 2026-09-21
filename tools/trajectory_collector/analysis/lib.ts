/**
 * Shared helpers for read-only trajectory analysis scripts.
 *
 * RULE: everything in here opens the database READ-ONLY. Write scripts
 * (files starting with `write-`, e.g. backup.ts) must NOT import from this
 * module — they go through their own path that prints a WRITE banner first.
 * See analysis/README.md.
 */
import { Database } from "bun:sqlite";
import { gunzipSync } from "node:zlib";

export const DEFAULT_DB =
  process.env.TABCOMPLETE_COLLECTOR_DB ??
  "/mnt/ssd/collector-data/collector.sqlite";

/** Open the collector DB read-only. Throws if the file is missing. */
export function openReadOnly(dbPath: string = DEFAULT_DB): Database {
  return new Database(dbPath, { readonly: true });
}

/** Minimal --key value / --key=value parser. Unknown flags throw. */
export function parseArgs(
  raw: string[],
  allowed: Set<string>,
): Record<string, string> {
  const out: Record<string, string> = {};
  for (let i = 0; i < raw.length; i++) {
    const tok = raw[i]!;
    if (!tok.startsWith("--")) throw new Error(`positional args unsupported: ${tok}`);
    const eq = tok.indexOf("=");
    let key: string;
    let value: string;
    if (eq >= 0) {
      key = tok.slice(2, eq);
      value = tok.slice(eq + 1);
    } else {
      key = tok.slice(2);
      value = raw[++i] ?? "";
    }
    if (!allowed.has(key)) throw new Error(`unknown flag --${key}`);
    out[key] = value;
  }
  return out;
}

export interface BlobRow {
  content: Uint8Array;
  compression: string;
}

/** Decode a blob to bytes. Throws on missing hash — prefixes are refused. */
export function blobBytes(db: Database, sha256: string): Uint8Array {
  if (!/^[0-9a-f]{64}$/.test(sha256)) {
    throw new Error(
      `refusing hash prefix ${JSON.stringify(sha256)}: pass the full 64-hex sha256 ` +
        `(prefixes silently match the wrong row — this bit us before)`,
    );
  }
  const row = db
    .query("SELECT content, compression FROM blobs WHERE sha256 = ?")
    .get(sha256) as BlobRow | null;
  if (!row) throw new Error(`blob not found: ${sha256}`);
  const bytes = row.content;
  if (row.compression === "gzip") return gunzipSync(bytes);
  return bytes;
}

export function blobText(db: Database, sha256: string): string {
  return new TextDecoder().decode(blobBytes(db, sha256));
}

export interface FileRow {
  file_id: number;
  repo_id: string;
  relative_path: string;
}

/**
 * Resolve a file by repo-relative path (preferred) or absolute client path.
 * Returns the files row plus every absolute path ever seen for it in payloads.
 */
export function resolveFile(
  db: Database,
  opts: { file?: string; repo?: string },
): { file: FileRow; absPaths: string[] } {
  if (!opts.file) throw new Error("missing required --file <repo-relative-path | absolute-path>");
  let file: FileRow | null = null;
  if (opts.file.startsWith("/")) {
    // Absolute path: find via payload paths -> relative path -> files row.
    const like = db.query(
      `SELECT DISTINCT f.file_id, f.repo_id, f.relative_path FROM files f
       WHERE f.relative_path IN (
         SELECT DISTINCT json_extract(e.payload_json, '$.path') FROM events e
         WHERE json_extract(e.payload_json, '$.path') = ?
       )`,
    );
    const rows = like.all(opts.file) as FileRow[];
    if (rows.length === 0) {
      // Fall back: match by suffix against relative paths.
      const suffix = db.query(
        "SELECT file_id, repo_id, relative_path FROM files WHERE relative_path LIKE ?",
      );
      const cands = suffix.all(`%${opts.file.split("/").slice(-2).join("/")}`) as FileRow[];
      if (cands.length !== 1) {
        throw new Error(
          `absolute path ${opts.file} matched ${cands.length} files; use --file <repo-relative-path>`,
        );
      }
      file = cands[0]!;
    } else {
      file = rows[0]!;
    }
  } else {
    const q = opts.repo
      ? db.query(
          "SELECT file_id, repo_id, relative_path FROM files WHERE relative_path = ? AND repo_id = ?",
        )
      : db.query("SELECT file_id, repo_id, relative_path FROM files WHERE relative_path = ?");
    const rows = (opts.repo
      ? q.all(opts.file, opts.repo)
      : q.all(opts.file)) as FileRow[];
    if (rows.length === 0) throw new Error(`no such file: ${opts.file}`);
    if (rows.length > 1 && !opts.repo) {
      throw new Error(
        `ambiguous path ${opts.file} across repos (${rows.map((r) => r.repo_id).join(", ")}); pass --repo`,
      );
    }
    file = rows[0]!;
  }
  // NOTE: events.file_id is NULL for everything the Neovim client sends
  // (the server nulls legacy string file ids), so file attribution MUST go
  // through payload_json path, never the file_id column. Both relative and
  // absolute payload paths occur; collect them all.
  const absRows = db.query(
    `SELECT DISTINCT json_extract(e.payload_json, '$.path') AS p FROM events e
     WHERE json_extract(e.payload_json, '$.path') IS NOT NULL
       AND (json_extract(e.payload_json, '$.path') = ?
         OR json_extract(e.payload_json, '$.path') LIKE ?)`,
  ).all(file.relative_path, `%/` + file.relative_path) as { p: string }[];
  return { file, absPaths: absRows.map((r) => r.p) };
}

export interface AnchorEvent {
  timestamp_ms: number;
  event_type: string;
  content_hash: string;
  session_id: string;
}

/** ALL anchors for a file, oldest first. Never truncated — LIMIT hid real data once. */
export function fileAnchors(
  db: Database,
  relPath: string,
  absPaths: string[],
): AnchorEvent[] {
  const paths = [relPath, ...absPaths];
  const placeholders = paths.map(() => "?").join(",");
  return db.query(
    `SELECT timestamp_ms, event_type, json_extract(payload_json, '$.content_hash') AS content_hash,
            session_id FROM events
     WHERE event_type IN ('buffer_open','buffer_write')
       AND json_extract(payload_json, '$.path') IN (${placeholders})
       AND json_extract(payload_json, '$.content_hash') IS NOT NULL
     ORDER BY timestamp_ms ASC`,
  ).all(...paths) as AnchorEvent[];
}

export interface DeltaEvent {
  timestamp_ms: number;
  sequence_number: number;
  session_id: string;
  payload: {
    start_row: number;
    old_end_row: number;
    new_end_row: number;
    deleted_text: string;
    inserted_text: string;
    path?: string;
  };
}

/** ALL edit deltas touching any of the file's paths, in capture order. */
export function fileDeltas(
  db: Database,
  relPath: string,
  absPaths: string[],
): DeltaEvent[] {
  const paths = [relPath, ...absPaths];
  const placeholders = paths.map(() => "?").join(",");
  const rows = db.query(
    `SELECT timestamp_ms, sequence_number, session_id, payload_json FROM events
     WHERE event_type = 'edit_delta'
       AND json_extract(payload_json, '$.path') IN (${placeholders})
     ORDER BY timestamp_ms ASC, sequence_number ASC`,
  ).all(...paths) as {
    timestamp_ms: number;
    sequence_number: number;
    session_id: string;
    payload_json: string;
  }[];
  return rows.map((r) => ({
    timestamp_ms: r.timestamp_ms,
    sequence_number: r.sequence_number,
    session_id: r.session_id,
    payload: JSON.parse(r.payload_json),
  }));
}

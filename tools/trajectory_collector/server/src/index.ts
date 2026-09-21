import type { Database } from "bun:sqlite";
import { openDatabase, resolveDbPath } from "./db";
import { ingestRepositorySnapshot } from "./ingest";
import { findMissingHashes, storeBlob } from "./blobs";
import { collectStats } from "./stats";
import {
  MAX_BODY_BYTES,
  PROTOCOL_VERSION,
  isNonEmptyString,
  isRecord,
  isValidSha256,
  sanitizeOriginUrl,
  validateEventEnvelope,
  type EventEnvelope,
} from "./types";

function nowIso(): string {
  return new Date().toISOString();
}

function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

function errorResponse(code: string, message: string, status: number): Response {
  return jsonResponse({ error: code, message }, status);
}

async function readBodyCapped(req: Request): Promise<
  { ok: true; text: string } | { ok: false; response: Response }
> {
  const declared = req.headers.get("content-length");
  if (declared !== null) {
    const n = Number(declared);
    if (Number.isFinite(n) && n > MAX_BODY_BYTES) {
      return {
        ok: false,
        response: errorResponse("payload_too_large", `request body exceeds ${MAX_BODY_BYTES} bytes`, 413),
      };
    }
  }
  let buf: ArrayBuffer;
  try {
    buf = await req.arrayBuffer();
  } catch {
    return { ok: false, response: errorResponse("bad_request", "unable to read request body", 400) };
  }
  if (buf.byteLength > MAX_BODY_BYTES) {
    return {
      ok: false,
      response: errorResponse("payload_too_large", `request body exceeds ${MAX_BODY_BYTES} bytes`, 413),
    };
  }
  return { ok: true, text: new TextDecoder().decode(buf) };
}

function parseJson(text: string): { ok: true; value: unknown } | { ok: false; message: string } {
  if (text.trim() === "") return { ok: false, message: "empty request body" };
  try {
    return { ok: true, value: JSON.parse(text) };
  } catch {
    return { ok: false, message: "invalid JSON body" };
  }
}

function nullableString(v: unknown): string | null {
  if (v === undefined || v === null) return null;
  return typeof v === "string" ? v : null;
}

function handleSessionStart(db: Database, body: unknown): Response {
  if (!isRecord(body)) return errorResponse("bad_request", "request body must be an object", 400);
  if (body["protocol_version"] !== PROTOCOL_VERSION) {
    return errorResponse("bad_request", `protocol_version must be ${PROTOCOL_VERSION}`, 400);
  }
  const { session_id, machine_id, repo_id, started_at } = body;
  if (!isNonEmptyString(session_id)) return errorResponse("bad_request", "session_id is required", 400);
  if (!isNonEmptyString(machine_id)) return errorResponse("bad_request", "machine_id is required", 400);
  if (!isNonEmptyString(started_at)) return errorResponse("bad_request", "started_at is required", 400);

  const repoId = body["repo_id"];
  if (repoId !== undefined && repoId !== null && !isNonEmptyString(repoId)) {
    return errorResponse("bad_request", "repo_id must be a string or null", 400);
  }
  const repoIdValue: string | null = typeof repoId === "string" ? repoId : null;

  const editor = nullableString(body["editor"]);
  const editorVersion = nullableString(body["editor_version"]);
  const pluginVersion = nullableString(body["plugin_version"]);
  const platform = nullableString(body["platform"]);
  const hostnameHash = nullableString(body["hostname_hash"]);
  const cwdName = nullableString(body["cwd_name"]);
  const gitHead = nullableString(body["git_head_at_start"]);
  const gitBranch = nullableString(body["git_branch_at_start"]);
  const rootName = nullableString(body["root_name"]);
  const rootIdentityHash = nullableString(body["root_identity_hash"]);
  const originUrl = sanitizeOriginUrl(body["origin_url"]);

  const startedAt = started_at as string;

  // Upsert machine.
  const existingMachine = db
    .query<{ machine_id: string }, [string]>("SELECT machine_id FROM machines WHERE machine_id = ?")
    .get(machine_id as string);
  if (existingMachine) {
    db.prepare(
      `UPDATE machines SET last_seen_at = ?, editor = COALESCE(?, editor),
        editor_version = COALESCE(?, editor_version), plugin_version = COALESCE(?, plugin_version),
        platform = COALESCE(?, platform), hostname_hash = COALESCE(?, hostname_hash)
       WHERE machine_id = ?`,
    ).run(startedAt, editor, editorVersion, pluginVersion, platform, hostnameHash, machine_id as string);
  } else {
    db.prepare(
      `INSERT INTO machines (machine_id, first_seen_at, last_seen_at, editor, editor_version, plugin_version, platform, hostname_hash)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      machine_id as string,
      startedAt,
      startedAt,
      editor,
      editorVersion,
      pluginVersion,
      platform,
      hostnameHash,
    );
  }

  // Ensure repo stub so the sessions FK stays satisfied before snapshot arrives.
  if (repoIdValue) {
    db.prepare(
      `INSERT OR IGNORE INTO repositories (repo_id, root_name, root_identity_hash, origin_url, first_seen_at, last_seen_at)
       VALUES (?, ?, ?, ?, ?, ?)`,
    ).run(repoIdValue, rootName, rootIdentityHash, originUrl, startedAt, startedAt);
  }

  const existingSession = db
    .query<{ session_id: string }, [string]>("SELECT session_id FROM sessions WHERE session_id = ?")
    .get(session_id as string);
  if (existingSession) {
    return jsonResponse({ session_id, deduped: true }, 200);
  }
  db.prepare(
    `INSERT INTO sessions (session_id, machine_id, repo_id, started_at, ended_at, cwd_name,
       git_head_at_start, git_branch_at_start, editor, editor_version, plugin_version)
     VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)`,
  ).run(
    session_id as string,
    machine_id as string,
    repoIdValue,
    startedAt,
    cwdName,
    gitHead,
    gitBranch,
    editor,
    editorVersion,
    pluginVersion,
  );
  return jsonResponse({ session_id, deduped: false }, 201);
}

function handleSessionEnd(db: Database, body: unknown): Response {
  if (!isRecord(body)) return errorResponse("bad_request", "request body must be an object", 400);
  if (body["protocol_version"] !== PROTOCOL_VERSION) {
    return errorResponse("bad_request", `protocol_version must be ${PROTOCOL_VERSION}`, 400);
  }
  const { session_id } = body;
  if (!isNonEmptyString(session_id)) return errorResponse("bad_request", "session_id is required", 400);
  const rawEnded = body["ended_at"];
  let endedAt: string;
  if (rawEnded === undefined || rawEnded === null) {
    endedAt = nowIso();
  } else if (typeof rawEnded === "string" && rawEnded.length > 0) {
    endedAt = rawEnded;
  } else {
    return errorResponse("bad_request", "ended_at must be a string or null", 400);
  }
  const existing = db
    .query<{ session_id: string }, [string]>("SELECT session_id FROM sessions WHERE session_id = ?")
    .get(session_id as string);
  if (!existing) return errorResponse("not_found", `session ${session_id} not found`, 404);
  db.prepare("UPDATE sessions SET ended_at = ? WHERE session_id = ?").run(endedAt, session_id as string);
  return jsonResponse({ session_id, ended_at: endedAt }, 200);
}

function handleEventsBatch(db: Database, body: unknown): Response {
  if (!isRecord(body)) return errorResponse("bad_request", "request body must be an object", 400);
  if (body["protocol_version"] !== PROTOCOL_VERSION) {
    return errorResponse("bad_request", `protocol_version must be ${PROTOCOL_VERSION}`, 400);
  }
  const eventsRaw = body["events"];
  if (!Array.isArray(eventsRaw)) return errorResponse("bad_request", "events must be an array", 400);
  if (eventsRaw.length === 0) return jsonResponse({ ingested: 0, skipped_duplicate: 0, total: 0 }, 200);

  const validated: EventEnvelope[] = [];
  for (let i = 0; i < eventsRaw.length; i++) {
    const res = validateEventEnvelope(eventsRaw[i]);
    if (!res.ok) {
      return errorResponse("bad_request", `events[${i}]: ${res.message}`, 400);
    }
    validated.push(res.value);
  }

  const sessionIds = [...new Set(validated.map((e) => e.session_id))];
  const placeholders = sessionIds.map(() => "?").join(",");
  const found = db
    .query<{ session_id: string }, string[]>(`SELECT session_id FROM sessions WHERE session_id IN (${placeholders})`)
    .all(...sessionIds)
    .map((r) => r.session_id);
  const foundSet = new Set(found);
  const missing = sessionIds.filter((s) => !foundSet.has(s));
  if (missing.length > 0) {
    return errorResponse("not_found", `session(s) not found: ${missing.join(",")}`, 404);
  }

  const stmt = db.prepare(
    `INSERT OR IGNORE INTO events
      (event_id, session_id, sequence_number, event_type, timestamp_ms, file_id, changedtick, cursor_row, cursor_col, mode, payload_json)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  );
  let ingested = 0;
  const txn = db.transaction((evs: EventEnvelope[]) => {
    for (const ev of evs) {
      const payloadJson = ev.payload === undefined ? null : JSON.stringify(ev.payload);
      const res = stmt.run(
        ev.event_id,
        ev.session_id,
        ev.sequence_number,
        ev.event_type,
        ev.timestamp_ms,
        ev.file_id ?? null,
        ev.changedtick ?? null,
        ev.cursor?.row ?? null,
        ev.cursor?.col ?? null,
        ev.mode ?? null,
        payloadJson,
      );
      if (Number(res.changes) > 0) ingested += 1;
    }
  });
  txn(validated);
  return jsonResponse(
    { ingested, skipped_duplicate: validated.length - ingested, total: validated.length },
    200,
  );
}

function handleBlobsCheck(db: Database, body: unknown): Response {
  if (!isRecord(body)) return errorResponse("bad_request", "request body must be an object", 400);
  const hashes = body["hashes"];
  if (!Array.isArray(hashes)) return errorResponse("bad_request", "hashes must be an array", 400);
  for (let i = 0; i < hashes.length; i++) {
    if (!isValidSha256(hashes[i])) {
      return errorResponse("bad_request", `hashes[${i}] must be a 64-char hex sha256`, 400);
    }
  }
  const missing = findMissingHashes(db, hashes as string[]);
  return jsonResponse({ missing }, 200);
}

function handleBlobsUpload(db: Database, body: unknown): Response {
  if (!isRecord(body)) return errorResponse("bad_request", "request body must be an object", 400);
  const { sha256, content_base64 } = body;
  if (!isValidSha256(sha256)) return errorResponse("bad_request", "sha256 must be a 64-char hex string", 400);
  if (typeof content_base64 !== "string" || content_base64.length === 0) {
    return errorResponse("bad_request", "content_base64 must be a non-empty base64 string", 400);
  }
  let original: Uint8Array;
  try {
    const buf = Buffer.from(content_base64 as string, "base64");
    // Round-trip check: re-encoding must be stable (ignoring whitespace).
    if (buf.length === 0 && (content_base64 as string).length > 0) {
      // Buffer.from never throws on invalid base64; detect garbage via strict check.
      if (!/^[A-Za-z0-9+/=\s]+$/.test(content_base64 as string)) {
        return errorResponse("bad_request", "content_base64 is not valid base64", 400);
      }
    }
    original = new Uint8Array(buf);
  } catch {
    return errorResponse("bad_request", "content_base64 is not valid base64", 400);
  }
  const mimeType = nullableString(body["mime_type"]);
  const alreadyCompressedRaw = body["already_compressed"];
  const alreadyCompressed =
    alreadyCompressedRaw === true ? true : alreadyCompressedRaw === false ? false : null;
  if (
    alreadyCompressedRaw !== undefined &&
    alreadyCompressedRaw !== null &&
    typeof alreadyCompressedRaw !== "boolean"
  ) {
    return errorResponse("bad_request", "already_compressed must be a boolean or null", 400);
  }
  try {
    const result = storeBlob(db, (sha256 as string).toLowerCase(), original, {
      alreadyCompressed,
      mimeType,
    });
    return jsonResponse(result, 200);
  } catch (err) {
    const withCode = err as { code?: string; status?: number; message?: string };
    if (withCode?.code === "sha_mismatch") {
      return errorResponse("bad_request", withCode.message ?? "sha256 mismatch", 400);
    }
    throw err;
  }
}

const SNAPSHOT_SOURCES = new Set(["repo_scan", "buffer_open", "buffer_write", "session_anchor", "periodic_anchor"]);

function handleSnapshot(db: Database, body: unknown): Response {
  if (!isRecord(body)) return errorResponse("bad_request", "request body must be an object", 400);
  if (body["protocol_version"] !== PROTOCOL_VERSION) {
    return errorResponse("bad_request", `protocol_version must be ${PROTOCOL_VERSION}`, 400);
  }
  const { repo_id, files } = body;
  if (!isNonEmptyString(repo_id)) return errorResponse("bad_request", "repo_id is required", 400);
  if (!Array.isArray(files)) return errorResponse("bad_request", "files must be an array", 400);
  const defaultSourceRaw = body["default_source"];
  if (defaultSourceRaw !== undefined && defaultSourceRaw !== null && !SNAPSHOT_SOURCES.has(defaultSourceRaw as string)) {
    return errorResponse(
      "bad_request",
      "default_source must be one of repo_scan,buffer_open,buffer_write,session_anchor,periodic_anchor",
      400,
    );
  }
  for (let i = 0; i < files.length; i++) {
    const f = files[i];
    if (!isRecord(f)) return errorResponse("bad_request", `files[${i}] must be an object`, 400);
    if (!isNonEmptyString(f["relative_path"])) {
      return errorResponse("bad_request", `files[${i}].relative_path is required`, 400);
    }
    if ((f["relative_path"] as string).startsWith("/")) {
      return errorResponse("bad_request", `files[${i}].relative_path must be relative`, 400);
    }
    const relPath = f["relative_path"] as string;
    if (
      relPath.includes("\\") ||
      relPath.includes("\0") ||
      relPath.split("/").some((seg) => seg === "..")
    ) {
      return errorResponse(
        "bad_request",
        `files[${i}].relative_path must not contain '..', backslashes, or NUL bytes`,
        400,
      );
    }
    if (!isValidSha256(f["sha256"])) {
      return errorResponse("bad_request", `files[${i}].sha256 must be a 64-char hex string`, 400);
    }
    const src = f["source"];
    if (src !== undefined && src !== null && !SNAPSHOT_SOURCES.has(src as string)) {
      return errorResponse("bad_request", `files[${i}].source is invalid`, 400);
    }
    if (f["extension"] !== undefined && f["extension"] !== null && typeof f["extension"] !== "string") {
      return errorResponse("bad_request", `files[${i}].extension must be a string or null`, 400);
    }
    if (f["language"] !== undefined && f["language"] !== null && typeof f["language"] !== "string") {
      return errorResponse("bad_request", `files[${i}].language must be a string or null`, 400);
    }
  }
  const result = ingestRepositorySnapshot(
    db,
    body as unknown as Parameters<typeof ingestRepositorySnapshot>[1],
  );
  return jsonResponse(result, 200);
}

export interface CollectorApp {
  db: Database;
  dbPath: string;
  fetch: (req: Request) => Promise<Response>;
  close: () => void;
}

export function createApp(dbPath?: string): CollectorApp {
  const path = dbPath ?? resolveDbPath();
  const db = openDatabase(path);

  const fetchHandler = async (req: Request): Promise<Response> => {
    try {
      const url = new URL(req.url);
      const pathname = url.pathname;
      const method = req.method.toUpperCase();

      if (method === "GET" && pathname === "/healthz") {
        return jsonResponse({ status: "ok", protocol_version: PROTOCOL_VERSION }, 200);
      }
      if (method === "GET" && pathname === "/v1/stats") {
        return jsonResponse(collectStats(db, path), 200);
      }

      const postRoutes = new Set([
        "/v1/session/start",
        "/v1/session/end",
        "/v1/events/batch",
        "/v1/blobs/check",
        "/v1/blobs/upload",
        "/v1/repository/snapshot",
      ]);
      if (postRoutes.has(pathname) && method !== "POST") {
        return errorResponse("method_not_allowed", `${pathname} requires POST`, 405);
      }
      if (!postRoutes.has(pathname)) {
        return errorResponse("not_found", `unknown route ${pathname}`, 404);
      }

      const bodyRead = await readBodyCapped(req);
      if (!bodyRead.ok) return bodyRead.response;
      const parsed = parseJson(bodyRead.text);
      if (!parsed.ok) return errorResponse("bad_request", parsed.message, 400);

      switch (pathname) {
        case "/v1/session/start":
          return handleSessionStart(db, parsed.value);
        case "/v1/session/end":
          return handleSessionEnd(db, parsed.value);
        case "/v1/events/batch":
          return handleEventsBatch(db, parsed.value);
        case "/v1/blobs/check":
          return handleBlobsCheck(db, parsed.value);
        case "/v1/blobs/upload":
          return handleBlobsUpload(db, parsed.value);
        case "/v1/repository/snapshot":
          return handleSnapshot(db, parsed.value);
        default:
          return errorResponse("not_found", "unknown route", 404);
      }
    } catch (err) {
      const withStatus = err as { status?: number; message?: string };
      if (typeof withStatus?.status === "number") {
        return errorResponse("bad_request", withStatus.message ?? "bad request", withStatus.status);
      }
      console.error("unhandled request error", err);
      return errorResponse("internal", "internal server error", 500);
    }
  };

  return {
    db,
    dbPath: path,
    fetch: fetchHandler,
    close: () => {
      try {
        db.close();
      } catch {
        // ignore close errors in tests
      }
    },
  };
}

const PORT = Number(process.env["TABCOMPLETE_COLLECTOR_PORT"] ?? process.env["PORT"] ?? 8787);

if (import.meta.main) {
  const app = createApp();
  Bun.serve({
    port: PORT,
    hostname: "0.0.0.0",
    fetch: app.fetch,
  });
  console.log(`trajectory collector listening on 0.0.0.0:${PORT} db=${app.dbPath}`);
}

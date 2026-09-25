export const PROTOCOL_VERSION = 1;

export const MAX_BODY_BYTES = 10 * 1024 * 1024;

export const EVENT_TYPES = [
  "session_start",
  "session_end",
  "repo_snapshot",
  "buffer_open",
  "buffer_close",
  "buffer_enter",
  "buffer_leave",
  "buffer_write",
  "edit_delta",
  "cursor_move",
  "file_jump",
  "mode_change",
  "key",
  "heartbeat",
  "prediction_requested",
  "prediction_shown",
  "prediction_accepted",
  "prediction_partially_accepted",
  "prediction_rejected",
  "prediction_generated",
  "prediction_dismissed",
] as const;

export type EventType = (typeof EVENT_TYPES)[number];

export const FILE_VERSION_SOURCES = [
  "repo_scan",
  "buffer_open",
  "buffer_write",
  "session_anchor",
  "periodic_anchor",
] as const;

export type FileVersionSource = (typeof FILE_VERSION_SOURCES)[number];

export interface Cursor {
  row: number;
  col: number;
}

export interface EditDeltaPayload {
  start_row: number;
  old_end_row: number;
  new_end_row: number;
  deleted_text: string;
  inserted_text: string;
  cursor_after: Cursor;
}

export interface EventEnvelope {
  protocol_version: number;
  event_id: string;
  session_id: string;
  sequence_number: number;
  timestamp_ms: number;
  event_type: EventType;
  file_id?: number | null;
  changedtick?: number | null;
  cursor?: Cursor | null;
  mode?: string | null;
  payload?: unknown;
}

export interface SessionStartRequest {
  protocol_version: number;
  session_id: string;
  machine_id: string;
  repo_id?: string | null;
  started_at: string;
  cwd_name?: string | null;
  git_head_at_start?: string | null;
  git_branch_at_start?: string | null;
  editor?: string | null;
  editor_version?: string | null;
  plugin_version?: string | null;
  // Machine-level extras (upserted into machines table).
  platform?: string | null;
  hostname_hash?: string | null;
  // Optional repo stub fields so FK stays satisfied before snapshot.
  root_name?: string | null;
  root_identity_hash?: string | null;
  origin_url?: string | null;
}

export interface SessionEndRequest {
  protocol_version: number;
  session_id: string;
  ended_at?: string | null;
}

export interface EventsBatchRequest {
  protocol_version: number;
  events: EventEnvelope[];
}

export interface BlobCheckRequest {
  hashes: string[];
}

export interface BlobUploadRequest {
  sha256: string;
  content_base64: string;
  mime_type?: string | null;
  already_compressed?: boolean | null;
}

export interface SnapshotFileEntry {
  relative_path: string;
  extension?: string | null;
  language?: string | null;
  sha256: string;
  observed_at?: string | null;
  source?: FileVersionSource | null;
}

export interface RepositorySnapshotRequest {
  protocol_version: number;
  repo_id: string;
  root_name?: string | null;
  root_identity_hash?: string | null;
  origin_url?: string | null;
  observed_at?: string | null;
  default_source?: FileVersionSource | null;
  files: SnapshotFileEntry[];
}

export interface ApiErrorBody {
  error: string;
  message: string;
}

const EVENT_TYPE_SET = new Set<string>(EVENT_TYPES);
const SOURCE_SET = new Set<string>(FILE_VERSION_SOURCES);

export function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

export function isNonEmptyString(v: unknown): v is string {
  return typeof v === "string" && v.length > 0;
}

export function isValidEventType(v: unknown): v is EventType {
  return typeof v === "string" && EVENT_TYPE_SET.has(v);
}

export function isValidSource(v: unknown): v is FileVersionSource {
  return typeof v === "string" && SOURCE_SET.has(v);
}

export function isValidSha256(v: unknown): v is string {
  return typeof v === "string" && /^[0-9a-f]{64}$/i.test(v);
}

export function isNonNegativeInt(v: unknown): v is number {
  return typeof v === "number" && Number.isInteger(v) && v >= 0;
}

export function validateCursor(
  cursor: unknown,
): { ok: true; value: Cursor | null } | { ok: false; message: string } {
  if (cursor === undefined || cursor === null) return { ok: true, value: null };
  if (!isRecord(cursor)) return { ok: false, message: "cursor must be an object" };
  const { row, col } = cursor;
  if (!isNonNegativeInt(row) || !isNonNegativeInt(col)) {
    return { ok: false, message: "cursor.row and cursor.col must be zero-based non-negative integers" };
  }
  return { ok: true, value: { row, col } };
}

export function validateEditDeltaPayload(
  payload: unknown,
): { ok: true } | { ok: false; message: string } {
  if (!isRecord(payload)) return { ok: false, message: "edit_delta payload must be an object" };
  const { start_row, old_end_row, new_end_row, deleted_text, inserted_text, cursor_after } = payload;
  if (!isNonNegativeInt(start_row)) {
    return { ok: false, message: "edit_delta.start_row must be a zero-based non-negative integer" };
  }
  if (!isNonNegativeInt(old_end_row)) {
    return { ok: false, message: "edit_delta.old_end_row must be a zero-based non-negative integer" };
  }
  if (!isNonNegativeInt(new_end_row)) {
    return { ok: false, message: "edit_delta.new_end_row must be a zero-based non-negative integer" };
  }
  if (old_end_row < start_row) {
    return { ok: false, message: "edit_delta.old_end_row must be >= start_row" };
  }
  if (new_end_row < start_row) {
    return { ok: false, message: "edit_delta.new_end_row must be >= start_row" };
  }
  if (typeof deleted_text !== "string" || typeof inserted_text !== "string") {
    return { ok: false, message: "edit_delta.deleted_text and inserted_text must be strings" };
  }
  const cursorCheck = validateCursor(cursor_after);
  if (!cursorCheck.ok || cursorCheck.value === null) {
    return { ok: false, message: "edit_delta.cursor_after must be {row,col} with zero-based integers" };
  }
  return { ok: true };
}

export function validateEventEnvelope(
  ev: unknown,
): { ok: true; value: EventEnvelope } | { ok: false; message: string } {
  if (!isRecord(ev)) return { ok: false, message: "event must be an object" };
  if (ev["protocol_version"] !== undefined && ev["protocol_version"] !== PROTOCOL_VERSION) {
    return { ok: false, message: `event protocol_version must be ${PROTOCOL_VERSION}` };
  }
  const { event_id, session_id, sequence_number, timestamp_ms, event_type } = ev;
  if (!isNonEmptyString(event_id)) return { ok: false, message: "event_id must be a non-empty string" };
  if (!isNonEmptyString(session_id)) return { ok: false, message: "session_id must be a non-empty string" };
  if (!isNonNegativeInt(sequence_number)) {
    return { ok: false, message: "sequence_number must be a non-negative integer" };
  }
  if (typeof timestamp_ms !== "number" || !Number.isInteger(timestamp_ms) || timestamp_ms < 0) {
    return { ok: false, message: "timestamp_ms must be a non-negative integer" };
  }
  if (!isValidEventType(event_type)) {
    return { ok: false, message: `event_type must be one of: ${EVENT_TYPES.join(",")}` };
  }
  const fileIdRaw = ev["file_id"];
  // file_id references files(file_id) which starts at 1; accept any integer.
  // Legacy string file identifiers (e.g. "repo:<root>:<rel>" from editor
  // clients that predate server-assigned integer ids) are accepted for
  // forward-compat and stored as NULL; file attribution then comes from
  // payload.path / anchor content_hash.
  let fileIdValue: number | null = null;
  if (fileIdRaw !== undefined && fileIdRaw !== null) {
    if (typeof fileIdRaw === "string") {
      fileIdValue = null;
    } else if (typeof fileIdRaw === "number" && Number.isInteger(fileIdRaw)) {
      fileIdValue = fileIdRaw;
    } else {
      return { ok: false, message: "file_id must be an integer, string, or null" };
    }
  }
  const changedtick = ev["changedtick"];
  if (changedtick !== undefined && changedtick !== null) {
    if (typeof changedtick !== "number" || !Number.isInteger(changedtick)) {
      return { ok: false, message: "changedtick must be an integer or null" };
    }
  }
  const cursorCheck = validateCursor(ev["cursor"]);
  if (!cursorCheck.ok) return { ok: false, message: cursorCheck.message };
  const mode = ev["mode"];
  if (mode !== undefined && mode !== null && typeof mode !== "string") {
    return { ok: false, message: "mode must be a string or null" };
  }
  if (event_type === "edit_delta") {
    const payloadCheck = validateEditDeltaPayload(ev["payload"]);
    if (!payloadCheck.ok) return { ok: false, message: payloadCheck.message };
  }
  return {
    ok: true,
    value: {
      protocol_version: PROTOCOL_VERSION,
      event_id: event_id as string,
      session_id: session_id as string,
      sequence_number: sequence_number as number,
      timestamp_ms: timestamp_ms as number,
      event_type: event_type as EventType,
      file_id: fileIdValue,
      changedtick: (changedtick ?? null) as number | null,
      cursor: cursorCheck.value,
      mode: (mode ?? null) as string | null,
      payload: ev["payload"] ?? null,
    },
  };
}

/** Remove userinfo credentials from an origin URL. Never store or log credentials. */
export function sanitizeOriginUrl(raw: unknown): string | null {
  if (raw === undefined || raw === null) return null;
  if (typeof raw !== "string") return null;
  const trimmed = raw.trim();
  if (trimmed.length === 0) return null;
  try {
    const parsed = new URL(trimmed);
    if (parsed.username !== "" || parsed.password !== "") {
      parsed.username = "";
      parsed.password = "";
    }
    return parsed.toString();
  } catch {
    // Not a parseable absolute URL (e.g. scp-like git remote). Strip
    // any `user:pass@` or `user@` prefix before the host heuristically
    // without ever echoing credentials.
    const atIndex = trimmed.lastIndexOf("@");
    if (atIndex >= 0) {
      const before = trimmed.slice(0, atIndex);
      // Only strip if the prefix looks like userinfo (no slashes or colons
      // indicating a path). Handles `user:pass@host:...` and `user@host/...`.
      if (!before.includes("/") && before.length < 256) {
        const after = trimmed.slice(atIndex + 1);
        // For scp-like syntax the host part precedes ':' or '/'; keep it.
        if (after.length > 0) {
          const colonSlash = after.search(/[/:]/);
          if (colonSlash === -1) return after;
          // Reconstruct without userinfo: drop everything up to '@'.
          // Preserve scp-like remainder.
          const prefixUpToAt = trimmed.slice(0, trimmed.indexOf("@"));
          void prefixUpToAt;
          return after;
        }
      }
    }
    return trimmed;
  }
}
